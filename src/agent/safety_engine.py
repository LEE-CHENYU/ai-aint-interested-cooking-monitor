"""
Deterministic safety engine.

Evaluates hard-coded safety rules from configs/safety_rules.yaml against
the current world state.  This is a hybrid approach: the model composes
most responses, but a deterministic engine guarantees critical hazards are
never missed (e.g., smoke → alert even if the model returns empty actions).

Condition DSL examples from the config:
    "stove.pot_present AND NOT counter.hands_active"
    "stove.smoke_suspected"
    "stove.water_state == boiling AND stove.steam_level == high"
"""

import logging
import time
from dataclasses import dataclass, field

import yaml

logger = logging.getLogger(__name__)


@dataclass
class SafetyViolation:
    """A safety rule that is currently violated."""
    rule_id: str
    description: str
    severity: str       # low, high, critical
    action: str         # speak or show_card
    message: str
    continuous_since: float | None = None  # for timeout-based rules


@dataclass
class SafetyRule:
    id: str
    description: str
    condition: str
    severity: str
    action: str
    message: str
    timeout_seconds: int = 0
    cooldown_seconds: int = 30


class SafetyEngine:
    """Evaluates safety rules against the flat world-state snapshot."""

    def __init__(self, rules_path: str = "configs/safety_rules.yaml"):
        self.rules: list[SafetyRule] = []
        # Tracks when each timeout-based rule first became true continuously
        self._violation_start: dict[str, float] = {}
        self._load_rules(rules_path)

    def _load_rules(self, path: str):
        with open(path) as f:
            data = yaml.safe_load(f)

        for r in data["rules"]:
            self.rules.append(SafetyRule(
                id=r["id"],
                description=r["description"],
                condition=r["condition"],
                severity=r["severity"],
                action=r["action"],
                message=r["message"],
                timeout_seconds=r.get("timeout_seconds", 0),
                cooldown_seconds=r.get("cooldown_seconds", 30),
            ))
        logger.info("Loaded %d safety rules", len(self.rules))

    # ------------------------------------------------------------------
    # Condition DSL evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_value(token: str, flat_state: dict):
        """
        Resolve a dotted reference like 'stove.pot_present' from the flat state.

        The flat state has keys like:
            {"stove.pot_present": True, "stove.water_state": "boiling", ...}
        """
        return flat_state.get(token)

    def _eval_condition(self, condition: str, flat_state: dict) -> bool:
        """
        Evaluate a simple boolean DSL condition against the flat state.

        Supported operators: AND, OR, NOT, ==
        Examples:
            "stove.pot_present AND NOT counter.hands_active"
            "stove.smoke_suspected"
            "stove.water_state == boiling AND stove.steam_level == high"
        """
        # Tokenize by spaces
        tokens = condition.split()
        # Build a Python expression string
        expr_parts = []
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "AND":
                expr_parts.append("and")
            elif tok == "OR":
                expr_parts.append("or")
            elif tok == "NOT":
                expr_parts.append("not")
            elif tok == "==":
                # Comparison: previous part was a ref, next is the value
                ref = expr_parts.pop()  # the Python expression for the ref
                i += 1
                if i >= len(tokens):
                    return False
                expected = tokens[i]
                expr_parts.append(f"({ref} == {expected!r})")
            else:
                # It's a dotted reference like "stove.pot_present"
                val = self._resolve_value(tok, flat_state)
                if val is None:
                    # Signal not present → treat as falsy
                    expr_parts.append("False")
                elif isinstance(val, bool):
                    expr_parts.append(str(val))
                elif isinstance(val, str):
                    expr_parts.append(repr(val))
                else:
                    expr_parts.append(str(val))
            i += 1

        expr_str = " ".join(expr_parts)
        try:
            return bool(eval(expr_str))  # noqa: S307  — safe, only our literals
        except Exception:
            logger.warning("Failed to evaluate safety condition: %s -> %s", condition, expr_str)
            return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_all(self, flat_state: dict, world_state) -> list[SafetyViolation]:
        """
        Evaluate all rules against the current flat state.

        Returns a list of active violations (respecting cooldowns and timeouts).
        """
        now = time.time()
        violations: list[SafetyViolation] = []

        for rule in self.rules:
            triggered = self._eval_condition(rule.condition, flat_state)

            if not triggered:
                # Reset continuous violation tracker
                self._violation_start.pop(rule.id, None)
                continue

            # For timeout-based rules, only fire after continuous violation
            if rule.timeout_seconds > 0:
                if rule.id not in self._violation_start:
                    self._violation_start[rule.id] = now
                elapsed = now - self._violation_start[rule.id]
                if elapsed < rule.timeout_seconds:
                    continue  # Not long enough yet

            # Respect cooldown
            if not world_state.can_alert(rule.id, rule.cooldown_seconds):
                continue

            violations.append(SafetyViolation(
                rule_id=rule.id,
                description=rule.description,
                severity=rule.severity,
                action=rule.action,
                message=rule.message,
                continuous_since=self._violation_start.get(rule.id),
            ))

        return violations

    def escalate_if_needed(
        self, violations: list[SafetyViolation], model_actions: list
    ) -> list[dict]:
        """
        If the model did NOT address a critical/high violation, force an action.

        Returns a list of forced actions (speak tool calls) to execute.
        """
        if not violations:
            return []

        # Collect priorities the model already addressed
        model_speak_texts = set()
        for action in model_actions:
            if getattr(action, "tool", None) == "speak":
                model_speak_texts.add(getattr(action, "text", ""))

        forced: list[dict] = []
        for v in violations:
            if v.severity not in ("critical", "high"):
                continue
            # Check if model already produced a speak about this
            already_handled = any(
                v.rule_id in text or v.message[:20] in text
                for text in model_speak_texts
            )
            if already_handled:
                continue

            logger.warning("Safety escalation for %s: %s", v.rule_id, v.message)
            forced.append({
                "tool": v.action,
                "text": v.message,
                "priority": v.severity,
                "source": "safety_engine",
            })

        return forced
