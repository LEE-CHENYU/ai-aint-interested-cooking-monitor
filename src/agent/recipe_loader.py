"""
Recipe loader.

Loads recipes from YAML files and validates them with Pydantic models.
Falls back to LLM generation for unknown dishes (cached after first call).
"""

import logging
from pathlib import Path

import yaml

from src.agent.schemas import Recipe, RecipeStep, CompletionType

logger = logging.getLogger(__name__)


def load_recipe(path: str) -> Recipe:
    """Load a recipe from a YAML file and validate it."""
    with open(path) as f:
        data = yaml.safe_load(f)

    steps = []
    for s in data["steps"]:
        steps.append(RecipeStep(
            id=s["id"],
            instruction=s["instruction"],
            completion_type=CompletionType(s["completion_type"]),
            vlm_signal=s.get("vlm_signal"),
            timer_seconds=s.get("timer_seconds"),
            depends_on=s.get("depends_on", []),
            parallel_group=s.get("parallel_group"),
        ))

    recipe = Recipe(
        dish=data["dish"],
        servings=data.get("servings", 2),
        estimated_time_minutes=data.get("estimated_time_minutes", 30),
        steps=steps,
    )
    logger.info("Loaded recipe '%s' with %d steps", recipe.dish, len(recipe.steps))
    return recipe


def _save_recipe(recipe: Recipe, path: str):
    """Write a Recipe back to YAML."""
    data = {
        "dish": recipe.dish,
        "servings": recipe.servings,
        "estimated_time_minutes": recipe.estimated_time_minutes,
        "steps": [
            {
                "id": s.id,
                "instruction": s.instruction,
                "completion_type": s.completion_type.value,
                **({"vlm_signal": s.vlm_signal} if s.vlm_signal else {}),
                **({"timer_seconds": s.timer_seconds} if s.timer_seconds else {}),
                "depends_on": s.depends_on,
                **({"parallel_group": s.parallel_group} if s.parallel_group else {}),
            }
            for s in recipe.steps
        ],
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    logger.info("Saved recipe to %s", path)


def get_or_generate_recipe(dish_name: str, recipes_dir: str = "configs/recipes") -> Recipe:
    """
    Check for a cached recipe YAML; fall back to LLM generation.

    The generated recipe is saved so subsequent runs use the cache.
    """
    slug = dish_name.lower().replace(" ", "_")
    path = Path(recipes_dir) / f"{slug}.yaml"

    if path.exists():
        logger.info("Found cached recipe at %s", path)
        return load_recipe(str(path))

    # Generate via LLM
    logger.info("No cached recipe for '%s', generating via LLM...", dish_name)
    from src.agent.model_loader import KitchenPolicyModel

    model = KitchenPolicyModel()
    model.load()

    prompt_text = (
        f"Generate a cooking recipe for '{dish_name}' as a YAML structure. "
        f"Each step needs: id (int), instruction (str), "
        f"completion_type (vlm|timer|user_confirm), "
        f"vlm_signal (str, only for vlm steps), "
        f"timer_seconds (int, only for timer steps), "
        f"depends_on (list of step ids). "
        f"Return ONLY valid YAML."
    )
    raw = model.generate({"request": prompt_text})

    # Try to parse the generated YAML
    try:
        data = yaml.safe_load(raw)
        if "steps" not in data:
            raise ValueError("No steps in generated recipe")
        recipe = Recipe(
            dish=dish_name,
            steps=[
                RecipeStep(
                    id=s["id"],
                    instruction=s["instruction"],
                    completion_type=CompletionType(s["completion_type"]),
                    vlm_signal=s.get("vlm_signal"),
                    timer_seconds=s.get("timer_seconds"),
                    depends_on=s.get("depends_on", []),
                )
                for s in data["steps"]
            ],
        )
        _save_recipe(recipe, str(path))
        return recipe
    except Exception as e:
        logger.error("Failed to parse generated recipe: %s", e)
        raise
