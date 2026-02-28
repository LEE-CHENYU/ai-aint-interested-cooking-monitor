# Agentic Cooking Workflow Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the end-to-end agentic cooking workflow: user gives a dish name, agent generates recipe, guides user step-by-step using VLM detection + timers + user confirmation, outputs via console + TTS.

**Architecture:** Inference-driven loop. LLM generates recipe once upfront (cached as YAML). Each cycle: load synthetic image → VLM inference → update state → rule-based step transitions → announce next step. No second LLM call per cycle.

**Tech Stack:** Python 3.10+, Pydantic (schemas), PyYAML (recipes), OpenCV (image loading), Transformers (Gemma 3 VLM), pyttsx3 or subprocess `say` (TTS), pytest (tests).

**Design doc:** `docs/plans/2026-02-28-agentic-workflow-design.md`

---

## Task 1: Recipe Step Schema

Add the new `RecipeStep` Pydantic model and a `Recipe` container to `src/agent/schemas.py`. This is the foundation — every other component depends on this schema.

**Files:**
- Modify: `src/agent/schemas.py`
- Create: `tests/test_schemas.py`

**Step 1: Write the failing test**

```python
# tests/test_schemas.py
import pytest
from src.agent.schemas import RecipeStep, Recipe


def test_recipe_step_vlm():
    step = RecipeStep(
        id=1,
        instruction="Dice the tofu into 1-inch cubes",
        completion_type="vlm",
        vlm_signal="diced tofu on cutting board",
    )
    assert step.completion_type == "vlm"
    assert step.vlm_signal == "diced tofu on cutting board"
    assert step.timer_seconds is None
    assert step.depends_on == []
    assert step.parallel_group is None
    assert step.status == "pending"


def test_recipe_step_timer():
    step = RecipeStep(
        id=4,
        instruction="Add doubanjiang and cook for 30 seconds",
        completion_type="timer",
        timer_seconds=30,
        depends_on=[3],
        parallel_group="A",
    )
    assert step.timer_seconds == 30
    assert step.depends_on == [3]
    assert step.parallel_group == "A"


def test_recipe_step_user_confirm():
    step = RecipeStep(
        id=2,
        instruction="Mince garlic, ginger, and scallions",
        completion_type="user_confirm",
    )
    assert step.completion_type == "user_confirm"


def test_recipe_step_invalid_completion_type():
    with pytest.raises(Exception):
        RecipeStep(id=1, instruction="test", completion_type="invalid")


def test_recipe():
    recipe = Recipe(
        dish="mapo_tofu",
        servings=2,
        estimated_time_minutes=25,
        steps=[
            RecipeStep(id=1, instruction="Step 1", completion_type="vlm", vlm_signal="sig"),
            RecipeStep(id=2, instruction="Step 2", completion_type="timer", timer_seconds=30, depends_on=[1]),
        ],
    )
    assert recipe.dish == "mapo_tofu"
    assert len(recipe.steps) == 2
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_schemas.py -v`
Expected: FAIL with ImportError (RecipeStep not defined yet)

**Step 3: Write minimal implementation**

Add to `src/agent/schemas.py` (after existing classes, before `AgentResponse`):

```python
class RecipeStep(BaseModel):
    """A single step in a cooking recipe."""
    id: int
    instruction: str = Field(description="What to tell the user")
    completion_type: str = Field(
        description="How to detect step completion",
        pattern="^(vlm|timer|user_confirm)$",
    )
    vlm_signal: str | None = Field(default=None, description="What VLM should look for")
    timer_seconds: int | None = Field(default=None, description="Duration for timer-based steps")
    depends_on: list[int] = Field(default_factory=list, description="Step IDs that must complete first")
    parallel_group: str | None = Field(default=None, description="Group ID for parallel steps")
    status: str = Field(default="pending", pattern="^(pending|active|done)$")


class Recipe(BaseModel):
    """A complete cooking recipe with ordered steps."""
    dish: str
    servings: int = 2
    estimated_time_minutes: int
    steps: list[RecipeStep]
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_schemas.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/agent/schemas.py tests/test_schemas.py
git commit -m "feat: add RecipeStep and Recipe Pydantic schemas"
```

---

## Task 2: Recipe Loader (YAML → Recipe Object)

Create a module that loads recipes from YAML files and converts to the `Recipe` Pydantic model. This handles the "cached" path. LLM generation (the "uncached" path) comes in Task 3.

**Files:**
- Create: `src/agent/recipe_loader.py`
- Create: `tests/test_recipe_loader.py`
- Modify: `configs/recipes/pasta_and_sauce.yaml` (update to new schema format)

**Step 1: Create a test recipe YAML in the new schema format**

Update `configs/recipes/pasta_and_sauce.yaml` to match the new schema:

```yaml
dish: pasta_and_sauce
servings: 2
estimated_time_minutes: 25

steps:
  - id: 1
    instruction: "Fill a large pot with water and bring to a rolling boil"
    completion_type: vlm
    vlm_signal: "pot of water at rolling boil on stove"
    depends_on: []

  - id: 2
    instruction: "Dice the onion and mince the garlic"
    completion_type: user_confirm
    depends_on: []

  - id: 3
    instruction: "Add pasta to the boiling water and stir"
    completion_type: timer
    timer_seconds: 480
    depends_on: [1]

  - id: 4
    instruction: "Saute garlic and onion in olive oil, then add tomato sauce"
    completion_type: timer
    timer_seconds: 600
    depends_on: [2]
    parallel_group: A

  - id: 5
    instruction: "Drain the pasta in a colander"
    completion_type: user_confirm
    depends_on: [3]

  - id: 6
    instruction: "Combine drained pasta with sauce and toss to coat"
    completion_type: user_confirm
    depends_on: [4, 5]
```

**Step 2: Write the failing test**

```python
# tests/test_recipe_loader.py
import os
import tempfile
import pytest
import yaml
from src.agent.recipe_loader import load_recipe
from src.agent.schemas import Recipe


def test_load_recipe_from_yaml():
    recipe = load_recipe("configs/recipes/pasta_and_sauce.yaml")
    assert isinstance(recipe, Recipe)
    assert recipe.dish == "pasta_and_sauce"
    assert len(recipe.steps) == 6
    assert recipe.steps[0].completion_type == "vlm"
    assert recipe.steps[2].timer_seconds == 480
    assert recipe.steps[2].depends_on == [1]


def test_load_recipe_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_recipe("nonexistent.yaml")


def test_load_recipe_step_defaults():
    recipe = load_recipe("configs/recipes/pasta_and_sauce.yaml")
    for step in recipe.steps:
        assert step.status == "pending"
```

**Step 3: Run test to verify it fails**

Run: `python -m pytest tests/test_recipe_loader.py -v`
Expected: FAIL (module doesn't exist)

**Step 4: Write minimal implementation**

```python
# src/agent/recipe_loader.py
"""
Recipe loader.

Loads recipes from YAML files and converts to Recipe Pydantic model.
"""

import yaml
from src.agent.schemas import Recipe, RecipeStep


def load_recipe(path: str) -> Recipe:
    """Load a recipe from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)

    steps = [RecipeStep(**s) for s in data["steps"]]
    return Recipe(
        dish=data["dish"],
        servings=data.get("servings", 2),
        estimated_time_minutes=data.get("estimated_time_minutes", 30),
        steps=steps,
    )
```

**Step 5: Run test to verify it passes**

Run: `python -m pytest tests/test_recipe_loader.py -v`
Expected: All 3 tests PASS

**Step 6: Commit**

```bash
git add src/agent/recipe_loader.py tests/test_recipe_loader.py configs/recipes/pasta_and_sauce.yaml
git commit -m "feat: add recipe loader with YAML parsing"
```

---

## Task 3: Image Sequencer

Replace the live-camera `CameraStream` concept with an `ImageSequencer` that loads images from a directory in numbered order. This feeds synthetic images to the VLM.

**Files:**
- Create: `src/perception/image_sequencer.py`
- Create: `tests/test_image_sequencer.py`

**Step 1: Write the failing test**

```python
# tests/test_image_sequencer.py
import os
import tempfile
import numpy as np
import cv2
import pytest
from src.perception.image_sequencer import ImageSequencer


@pytest.fixture
def image_dir():
    """Create a temp directory with 3 numbered test images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for i, name in enumerate(["001_cold.png", "002_simmer.png", "003_boil.png"]):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            img[:] = (i * 80, i * 80, i * 80)  # Different brightness per image
            cv2.imwrite(os.path.join(tmpdir, name), img)
        yield tmpdir


def test_sequencer_loads_in_order(image_dir):
    seq = ImageSequencer(image_dir)
    assert len(seq) == 3
    assert seq.filenames[0] == "001_cold.png"
    assert seq.filenames[2] == "003_boil.png"


def test_sequencer_next_returns_frame(image_dir):
    seq = ImageSequencer(image_dir)
    frame, filename = seq.next()
    assert frame is not None
    assert frame.shape == (100, 100, 3)
    assert filename == "001_cold.png"


def test_sequencer_advances(image_dir):
    seq = ImageSequencer(image_dir)
    _, f1 = seq.next()
    _, f2 = seq.next()
    assert f1 == "001_cold.png"
    assert f2 == "002_simmer.png"


def test_sequencer_done(image_dir):
    seq = ImageSequencer(image_dir)
    seq.next()
    seq.next()
    seq.next()
    assert seq.done


def test_sequencer_empty_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        seq = ImageSequencer(tmpdir)
        assert len(seq) == 0
        assert seq.done
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_image_sequencer.py -v`
Expected: FAIL (module doesn't exist)

**Step 3: Write minimal implementation**

```python
# src/perception/image_sequencer.py
"""
Image sequencer for demo mode.

Loads images from a directory in sorted filename order,
simulating a scripted cooking session.
"""

import os

import cv2
import numpy as np


class ImageSequencer:
    """Loads synthetic images in order from a directory."""

    def __init__(self, image_dir: str):
        self.image_dir = image_dir
        self.filenames = sorted(
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        )
        self.index = 0

    def __len__(self) -> int:
        return len(self.filenames)

    @property
    def done(self) -> bool:
        return self.index >= len(self.filenames)

    def next(self) -> tuple[np.ndarray | None, str | None]:
        """Return the next image and its filename. Returns (None, None) if done."""
        if self.done:
            return None, None
        filename = self.filenames[self.index]
        path = os.path.join(self.image_dir, filename)
        frame = cv2.imread(path)
        self.index += 1
        return frame, filename

    def reset(self):
        """Reset to the beginning of the sequence."""
        self.index = 0
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_image_sequencer.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add src/perception/image_sequencer.py tests/test_image_sequencer.py
git commit -m "feat: add ImageSequencer for scripted demo image playback"
```

---

## Task 4: Step Transition Engine

This is the core rule-based engine that checks whether the current step should transition to `done`, and advances to the next step. It does NOT call any LLM — it uses VLM detection results, timer status, and user input.

**Files:**
- Create: `src/agent/step_engine.py`
- Create: `tests/test_step_engine.py`

**Step 1: Write the failing test**

```python
# tests/test_step_engine.py
import pytest
from src.agent.schemas import RecipeStep, Recipe
from src.agent.step_engine import StepEngine


def make_recipe():
    return Recipe(
        dish="test",
        estimated_time_minutes=10,
        steps=[
            RecipeStep(id=1, instruction="Boil water", completion_type="vlm", vlm_signal="boiling water"),
            RecipeStep(id=2, instruction="Chop onion", completion_type="user_confirm", depends_on=[]),
            RecipeStep(id=3, instruction="Add pasta", completion_type="timer", timer_seconds=480, depends_on=[1]),
            RecipeStep(id=4, instruction="Make sauce", completion_type="timer", timer_seconds=300, depends_on=[2], parallel_group="A"),
            RecipeStep(id=5, instruction="Combine", completion_type="user_confirm", depends_on=[3, 4]),
        ],
    )


def test_initial_active_step():
    engine = StepEngine(make_recipe())
    active = engine.get_active_step()
    assert active is not None
    assert active.id == 1
    assert active.status == "active"


def test_dependencies_block_step():
    engine = StepEngine(make_recipe())
    step3 = engine.get_step(3)
    assert step3.status == "pending"  # blocked by step 1


def test_vlm_completes_step():
    engine = StepEngine(make_recipe())
    actions = engine.check_vlm_signal("boiling water")
    assert engine.get_step(1).status == "done"
    assert any(a["type"] == "step_done" for a in actions)


def test_vlm_no_match_keeps_step():
    engine = StepEngine(make_recipe())
    actions = engine.check_vlm_signal("cold water")
    assert engine.get_step(1).status == "active"
    assert len(actions) == 0


def test_advance_after_completion():
    engine = StepEngine(make_recipe())
    engine.check_vlm_signal("boiling water")  # completes step 1
    # Step 3 should now be eligible (depends on step 1)
    active = engine.get_active_step()
    assert active.id == 3


def test_user_confirm_completes_step():
    recipe = make_recipe()
    # Make step 2 active by setting step 1 to done so steps can progress
    engine = StepEngine(recipe)
    # Step 2 has no dependencies, so it could be active
    # First active is step 1 (lower id). Let's complete it.
    engine.check_vlm_signal("boiling water")
    # Now step 2 should still be pending with no deps...
    # actually step 2 has no deps, so after step 1 is done,
    # step 3 depends on [1] and step 2 has no deps.
    # The engine should activate step 3 next (first unblocked by id).
    # Step 2 also has no deps so it could also be active.
    # For user_confirm, the user explicitly confirms:
    engine.user_confirm(2)
    assert engine.get_step(2).status == "done"


def test_timer_starts_on_activation():
    engine = StepEngine(make_recipe())
    engine.check_vlm_signal("boiling water")  # completes step 1, step 3 becomes active
    active = engine.get_active_step()
    assert active.id == 3
    assert active.completion_type == "timer"
    actions = engine.get_pending_actions()
    assert any(a["type"] == "start_timer" for a in actions)


def test_timer_expiry_completes_step():
    engine = StepEngine(make_recipe())
    engine.check_vlm_signal("boiling water")  # step 1 done
    engine.check_timer_expired("step_3_timer")  # step 3 timer expired
    assert engine.get_step(3).status == "done"


def test_all_done():
    engine = StepEngine(make_recipe())
    engine.check_vlm_signal("boiling water")  # step 1
    engine.user_confirm(2)  # step 2
    engine.check_timer_expired("step_3_timer")  # step 3
    engine.check_timer_expired("step_4_timer")  # step 4
    engine.user_confirm(5)  # step 5
    assert engine.all_done


def test_safety_alert():
    engine = StepEngine(make_recipe())
    actions = engine.check_safety({"safe": False, "state": "boil_over", "reason": "liquid spilling"})
    assert any(a["type"] == "safety_alert" for a in actions)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_step_engine.py -v`
Expected: FAIL (module doesn't exist)

**Step 3: Write minimal implementation**

```python
# src/agent/step_engine.py
"""
Step transition engine.

Rule-based engine that manages recipe step progression.
Checks VLM signals, timer expiry, and user confirmations
to advance through recipe steps. No LLM calls.
"""

from src.agent.schemas import Recipe, RecipeStep


class StepEngine:
    """Manages recipe step state transitions."""

    def __init__(self, recipe: Recipe):
        self.recipe = recipe
        self.steps = {s.id: s for s in recipe.steps}
        self.active_timers: dict[str, int] = {}  # timer_name -> step_id
        self._pending_actions: list[dict] = []

        # Activate the first eligible step
        self._advance()

    def get_step(self, step_id: int) -> RecipeStep:
        return self.steps[step_id]

    def get_active_step(self) -> RecipeStep | None:
        """Return the currently active step (shown to user)."""
        for step in self.recipe.steps:
            if step.status == "active":
                return step
        return None

    @property
    def all_done(self) -> bool:
        return all(s.status == "done" for s in self.recipe.steps)

    def _deps_satisfied(self, step: RecipeStep) -> bool:
        return all(self.steps[d].status == "done" for d in step.depends_on)

    def _advance(self):
        """Activate the next eligible step after a completion."""
        if self.all_done:
            self._pending_actions.append({
                "type": "recipe_done",
                "dish": self.recipe.dish,
            })
            return

        # Find next pending step with satisfied dependencies
        for step in self.recipe.steps:
            if step.status == "pending" and self._deps_satisfied(step):
                step.status = "active"
                self._pending_actions.append({
                    "type": "step_activated",
                    "step_id": step.id,
                    "instruction": step.instruction,
                    "completion_type": step.completion_type,
                })
                # If timer step, queue a timer start
                if step.completion_type == "timer" and step.timer_seconds:
                    timer_name = f"step_{step.id}_timer"
                    self.active_timers[timer_name] = step.id
                    self._pending_actions.append({
                        "type": "start_timer",
                        "timer_name": timer_name,
                        "seconds": step.timer_seconds,
                        "step_id": step.id,
                    })
                break  # Only activate one step at a time (progressive disclosure)

    def _complete_step(self, step_id: int):
        """Mark a step as done and advance."""
        step = self.steps[step_id]
        step.status = "done"
        self._pending_actions.append({
            "type": "step_done",
            "step_id": step_id,
            "instruction": step.instruction,
        })
        self._advance()

    def check_vlm_signal(self, detected_signal: str) -> list[dict]:
        """Check if the VLM detection matches the active step's expected signal."""
        self._pending_actions = []
        active = self.get_active_step()
        if active and active.completion_type == "vlm" and active.vlm_signal:
            # Simple substring match — the VLM output should contain the expected signal
            if active.vlm_signal.lower() in detected_signal.lower():
                self._complete_step(active.id)
        return self._pending_actions

    def check_timer_expired(self, timer_name: str) -> list[dict]:
        """Handle a timer expiry event."""
        self._pending_actions = []
        if timer_name in self.active_timers:
            step_id = self.active_timers.pop(timer_name)
            self._complete_step(step_id)
        return self._pending_actions

    def user_confirm(self, step_id: int) -> list[dict]:
        """Handle user confirmation that a step is done."""
        self._pending_actions = []
        step = self.steps.get(step_id)
        if step and step.status in ("active", "pending"):
            self._complete_step(step_id)
        return self._pending_actions

    def check_safety(self, vlm_result: dict) -> list[dict]:
        """Check VLM result for safety signals."""
        self._pending_actions = []
        if not vlm_result.get("safe", True):
            self._pending_actions.append({
                "type": "safety_alert",
                "state": vlm_result.get("state", "unknown"),
                "reason": vlm_result.get("reason", "Safety concern detected"),
            })
        return self._pending_actions

    def get_pending_actions(self) -> list[dict]:
        """Return and clear any pending actions."""
        actions = self._pending_actions
        self._pending_actions = []
        return actions
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_step_engine.py -v`
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add src/agent/step_engine.py tests/test_step_engine.py
git commit -m "feat: add rule-based step transition engine"
```

---

## Task 5: VLM Detector Implementation

Implement the `EventDetector` to load a merged Gemma 3 VLM and run inference on images. Returns structured JSON matching the fine-tuned model's output format.

**Important:** This requires the merged model on disk. For local development/testing without GPU, provide a mock mode.

**Files:**
- Modify: `src/perception/detector.py`
- Create: `tests/test_detector.py`

**Step 1: Write the failing test**

```python
# tests/test_detector.py
import numpy as np
import json
import pytest
from src.perception.detector import VLMDetector


def test_mock_detector_returns_valid_json():
    """Test with mock mode (no GPU required)."""
    detector = VLMDetector(model_path=None, mock=True)
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    result = detector.detect(frame, step_signal="boiling water")
    assert "dish" in result
    assert "state" in result
    assert "safe" in result
    assert "reason" in result
    assert isinstance(result["safe"], bool)


def test_mock_detector_returns_dict():
    detector = VLMDetector(model_path=None, mock=True)
    frame = np.zeros((224, 224, 3), dtype=np.uint8)
    result = detector.detect(frame)
    assert isinstance(result, dict)
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_detector.py -v`
Expected: FAIL (VLMDetector not defined)

**Step 3: Write implementation**

Replace `src/perception/detector.py` entirely:

```python
# src/perception/detector.py
"""
VLM-based kitchen event detector.

Uses a merged Gemma 3 VLM to classify kitchen state from images.
Returns structured JSON: {dish, state, safe, reason}.

Supports mock mode for development without GPU.
"""

import json
import time

import numpy as np


SYSTEM_PROMPT = (
    "You are a kitchen safety monitor. Look at this top-down photo of a stove.\n"
    'Respond with ONLY a JSON object, no other text:\n'
    '{"dish": "<what food>", "state": "<cold|simmering|boiling|boil_over|done|burnt>",\n'
    ' "safe": true/false, "reason": "<10 words max>"}'
)

STEP_AWARE_PROMPT_TEMPLATE = (
    "You are a kitchen safety monitor. Look at this photo of a cooking scene.\n"
    "The user is currently working on: {step_signal}\n"
    "Is this step visually complete? Also check for safety issues.\n"
    'Respond with ONLY a JSON object, no other text:\n'
    '{{"dish": "<what food>", "state": "<cold|simmering|boiling|boil_over|done|burnt>",'
    ' "safe": true/false, "reason": "<10 words max>",'
    ' "step_complete": true/false}}'
)

MOCK_RESPONSES = [
    {"dish": "pasta", "state": "cold", "safe": True, "reason": "Not cooking yet", "step_complete": False},
    {"dish": "pasta", "state": "simmering", "safe": True, "reason": "Gentle simmer", "step_complete": False},
    {"dish": "pasta", "state": "boiling", "safe": True, "reason": "Active boil", "step_complete": True},
    {"dish": "pasta", "state": "boil_over", "safe": False, "reason": "Liquid spilling over", "step_complete": False},
]


class VLMDetector:
    """Detects kitchen events from images using Gemma 3 VLM."""

    def __init__(self, model_path: str | None = None, mock: bool = False):
        self.model_path = model_path
        self.mock = mock
        self.model = None
        self.processor = None
        self._mock_index = 0

        if not mock and model_path:
            self._load_model()

    def _load_model(self):
        """Load the merged Gemma 3 VLM model."""
        from transformers import AutoProcessor, AutoModelForCausalLM
        import torch

        print(f"Loading VLM from {self.model_path}...")
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        print("VLM loaded.")

    def detect(self, frame: np.ndarray, step_signal: str | None = None) -> dict:
        """Run VLM inference on a frame. Returns parsed JSON dict."""
        if self.mock:
            return self._mock_detect()

        if self.model is None:
            raise RuntimeError("Model not loaded. Provide model_path or use mock=True.")

        return self._real_detect(frame, step_signal)

    def _mock_detect(self) -> dict:
        """Return mock responses for testing without GPU."""
        result = MOCK_RESPONSES[self._mock_index % len(MOCK_RESPONSES)]
        self._mock_index += 1
        return dict(result)  # Return a copy

    def _real_detect(self, frame: np.ndarray, step_signal: str | None = None) -> dict:
        """Run actual VLM inference."""
        import torch
        from PIL import Image

        image = Image.fromarray(frame[:, :, ::-1])  # BGR to RGB

        if step_signal:
            prompt = STEP_AWARE_PROMPT_TEMPLATE.format(step_signal=step_signal)
        else:
            prompt = SYSTEM_PROMPT

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}
        ]

        input_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=input_text, images=[image], return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs, max_new_tokens=200, temperature=0.1, do_sample=True
            )

        response = self.processor.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        return self._parse_json(response)

    def _parse_json(self, response: str) -> dict:
        """Extract JSON from model response."""
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > 0:
                return json.loads(response[start:end])
        except (json.JSONDecodeError, ValueError):
            pass
        return {"dish": "unknown", "state": "unknown", "safe": True, "reason": "parse error", "step_complete": False}
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_detector.py -v`
Expected: All 2 tests PASS

**Step 5: Commit**

```bash
git add src/perception/detector.py tests/test_detector.py
git commit -m "feat: implement VLM detector with mock mode for testing"
```

---

## Task 6: TTS Action Executor

Wire the `speak` action in `tools.py` to actually produce audio output using `subprocess` with the system `say` command (macOS) or `pyttsx3` as fallback.

**Files:**
- Create: `src/agent/tts.py`
- Modify: `src/agent/tools.py`
- Create: `tests/test_tts.py`

**Step 1: Write the failing test**

```python
# tests/test_tts.py
import pytest
from src.agent.tts import TTSEngine


def test_tts_console_mode():
    """Test TTS in console-only mode (no audio)."""
    tts = TTSEngine(audio=False)
    tts.speak("Hello world", priority="medium")
    assert tts.last_spoken == "Hello world"


def test_tts_priority_formatting():
    tts = TTSEngine(audio=False)
    tts.speak("Fire!", priority="critical")
    assert tts.last_spoken == "Fire!"
    assert tts.last_priority == "critical"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tts.py -v`
Expected: FAIL (module doesn't exist)

**Step 3: Write implementation**

```python
# src/agent/tts.py
"""
Text-to-speech engine.

Uses macOS `say` command for audio output.
Falls back to console-only mode when audio is unavailable.
"""

import subprocess
import sys


class TTSEngine:
    """Speaks text aloud and prints to console."""

    def __init__(self, audio: bool = True):
        self.audio = audio
        self.last_spoken: str | None = None
        self.last_priority: str | None = None

    def speak(self, text: str, priority: str = "medium"):
        """Speak text aloud and print to console."""
        self.last_spoken = text
        self.last_priority = priority

        # Console output with priority prefix
        prefix = {
            "low": "   ",
            "medium": ">> ",
            "high": "!! ",
            "critical": "** WARNING ** ",
        }.get(priority, ">> ")
        print(f"{prefix}{text}")

        # Audio output
        if self.audio:
            self._say(text, priority)

    def _say(self, text: str, priority: str):
        """Use system TTS to speak."""
        if sys.platform == "darwin":
            rate = "180" if priority == "critical" else "200"
            try:
                subprocess.Popen(
                    ["say", "-r", rate, text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                pass  # `say` not available
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_tts.py -v`
Expected: All 2 tests PASS

**Step 5: Update tools.py to use TTSEngine**

In `src/agent/tools.py`, update the `speak` case:

```python
# In ToolExecutor.__init__, add tts parameter:
def __init__(self, world_state, ui_server=None, tts=None):
    self.world_state = world_state
    self.ui_server = ui_server
    self.tts = tts

# In _execute_one, update speak case:
case "speak":
    if self.tts:
        self.tts.speak(action.text, action.priority)
    else:
        print(f"[SPEAK:{action.priority}] {action.text}")
    return {"tool": "speak", "status": "ok"}
```

**Step 6: Commit**

```bash
git add src/agent/tts.py src/agent/tools.py tests/test_tts.py
git commit -m "feat: add TTS engine with macOS say support"
```

---

## Task 7: Update World State for New Recipe Schema

Update `src/world_state/state.py` to work with the new `Recipe` / `RecipeStep` Pydantic models instead of the old dataclass-based `RecipeStep`.

**Files:**
- Modify: `src/world_state/state.py`
- Create: `tests/test_world_state.py`

**Step 1: Write the failing test**

```python
# tests/test_world_state.py
import pytest
from src.agent.schemas import Recipe, RecipeStep
from src.world_state.state import WorldState


def make_test_recipe():
    return Recipe(
        dish="test_dish",
        estimated_time_minutes=10,
        steps=[
            RecipeStep(id=1, instruction="Step 1", completion_type="vlm", vlm_signal="sig"),
            RecipeStep(id=2, instruction="Step 2", completion_type="timer", timer_seconds=60, depends_on=[1]),
        ],
    )


def test_world_state_init_with_recipe():
    recipe = make_test_recipe()
    ws = WorldState(recipe=recipe)
    assert ws.recipe.dish == "test_dish"
    assert len(ws.recipe.steps) == 2


def test_world_state_snapshot_includes_recipe():
    recipe = make_test_recipe()
    ws = WorldState(recipe=recipe)
    snapshot = ws.get_snapshot()
    assert "recipe" in snapshot
    assert snapshot["recipe"]["dish"] == "test_dish"
    assert len(snapshot["recipe"]["steps"]) == 2


def test_world_state_update_zone():
    ws = WorldState()
    ws.update_zone("stove", {"water_state": "boiling"}, {"water_state": 0.9})
    assert ws.zone_signals["stove"]["signals"]["water_state"] == "boiling"
```

**Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_world_state.py -v`
Expected: FAIL (WorldState doesn't accept recipe= parameter)

**Step 3: Update world_state/state.py**

Replace `src/world_state/state.py`:

```python
"""
World state aggregation.

Maintains the complete kitchen state by combining:
- Zone signals from perception (per-camera detections)
- Active timers from TimerEngine
- Recipe and step tracking
- Alert cooldowns (anti-nag)
"""

import time

import yaml

from src.agent.schemas import Recipe, RecipeStep
from src.world_state.timer_engine import TimerEngine


class WorldState:
    """Aggregates all kitchen state for the agent."""

    def __init__(self, recipe: Recipe | None = None, recipe_path: str | None = None):
        self.timer_engine = TimerEngine()
        self.zone_signals: dict[str, dict] = {}
        self.recipe: Recipe | None = recipe
        self.last_alert_times: dict[str, float] = {}

        if recipe_path and not recipe:
            self._load_recipe(recipe_path)

    def _load_recipe(self, path: str):
        from src.agent.recipe_loader import load_recipe
        self.recipe = load_recipe(path)

    def update_zone(self, zone: str, signals: dict, confidence: dict):
        """Update signals from a perception zone."""
        self.zone_signals[zone] = {
            "signals": signals,
            "confidence": confidence,
            "ts": time.time(),
        }

    def can_alert(self, rule_id: str, cooldown_seconds: float) -> bool:
        last = self.last_alert_times.get(rule_id, 0)
        return (time.time() - last) >= cooldown_seconds

    def record_alert(self, rule_id: str):
        self.last_alert_times[rule_id] = time.time()

    def get_snapshot(self) -> dict:
        """Build a compact state snapshot for the agent prompt."""
        snapshot = {
            "zones": self.zone_signals,
            "timers": self.timer_engine.to_dict(),
            "expired_timers": [t.to_dict() for t in self.timer_engine.get_expired()],
        }
        if self.recipe:
            snapshot["recipe"] = {
                "dish": self.recipe.dish,
                "steps": [
                    {"id": s.id, "instruction": s.instruction, "status": s.status}
                    for s in self.recipe.steps
                ],
            }
        return snapshot
```

**Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_world_state.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/world_state/state.py tests/test_world_state.py
git commit -m "feat: update WorldState to use new Recipe schema"
```

---

## Task 8: Temporal Smoother Adjustment

Adjust the temporal smoother's default window to 3 (from 7) and threshold to 0.67 (2/3 agreement) as specified in the design.

**Files:**
- Modify: `src/perception/temporal_smoother.py`
- Create: `tests/test_temporal_smoother.py`

**Step 1: Write the test**

```python
# tests/test_temporal_smoother.py
import pytest
from src.perception.temporal_smoother import TemporalSmoother


def test_default_window_is_3():
    smoother = TemporalSmoother()
    assert smoother.window_size == 3


def test_two_of_three_agreement():
    smoother = TemporalSmoother()
    smoother.update("signal", "boiling")
    smoother.update("signal", "simmering")
    result = smoother.update("signal", "boiling")
    assert result.stable is True
    assert result.value == "boiling"


def test_no_agreement():
    smoother = TemporalSmoother()
    smoother.update("signal", "cold")
    smoother.update("signal", "simmering")
    result = smoother.update("signal", "boiling")
    assert result.stable is False
```

**Step 2: Run test, update defaults, verify passes**

In `src/perception/temporal_smoother.py`, change line 22:

```python
def __init__(self, window_size: int = 3, threshold: float = 0.67):
```

Run: `python -m pytest tests/test_temporal_smoother.py -v`
Expected: All 3 tests PASS

**Step 3: Commit**

```bash
git add src/perception/temporal_smoother.py tests/test_temporal_smoother.py
git commit -m "feat: adjust temporal smoother defaults to 3-frame window"
```

---

## Task 9: Demo Runner (End-to-End Wiring)

Wire all components together in `scripts/run_demo.py`. This is the main entry point for the hackathon demo.

**Files:**
- Modify: `scripts/run_demo.py`

**Step 1: Write the demo runner**

Replace `scripts/run_demo.py`:

```python
"""
End-to-end demo runner.

Wires together all components for the hackathon demo:
1. Load recipe from YAML
2. Load synthetic image sequence
3. Initialize VLM detector
4. Run inference-driven agent loop
5. Step transitions via rule engine
6. Output via console + TTS

Usage:
    python scripts/run_demo.py --recipe configs/recipes/pasta_and_sauce.yaml --images data/demo_sequences/pasta/
    python scripts/run_demo.py --recipe configs/recipes/pasta_and_sauce.yaml --images data/demo_sequences/pasta/ --mock
"""

import argparse
import time

from src.agent.recipe_loader import load_recipe
from src.agent.step_engine import StepEngine
from src.agent.tts import TTSEngine
from src.perception.detector import VLMDetector
from src.perception.image_sequencer import ImageSequencer
from src.perception.temporal_smoother import TemporalSmoother
from src.world_state.state import WorldState


def main():
    parser = argparse.ArgumentParser(description="Run the cooking monitor demo")
    parser.add_argument("--recipe", required=True, help="Recipe YAML path")
    parser.add_argument("--images", required=True, help="Image sequence directory")
    parser.add_argument("--model", default=None, help="Merged VLM model path")
    parser.add_argument("--mock", action="store_true", help="Use mock VLM (no GPU)")
    parser.add_argument("--no-audio", action="store_true", help="Disable TTS audio")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between frames in mock mode (seconds)")
    args = parser.parse_args()

    # Banner
    print("=" * 50)
    print("  AI Kitchen Cooking Monitor - Demo")
    print("=" * 50)
    print(f"  Recipe: {args.recipe}")
    print(f"  Images: {args.images}")
    print(f"  Model:  {args.model or 'MOCK MODE'}")
    print(f"  Audio:  {'OFF' if args.no_audio else 'ON'}")
    print("=" * 50)

    # 1. Load recipe
    recipe = load_recipe(args.recipe)
    print(f"\nLoaded recipe: {recipe.dish} ({len(recipe.steps)} steps, ~{recipe.estimated_time_minutes} min)")

    # 2. Initialize components
    world_state = WorldState(recipe=recipe)
    step_engine = StepEngine(recipe)
    sequencer = ImageSequencer(args.images)
    detector = VLMDetector(model_path=args.model, mock=args.mock)
    smoother = TemporalSmoother()
    tts = TTSEngine(audio=not args.no_audio)

    print(f"Loaded {len(sequencer)} images from {args.images}")
    print()

    # 3. Announce first step
    active = step_engine.get_active_step()
    if active:
        tts.speak(f"Let's make {recipe.dish}! Step {active.id}: {active.instruction}", priority="medium")
        if active.completion_type == "user_confirm":
            tts.speak("Say 'done' or press Enter when ready.", priority="low")

    # 4. Main loop (inference-driven)
    print("\n--- Starting cooking session ---\n")

    while not sequencer.done and not step_engine.all_done:
        active = step_engine.get_active_step()
        if not active:
            break

        # CAPTURE
        frame, filename = sequencer.next()
        if frame is None:
            break
        print(f"[Frame] {filename}")

        # INFER
        vlm_signal = active.vlm_signal if active.completion_type == "vlm" else None
        vlm_result = detector.detect(frame, step_signal=vlm_signal)
        print(f"  VLM: {vlm_result}")

        # SAFETY CHECK
        safety_actions = step_engine.check_safety(vlm_result)
        for action in safety_actions:
            tts.speak(f"WARNING: {action['reason']}", priority="critical")

        # SMOOTH (for VLM-based steps)
        if active.completion_type == "vlm":
            step_complete = vlm_result.get("step_complete", False)
            smoothed = smoother.update("step_complete", step_complete)
            print(f"  Smoothed: {smoothed.value} (stable={smoothed.stable}, ratio={smoothed.agreement_ratio:.2f})")

            if smoothed.stable and smoothed.value:
                actions = step_engine.check_vlm_signal(active.vlm_signal)
                for action in actions:
                    if action["type"] == "step_done":
                        tts.speak(f"Step {action['step_id']} complete!", priority="medium")
                    elif action["type"] == "step_activated":
                        _announce_step(tts, action)
                    elif action["type"] == "start_timer":
                        tts.speak(f"Timer started: {action['seconds']} seconds", priority="medium")
                        world_state.timer_engine.create_timer(action["timer_name"], action["seconds"])
                    elif action["type"] == "recipe_done":
                        tts.speak(f"{recipe.dish} is done! Enjoy your meal!", priority="high")
                smoother.reset("step_complete")

        # TIMER CHECK
        for timer in world_state.timer_engine.get_expired():
            actions = step_engine.check_timer_expired(timer.name)
            for action in actions:
                if action["type"] == "step_done":
                    tts.speak(f"Timer done! Step {action['step_id']} complete.", priority="medium")
                elif action["type"] == "step_activated":
                    _announce_step(tts, action)
                elif action["type"] == "recipe_done":
                    tts.speak(f"{recipe.dish} is done! Enjoy your meal!", priority="high")
            world_state.timer_engine.cancel_timer(timer.name)

        # USER CONFIRM (simulate with input in demo)
        if active.completion_type == "user_confirm" and active.status == "active":
            user_input = input(f"  [Step {active.id}] Press Enter when done (or 'q' to quit): ")
            if user_input.strip().lower() == "q":
                break
            actions = step_engine.user_confirm(active.id)
            for action in actions:
                if action["type"] == "step_done":
                    tts.speak(f"Step {action['step_id']} complete!", priority="medium")
                elif action["type"] == "step_activated":
                    _announce_step(tts, action)
                elif action["type"] == "start_timer":
                    tts.speak(f"Timer started: {action['seconds']} seconds", priority="medium")
                    world_state.timer_engine.create_timer(action["timer_name"], action["seconds"])
                elif action["type"] == "recipe_done":
                    tts.speak(f"{recipe.dish} is done! Enjoy your meal!", priority="high")

        # DELAY (for demo pacing in mock mode)
        if args.mock:
            time.sleep(args.delay)

    # 5. Done
    print("\n--- Cooking session ended ---")
    if step_engine.all_done:
        tts.speak(f"All steps complete. {recipe.dish} is ready!", priority="high")
    else:
        remaining = sum(1 for s in recipe.steps if s.status != "done")
        print(f"Session ended with {remaining} steps remaining.")


def _announce_step(tts: TTSEngine, action: dict):
    """Announce a newly activated step."""
    msg = f"Step {action['step_id']}: {action['instruction']}"
    tts.speak(msg, priority="medium")
    if action["completion_type"] == "user_confirm":
        tts.speak("Press Enter when done.", priority="low")
    elif action["completion_type"] == "timer":
        tts.speak("Timer will start automatically.", priority="low")


if __name__ == "__main__":
    main()
```

**Step 2: Smoke test with mock mode**

This requires at least one image in a demo directory. Create a placeholder:

```bash
mkdir -p data/demo_sequences/test
python -c "import cv2; import numpy as np; cv2.imwrite('data/demo_sequences/test/001_test.png', np.zeros((100,100,3), dtype=np.uint8))"
```

Run: `python scripts/run_demo.py --recipe configs/recipes/pasta_and_sauce.yaml --images data/demo_sequences/test/ --mock --no-audio`

Expected: Script starts, loads recipe, processes 1 image, prompts for user input on user_confirm step, exits.

**Step 3: Commit**

```bash
git add scripts/run_demo.py
git commit -m "feat: wire demo runner with all components"
```

---

## Task 10: Create Demo Recipe YAML (Mapo Tofu)

Create the mapo tofu recipe file that matches the design doc example.

**Files:**
- Create: `configs/recipes/mapo_tofu.yaml`

**Step 1: Write the recipe file**

```yaml
# configs/recipes/mapo_tofu.yaml
dish: mapo_tofu
servings: 2
estimated_time_minutes: 25

steps:
  - id: 1
    instruction: "Dice the tofu into 1-inch cubes"
    completion_type: vlm
    vlm_signal: "diced tofu on cutting board"
    depends_on: []

  - id: 2
    instruction: "Mince garlic, ginger, and scallions"
    completion_type: user_confirm
    depends_on: []

  - id: 3
    instruction: "Heat oil in wok until shimmering"
    completion_type: vlm
    vlm_signal: "oil shimmering in hot wok"
    depends_on: [1, 2]

  - id: 4
    instruction: "Add doubanjiang and cook for 30 seconds"
    completion_type: timer
    timer_seconds: 30
    depends_on: [3]
    parallel_group: A

  - id: 5
    instruction: "Add tofu and broth, simmer for 5 minutes"
    completion_type: timer
    timer_seconds: 300
    depends_on: [4]
    parallel_group: A

  - id: 6
    instruction: "Add cornstarch slurry and stir until thickened"
    completion_type: vlm
    vlm_signal: "thick glossy sauce coating tofu"
    depends_on: [5]
```

**Step 2: Validate it loads**

Run: `python -c "from src.agent.recipe_loader import load_recipe; r = load_recipe('configs/recipes/mapo_tofu.yaml'); print(f'{r.dish}: {len(r.steps)} steps')"`

Expected: `mapo_tofu: 6 steps`

**Step 3: Commit**

```bash
git add configs/recipes/mapo_tofu.yaml
git commit -m "feat: add mapo tofu demo recipe"
```

---

## Task 11: Generate Demo Image Sequence (Placeholder)

Create the directory structure and a script to generate placeholder images. Real images will be generated via Gemini separately.

**Files:**
- Create: `data/demo_sequences/README.md`
- Create: `scripts/generate_demo_placeholders.py`

**Step 1: Write placeholder generator**

```python
# scripts/generate_demo_placeholders.py
"""
Generate placeholder demo image sequences.

Creates numbered placeholder images for each recipe step.
Replace with real synthetic images generated via Gemini.

Usage:
    python scripts/generate_demo_placeholders.py --recipe configs/recipes/mapo_tofu.yaml --output data/demo_sequences/mapo_tofu/
"""

import argparse
import os

import cv2
import numpy as np

from src.agent.recipe_loader import load_recipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    recipe = load_recipe(args.recipe)
    os.makedirs(args.output, exist_ok=True)

    idx = 1
    for step in recipe.steps:
        # Create "before" image for each step
        img = np.zeros((512, 512, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)  # Dark gray background
        label = f"Step {step.id}: {step.instruction[:40]}"
        cv2.putText(img, label, (20, 256), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(img, f"[{step.completion_type}]", (20, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1)
        filename = f"{idx:03d}_step{step.id}_before.png"
        cv2.imwrite(os.path.join(args.output, filename), img)
        idx += 1

        # Create "after" / completion image
        img[:] = (60, 80, 60)  # Slightly green to indicate completion
        cv2.putText(img, f"Step {step.id} COMPLETE", (20, 256), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if step.vlm_signal:
            cv2.putText(img, f"Signal: {step.vlm_signal[:40]}", (20, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        filename = f"{idx:03d}_step{step.id}_after.png"
        cv2.imwrite(os.path.join(args.output, filename), img)
        idx += 1

    print(f"Generated {idx - 1} placeholder images in {args.output}")


if __name__ == "__main__":
    main()
```

**Step 2: Generate placeholders**

Run: `python scripts/generate_demo_placeholders.py --recipe configs/recipes/mapo_tofu.yaml --output data/demo_sequences/mapo_tofu/`

Expected: 12 placeholder PNG files created in `data/demo_sequences/mapo_tofu/`

**Step 3: Commit**

```bash
git add scripts/generate_demo_placeholders.py data/demo_sequences/
git commit -m "feat: add demo image placeholder generator"
```

---

## Task 12: Integration Test (End-to-End Mock Demo)

Create an integration test that runs the full pipeline in mock mode to verify all components work together.

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write the integration test**

```python
# tests/test_integration.py
"""End-to-end integration test using mock VLM."""

import os
import tempfile

import cv2
import numpy as np
import pytest

from src.agent.recipe_loader import load_recipe
from src.agent.schemas import Recipe, RecipeStep
from src.agent.step_engine import StepEngine
from src.agent.tts import TTSEngine
from src.perception.detector import VLMDetector
from src.perception.image_sequencer import ImageSequencer
from src.perception.temporal_smoother import TemporalSmoother
from src.world_state.state import WorldState


@pytest.fixture
def mock_image_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(6):
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(tmpdir, f"{i:03d}.png"), img)
        yield tmpdir


def make_simple_recipe():
    return Recipe(
        dish="test_pasta",
        estimated_time_minutes=5,
        steps=[
            RecipeStep(id=1, instruction="Boil water", completion_type="vlm", vlm_signal="boiling water"),
            RecipeStep(id=2, instruction="Add pasta", completion_type="timer", timer_seconds=1, depends_on=[1]),
            RecipeStep(id=3, instruction="Drain", completion_type="user_confirm", depends_on=[2]),
        ],
    )


def test_components_initialize():
    recipe = make_simple_recipe()
    world_state = WorldState(recipe=recipe)
    step_engine = StepEngine(recipe)
    detector = VLMDetector(mock=True)
    smoother = TemporalSmoother()
    tts = TTSEngine(audio=False)

    assert step_engine.get_active_step().id == 1
    assert not step_engine.all_done


def test_vlm_step_completes_with_signal(mock_image_dir):
    recipe = make_simple_recipe()
    step_engine = StepEngine(recipe)
    detector = VLMDetector(mock=True)

    # Simulate VLM detecting "boiling water"
    actions = step_engine.check_vlm_signal("boiling water")
    assert step_engine.get_step(1).status == "done"


def test_full_recipe_flow():
    recipe = make_simple_recipe()
    step_engine = StepEngine(recipe)
    tts = TTSEngine(audio=False)

    # Step 1: VLM detects boiling
    step_engine.check_vlm_signal("boiling water")
    assert step_engine.get_step(1).status == "done"

    # Step 2: Timer (simulate expiry)
    active = step_engine.get_active_step()
    assert active.id == 2
    timer_name = f"step_{active.id}_timer"
    step_engine.check_timer_expired(timer_name)
    assert step_engine.get_step(2).status == "done"

    # Step 3: User confirm
    active = step_engine.get_active_step()
    assert active.id == 3
    step_engine.user_confirm(3)
    assert step_engine.get_step(3).status == "done"

    assert step_engine.all_done
```

**Step 2: Run the integration test**

Run: `python -m pytest tests/test_integration.py -v`
Expected: All 3 tests PASS

**Step 3: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration tests for mock pipeline"
```

---

## Task Summary

| # | Task | New Files | Modified Files | Tests |
|---|---|---|---|---|
| 1 | Recipe Step Schema | `tests/test_schemas.py` | `src/agent/schemas.py` | 5 |
| 2 | Recipe Loader | `src/agent/recipe_loader.py`, `tests/test_recipe_loader.py` | `configs/recipes/pasta_and_sauce.yaml` | 3 |
| 3 | Image Sequencer | `src/perception/image_sequencer.py`, `tests/test_image_sequencer.py` | — | 5 |
| 4 | Step Transition Engine | `src/agent/step_engine.py`, `tests/test_step_engine.py` | — | 10 |
| 5 | VLM Detector | `tests/test_detector.py` | `src/perception/detector.py` | 2 |
| 6 | TTS Action Executor | `src/agent/tts.py`, `tests/test_tts.py` | `src/agent/tools.py` | 2 |
| 7 | World State Update | `tests/test_world_state.py` | `src/world_state/state.py` | 3 |
| 8 | Temporal Smoother Adjust | `tests/test_temporal_smoother.py` | `src/perception/temporal_smoother.py` | 3 |
| 9 | Demo Runner | — | `scripts/run_demo.py` | smoke test |
| 10 | Mapo Tofu Recipe | `configs/recipes/mapo_tofu.yaml` | — | validation |
| 11 | Demo Image Placeholders | `scripts/generate_demo_placeholders.py` | — | — |
| 12 | Integration Test | `tests/test_integration.py` | — | 3 |

**Total: 36+ tests across 12 tasks.**
