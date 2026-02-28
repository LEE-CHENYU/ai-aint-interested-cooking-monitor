"""
Per-task configuration registry for the VLM fine-tuning pipeline.

Each task defines:
- Matrix structure (rows × cols) with ground truth labels
- System prompt for VLM inference
- Training hyperparameters
- Binary classification key and thresholds
"""

# Default execution order — task1/task6 bake-off decides which goes first/last
TASK_ORDER = ["task1_boilover", "task2_smoke", "task3_person", "task6_boiling"]

# Tasks that compete in the bake-off (similar liquid-in-pot tasks)
BAKEOFF_TASKS = ("task1_boilover", "task6_boiling")

TASK_CONFIGS = {
    # =========================================================================
    # TASK 1 — POT BOILING OVER (overflow severity)
    # =========================================================================
    "task1_boilover": {
        "description": "Pot overflow / boilover severity detection",
        "rows": ["stainless", "saucepan", "stockpot", "dutch", "blackpot", "wok"],
        "cols": [
            "1_safe_low", "2_safe_simmer", "3_rising",
            "4_AT_RIM", "5_SPILLING", "6_OVERFLOW",
        ],
        "safe_cols": ["1_safe_low", "2_safe_simmer", "3_rising"],
        "unsafe_cols": ["4_AT_RIM", "5_SPILLING", "6_OVERFLOW"],
        "binary_key": "safe",
        "row_field": "vessel",
        "state_field": "state",
        "row_labels": {
            "stainless": "large stainless steel pot",
            "saucepan": "medium saucepan",
            "stockpot": "large stock pot",
            "dutch": "cast iron dutch oven",
            "blackpot": "black enamel pot",
            "wok": "carbon steel wok",
        },
        "col_labels": {
            "1_safe_low": {
                "state": "calm water at low level",
                "safe": True,
                "reason": "Water level well below rim, no overflow risk",
            },
            "2_safe_simmer": {
                "state": "gentle simmer below rim",
                "safe": True,
                "reason": "Light simmering with water safely below rim",
            },
            "3_rising": {
                "state": "foam building, approaching rim",
                "safe": True,
                "reason": "Foam rising but still contained within pot",
            },
            "4_AT_RIM": {
                "state": "liquid at the rim, about to overflow",
                "safe": False,
                "reason": "Liquid right at rim edge, imminent overflow",
            },
            "5_SPILLING": {
                "state": "liquid spilling over rim",
                "safe": False,
                "reason": "Liquid actively spilling over the pot rim",
            },
            "6_OVERFLOW": {
                "state": "massive overflow on all sides",
                "safe": False,
                "reason": "Liquid overflowing massively, stovetop flooded",
            },
        },
        "system_prompt": (
            "Analyze this top-down photo of a pot on a stove. "
            "Determine if the pot is at risk of boiling over.\n\n"
            "Respond ONLY with JSON:\n"
            '{"vessel": "<type>", "state": "<description>", '
            '"safe": true/false, "confidence": 0.0-1.0, '
            '"reason": "<explanation>"}\n\n'
            "safe=true: Liquid level safely below rim, normal cooking\n"
            "safe=false: Liquid at rim, spilling, or overflowing"
        ),
        "epochs": 3,
        "lr": 2e-4,
        "lora_rank": 32,
        "finetune_vision": False,
        "oversample_unsafe": 3,
    },

    # =========================================================================
    # TASK 2 — SMOKE / FIRE DETECTION
    # =========================================================================
    "task2_smoke": {
        "description": "Smoke and fire detection",
        "rows": ["frypan", "wok", "skillet", "pot", "deepoil", "oventop"],
        "cols": [
            "1_clear", "2_steam", "3_light_haze",
            "4_SMOKE", "5_HEAVY_SMOKE", "6_FIRE",
        ],
        "safe_cols": ["1_clear", "2_steam", "3_light_haze"],
        "unsafe_cols": ["4_SMOKE", "5_HEAVY_SMOKE", "6_FIRE"],
        "binary_key": "safe",
        "row_field": "scenario",
        "state_field": "air_quality",
        "row_labels": {
            "frypan": "stainless steel frying pan",
            "wok": "carbon steel wok",
            "skillet": "cast iron skillet",
            "pot": "stainless steel pot",
            "deepoil": "deep frying pot with oil",
            "oventop": "oven with baking tray",
        },
        "col_labels": {
            "1_clear": {
                "state": "completely clear air, no smoke",
                "safe": True,
                "reason": "Air is clear with no smoke or haze visible",
            },
            "2_steam": {
                "state": "normal cooking steam",
                "safe": True,
                "reason": "Clean white steam from cooking, normal and safe",
            },
            "3_light_haze": {
                "state": "slight haze, early warning",
                "safe": True,
                "reason": "Very faint haze but no actual smoke yet",
            },
            "4_SMOKE": {
                "state": "visible smoke from overheating",
                "safe": False,
                "reason": "Grey-white smoke rising, oil or food overheated",
            },
            "5_HEAVY_SMOKE": {
                "state": "heavy dense smoke, dangerous",
                "safe": False,
                "reason": "Thick smoke billowing, visibility reduced, dangerous",
            },
            "6_FIRE": {
                "state": "fire with visible flames",
                "safe": False,
                "reason": "Active fire with flames and smoke, extremely dangerous",
            },
        },
        "system_prompt": (
            "Analyze this top-down photo of cooking on a stove. "
            "Determine if there is dangerous smoke or fire.\n\n"
            "Respond ONLY with JSON:\n"
            '{"scenario": "<cooking type>", "air_quality": "<description>", '
            '"safe": true/false, "confidence": 0.0-1.0, '
            '"reason": "<explanation>"}\n\n'
            "safe=true: Clear air, normal steam, or very light haze\n"
            "safe=false: Visible smoke, heavy smoke, or fire detected"
        ),
        "epochs": 3,
        "lr": 2e-4,
        "lora_rank": 32,
        "finetune_vision": True,
        "oversample_unsafe": 3,
    },

    # =========================================================================
    # TASK 3 — PERSON PRESENT / ABSENT
    # =========================================================================
    "task3_person": {
        "description": "Person presence detection",
        "rows": ["stirring", "chopping", "wokfry", "plating", "pouring", "searing"],
        "cols": [
            "1_active", "2_both_hands", "3_one_hand",
            "4_ARM_EDGE", "5_JUST_LEFT", "6_ABSENT",
        ],
        "safe_cols": ["1_active", "2_both_hands", "3_one_hand"],
        "unsafe_cols": ["4_ARM_EDGE", "5_JUST_LEFT", "6_ABSENT"],
        "binary_key": "present",
        "row_field": "activity",
        "state_field": "presence_level",
        "row_labels": {
            "stirring": "stirring pot with spoon",
            "chopping": "chopping on cutting board",
            "wokfry": "wok stir-frying",
            "plating": "plating food on counter",
            "pouring": "pouring liquid into pot",
            "searing": "pan searing with tongs",
        },
        "col_labels": {
            "1_active": {
                "state": "actively cooking, fully present",
                "present": True,
                "reason": "Both hands visible, person actively engaged in cooking",
            },
            "2_both_hands": {
                "state": "both hands visible, present but pausing",
                "present": True,
                "reason": "Two hands clearly visible in frame, person present",
            },
            "3_one_hand": {
                "state": "one hand visible, partially present",
                "present": True,
                "reason": "At least one hand visible, person still in the area",
            },
            "4_ARM_EDGE": {
                "state": "barely visible at frame edge",
                "present": False,
                "reason": "Only edge of arm barely visible, person mostly gone",
            },
            "5_JUST_LEFT": {
                "state": "person just stepped away",
                "present": False,
                "reason": "No hands visible, utensils left as if just released",
            },
            "6_ABSENT": {
                "state": "completely absent, nobody present",
                "present": False,
                "reason": "No person visible anywhere, kitchen unattended",
            },
        },
        "system_prompt": (
            "Analyze this top-down photo of a kitchen scene. "
            "Determine if a person is present and actively cooking.\n\n"
            "Respond ONLY with JSON:\n"
            '{"activity": "<cooking activity>", "presence_level": "<description>", '
            '"present": true/false, "confidence": 0.0-1.0, '
            '"reason": "<explanation>"}\n\n'
            "present=true: Person actively present with hands/arms clearly visible\n"
            "present=false: Person mostly gone, just left, or completely absent"
        ),
        "epochs": 3,
        "lr": 2e-4,
        "lora_rank": 32,
        "finetune_vision": True,
        "oversample_unsafe": 3,
    },

    # =========================================================================
    # TASK 6 — WATER BOILING STATE
    # =========================================================================
    "task6_boiling": {
        "description": "Water boiling state detection",
        "rows": ["stainless", "saucepan", "stockpot", "glass", "dutch", "wok"],
        "cols": [
            "1_cold", "2_warm", "3_pre_simmer",
            "4_simmer", "5_BOIL", "6_RAPID_BOIL",
        ],
        "safe_cols": ["1_cold", "2_warm", "3_pre_simmer"],
        "unsafe_cols": ["4_simmer", "5_BOIL", "6_RAPID_BOIL"],
        "binary_key": "boiling",
        "row_field": "vessel",
        "state_field": "water_state",
        "row_labels": {
            "stainless": "large stainless steel pot",
            "saucepan": "medium saucepan",
            "stockpot": "large stock pot",
            "glass": "clear glass pot",
            "dutch": "cast iron dutch oven",
            "wok": "carbon steel wok",
        },
        "col_labels": {
            "1_cold": {
                "state": "cold still water, no heat",
                "boiling": False,
                "reason": "Perfectly still water, no bubbles or steam",
            },
            "2_warm": {
                "state": "slightly warm, micro bubbles",
                "boiling": False,
                "reason": "Tiny bubbles on sides, water surface still calm",
            },
            "3_pre_simmer": {
                "state": "getting hotter, small bubbles rising",
                "boiling": False,
                "reason": "Small bubbles rising but not yet simmering",
            },
            "4_simmer": {
                "state": "gentle simmer with steady bubbles",
                "boiling": True,
                "reason": "Steady bubbles breaking at surface, light steam rising",
            },
            "5_BOIL": {
                "state": "full rolling boil",
                "boiling": True,
                "reason": "Large bubbles continuously breaking surface, significant steam",
            },
            "6_RAPID_BOIL": {
                "state": "violent rapid boil",
                "boiling": True,
                "reason": "Massive bubbles, splashing water, dense steam cloud",
            },
        },
        "system_prompt": (
            "Analyze this top-down photo of liquid in a pot on a stove. "
            "Determine if the water is boiling.\n\n"
            "Respond ONLY with JSON:\n"
            '{"vessel": "<type>", "water_state": "<description>", '
            '"boiling": true/false, "confidence": 0.0-1.0, '
            '"reason": "<explanation>"}\n\n'
            "boiling=true: Water is simmering, at full boil, or rapid boil\n"
            "boiling=false: Water is cold, warm, or pre-simmer only"
        ),
        "epochs": 3,
        "lr": 2e-4,
        "lora_rank": 32,
        "finetune_vision": False,
        "oversample_unsafe": 3,
    },
}
