"""
Capability test for Gemma 3 on kitchen monitoring tasks.

Tests the base model's ability to:
1. Procedure compliance (state -> action tool calls)
2. Safety detection (smoke, unattended cooking)
3. Multi-task coordination (wok cooking sequences)
4. No-action restraint (don't over-alert)
5. Cuisine-specific timing (Chinese, Thai)
6. Timer expiry response

Usage:
    python infra/fine_tuning/eval/capability_test.py
    python infra/fine_tuning/eval/capability_test.py --model unsloth/gemma-3-12b-it
"""

import argparse
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


SYSTEM_PROMPT = (
    'You are a kitchen monitoring agent. Given the current kitchen state as JSON, '
    'decide what actions to take. Output ONLY a valid JSON object with an "actions" array. '
    'Available tools:\n'
    '- set_timer(name, seconds)\n'
    '- speak(text, priority)  [priority: low/medium/high/critical]\n'
    '- mark_step_done(step_id)\n'
    '- show_card(title, bullets)\n'
    'If no action is needed, return {"actions": []}.\n'
    'Output ONLY the JSON, no explanation.'
)

TEST_CASES = [
    {
        "name": "Procedure: Water boiling, pasta step pending",
        "state": {
            "stove": {"water_state": "boiling", "pot_present": True, "smoke_suspected": False, "steam_level": "high"},
            "timers": [],
            "steps": [
                {"id": 1, "name": "boil water", "status": "done"},
                {"id": 2, "name": "add pasta to boiling water", "status": "pending"},
                {"id": 3, "name": "make tomato sauce", "status": "pending"},
            ],
        },
        "expected_behavior": "Should speak about adding pasta, set 8-min timer, mark step 2 done",
    },
    {
        "name": "Safety: Smoke detected, no person present",
        "state": {
            "stove": {"water_state": "simmering", "pot_present": True, "smoke_suspected": True, "steam_level": "high"},
            "counter": {"hands_active": False, "person_present": False},
            "timers": [{"name": "pasta", "remaining_s": 180}],
            "steps": [
                {"id": 2, "name": "add pasta", "status": "done"},
                {"id": 3, "name": "make sauce", "status": "in_progress"},
            ],
        },
        "expected_behavior": "Should alert critically about smoke AND unattended cooking",
    },
    {
        "name": "Multi-task: Pad Thai wok sequence",
        "state": {
            "stove": {"wok_state": "smoking_hot", "oil_just_added": True, "temperature": "very_high"},
            "counter": {"soaked_rice_noodles": True, "pad_thai_sauce_mixed": True, "beaten_eggs": True},
            "timers": [],
            "current_recipe": "pad_thai",
            "steps": [
                {"id": 1, "name": "soak rice noodles 20min", "status": "done"},
                {"id": 2, "name": "mix pad thai sauce", "status": "done"},
                {"id": 3, "name": "sear shrimp in wok 60s", "status": "pending"},
                {"id": 4, "name": "add noodles and sauce", "status": "pending"},
                {"id": 5, "name": "push aside and scramble egg", "status": "pending"},
                {"id": 6, "name": "toss in bean sprouts and peanuts", "status": "pending"},
            ],
        },
        "expected_behavior": "Should instruct to add shrimp now (wok is ready), set short timer",
    },
    {
        "name": "No-action: Normal cooking in progress",
        "state": {
            "stove": {"water_state": "simmering", "pot_present": True, "smoke_suspected": False, "steam_level": "low"},
            "counter": {"hands_active": True, "person_present": True, "knife_in_use": True},
            "timers": [{"name": "pasta", "remaining_s": 300}],
            "steps": [
                {"id": 2, "name": "add pasta", "status": "done"},
                {"id": 3, "name": "chop onion for sauce", "status": "in_progress"},
                {"id": 4, "name": "saute onion and garlic", "status": "pending"},
            ],
        },
        "expected_behavior": "Should return empty actions or minimal status update. NO unnecessary alerts.",
    },
    {
        "name": "Chinese: Mapo tofu - doubanjiang in hot oil",
        "state": {
            "stove": {"wok_state": "oil_hot", "sichuan_peppercorns_bloomed": True, "temperature": "high"},
            "counter": {"doubanjiang_measured": True, "tofu_blanched_and_cubed": True},
            "timers": [],
            "current_recipe": "mapo_tofu",
            "steps": [
                {"id": 1, "name": "blanch tofu cubes 2min", "status": "done"},
                {"id": 2, "name": "bloom sichuan peppercorns in oil", "status": "done"},
                {"id": 3, "name": "fry doubanjiang paste 30s (burns fast!)", "status": "pending"},
                {"id": 4, "name": "add ground pork and stir", "status": "pending"},
                {"id": 5, "name": "add tofu and stock, simmer 5min", "status": "pending"},
            ],
        },
        "expected_behavior": "Should urgently instruct to add doubanjiang NOW and stir constantly (burns in seconds)",
    },
    {
        "name": "Timer expired: Pasta timer hit zero",
        "state": {
            "stove": {"water_state": "boiling", "pot_present": True, "smoke_suspected": False},
            "counter": {"colander_in_sink": True, "sauce_ready": True},
            "timers": [{"name": "pasta", "remaining_s": 0}],
            "steps": [
                {"id": 2, "name": "cook pasta 8min", "status": "in_progress"},
                {"id": 3, "name": "drain pasta immediately", "status": "pending"},
                {"id": 4, "name": "toss with sauce", "status": "pending"},
            ],
        },
        "expected_behavior": "Should urgently say to drain pasta NOW, mark step done",
    },
]


def run_test(model, tokenizer, test_case, test_num):
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"TEST {test_num}: {test_case['name']}")
    print(sep)
    print(f"Expected: {test_case['expected_behavior']}")

    messages = [
        {
            "role": "user",
            "content": f"{SYSTEM_PROMPT}\n\nCurrent kitchen state:\n{json.dumps(test_case['state'], indent=2)}",
        }
    ]

    input_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    start = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.1,
            do_sample=True,
        )
    gen_time = time.time() - start

    response = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    print(f"\nResponse ({gen_time:.2f}s):")
    print(response[:800])

    # Validate JSON
    valid_json = False
    num_actions = 0
    try:
        start_idx = response.find("{")
        end_idx = response.rfind("}") + 1
        if start_idx >= 0 and end_idx > 0:
            parsed = json.loads(response[start_idx:end_idx])
            if "actions" in parsed and isinstance(parsed["actions"], list):
                valid_json = True
                num_actions = len(parsed["actions"])
    except (json.JSONDecodeError, ValueError):
        pass

    status = "PASS" if valid_json else "FAIL"
    print(f"\n>> JSON Valid: {valid_json} | Actions: {num_actions} | {status}")
    return {"name": test_case["name"], "valid_json": valid_json, "num_actions": num_actions, "time": gen_time}


def main():
    parser = argparse.ArgumentParser(description="Test Gemma capability on kitchen tasks")
    parser.add_argument("--model", default="unsloth/gemma-3-4b-it", help="Model to test")
    args = parser.parse_args()

    sep = "=" * 60
    print(sep)
    print("GEMMA KITCHEN MONITORING CAPABILITY TEST")
    print(sep)

    # Load model
    print(f"\nLoading {args.model}...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    print(f"Loaded in {time.time() - start:.1f}s")
    print(f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    # Run all tests
    results = []
    for i, test_case in enumerate(TEST_CASES, 1):
        result = run_test(model, tokenizer, test_case, i)
        results.append(result)

    # Summary
    print(f"\n{sep}")
    print("RESULTS SUMMARY")
    print(sep)
    total = len(results)
    passed = sum(1 for r in results if r["valid_json"])
    avg_time = sum(r["time"] for r in results) / total

    for r in results:
        icon = "OK" if r["valid_json"] else "FAIL"
        print(f"  [{icon}] {r['name']} ({r['num_actions']} actions, {r['time']:.2f}s)")

    print(f"\nJSON Schema Compliance: {passed}/{total} ({passed/total*100:.0f}%)")
    print(f"Average Generation Time: {avg_time:.2f}s")

    print(f"\n{sep}")
    print("INSTANCE SPECS")
    print(sep)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    vram_peak = torch.cuda.max_memory_allocated() / 1e9
    print(f"VRAM Total: {vram_total:.1f} GB")
    print(f"VRAM Peak: {vram_peak:.1f} GB")
    print(f"VRAM Headroom: {vram_total - vram_peak:.1f} GB")
    print(f"Model: {args.model} (bfloat16)")

    # Can we fit 12B?
    print(f"\n{sep}")
    print("MODEL SIZE RECOMMENDATIONS")
    print(sep)
    print(f"  Gemma 3 4B  (bf16): ~8 GB  -> FITS ({vram_total:.0f} GB available)")
    print(f"  Gemma 3 12B (bf16): ~24 GB -> FITS ({vram_total:.0f} GB available)")
    print(f"  Gemma 3 27B (bf16): ~54 GB -> FITS ({vram_total:.0f} GB available)")
    print(f"  Gemma 3 12B + LoRA training: ~30 GB -> FITS with room for batch")
    print(f"  Gemma 3 27B + LoRA training: ~70 GB -> FITS (tight)")


if __name__ == "__main__":
    main()
