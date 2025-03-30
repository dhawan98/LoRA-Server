import requests
import json
import sys

BASE_URL = "http://localhost:8501"

# Only edit these if you're using "load" or "update_weights"

# to run use python change_lora.py load/clear/update_weights/status
LORA_REPOS = [
    "Negan454/divyanshu_lora"
]
LORA_WEIGHTS = [0.8]

def load_loras():
    response = requests.post(f"{BASE_URL}/load_loras", json={
        "repos": LORA_REPOS,
        "weights": LORA_WEIGHTS
    })
    print(json.dumps(response.json(), indent=2))

def update_weights():
    response = requests.post(f"{BASE_URL}/update_weights", json={
        "repos": LORA_REPOS,
        "weights": LORA_WEIGHTS
    })
    print(json.dumps(response.json(), indent=2))

def clear_loras():
    response = requests.post(f"{BASE_URL}/clear_loras")
    print(json.dumps(response.json(), indent=2))

def list_status():
    response = requests.get(f"{BASE_URL}/status")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python change_lora.py [load|update_weights|clear|status]")
        sys.exit(1)

    action = sys.argv[1]

    if action == "load":
        load_loras()
    elif action == "update_weights":
        update_weights()
    elif action == "clear":
        clear_loras()
    elif action == "status":
        list_status()
    else:
        print(f"❌ Unknown action: '{action}'")
        print("Usage: python change_lora.py [load|update_weights|clear|status]")
