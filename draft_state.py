import json
import os

STATE_FILE = "draft_state.json"

def load_state():
    if not os.path.exists(STATE_FILE):
        return {"my_picks": [], "drafted": []}
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"my_picks": [], "drafted": []}

def save_state(my_picks, drafted):
    state = {"my_picks": my_picks, "drafted": drafted}
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)
