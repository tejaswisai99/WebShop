# test_option_a_driver.py
import requests, json

BASE = "http://127.0.0.1:5000/api"
s = requests.Session()
s.headers.update({"Content-Type": "application/json"})

def start_goal(goal_idx, obs_mode="text"):
    r = s.post(f"{BASE}/start", json={"goal_idx": goal_idx, "observation_mode": obs_mode})
    r.raise_for_status()
    print(r.json())
    return r.json()

def replay(goal_idx, actions, obs_mode="text"):
    r = s.post(f"{BASE}/replay", json={"goal_idx": goal_idx, "actions": actions, "observation_mode": obs_mode})
    r.raise_for_status()
    print(r.json())
    return r.json()

def main():
    NUM = 50
    for g in range(NUM):
        # (1) deterministically bind to goal g (for reference only)
        meta = start_goal(g)
        instr = meta["instruction_text"]
        # (2) build or fetch the "current action" prefix you want to simulate to
        # For demo, we use an empty history (i.e., current page is the search page)
        prefix_actions = []  # replace with your real prefix for that goal
        # (3) stateless replay to get the page you want
        snap = replay(g, prefix_actions)
        # (4) you now have the "current webpage" for that goal/prefix
        obs = snap["observation"]; url = snap["url"]
        # (5) no reset/cleanup needed (ephemeral)
        if (g+1) % 100 == 0:
            print(f"Done {g+1}/{NUM} goals")

if __name__ == "__main__":
    main()
