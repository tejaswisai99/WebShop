from flask import Flask, request, jsonify
from threading import Lock
from web_agent_site.envs import WebAgentTextEnv
from web_agent_site.utils import DEBUG_PROD_SIZE

app = Flask(__name__)

SESSIONS = {}

def get_or_create_env(session_id: str, observation_mode='text', render=False, num_products=DEBUG_PROD_SIZE):
    if session_id not in SESSIONS:
        env = WebAgentTextEnv(observation_mode=observation_mode, render=render, num_products=num_products)
        SESSIONS[session_id] = {"env": env, "lock": Lock()}
    return SESSIONS[session_id]["env"], SESSIONS[session_id]["lock"]

@app.route("/reset", methods=["POST"])
def reset():
    data = request.get_json(force=True)
    sid = data["session_id"]
    num_products = data.get("num_products", DEBUG_PROD_SIZE)

    env, lock = get_or_create_env(sid, observation_mode='text', render=False, num_products=num_products)
    with lock:
        obs = env.reset()
        # immediately expose available actions so the client doesn’t need a 2nd call
        available = env.get_available_actions()
    return jsonify({"session_id": sid, "observation": obs, "available_actions": available, "done": False, "reward": 0.0})

@app.route("/step",methods=["POST"])
def step():
    data = request.get_json(force=True)
    sid = data["session_id"]
    action = data["action"]  # e.g., 'search[black running shoes under $70]'

    env, lock = get_or_create_env(sid)
    with lock:
        obs, reward, done, info = env.step(action)
        available = env.get_available_actions() if not done else {}
    # keep the env alive even if done; the client may /reset later
    return jsonify({
        "session_id": sid,
        "observation": obs,
        "available_actions": available,
        "reward": float(reward),
        "done": bool(done),
        "info": info,
    })

@app.route("/observation", methods=["GET"])
def observation():
    sid = request.args["session_id"]
    env, lock = get_or_create_env(sid)
    with lock:
        obs = env.observation
        available = env.get_available_actions()
    return jsonify({"session_id": sid, "observation": obs, "available_actions": available})

@app.route("/close", methods=["POST"])
def close():
    data = request.get_json(force=True)
    sid = data["session_id"]
    entry = SESSIONS.pop(sid, None)
    if entry:
        with entry["lock"]:
            entry["env"].close()
    return jsonify({"closed": bool(entry)})

if __name__ == "__main__":
    # You can change the port if you like
    get_or_create_env('1',observation_mode='text', render=False, num_products=DEBUG_PROD_SIZE)
    app.run(host="127.0.0.1", port=8001, threaded=True)