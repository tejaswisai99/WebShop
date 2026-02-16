# ====== API LAYER: start/step service for WebShop text env ======
from flask import request, jsonify, Flask
from threading import Lock

from web_agent_site.envs.web_agent_text_env import SimServer, WebAgentTextEnv
from web_agent_site.utils import DEFAULT_FILE_PATH

app = Flask(__name__)

shared_server = SimServer(
    base_url='http://127.0.0.1:3000',
    file_path=DEFAULT_FILE_PATH,
    filter_goals=None,
    limit_goals=-1,
    num_products=None,
    human_goals=1,
    show_attrs=False,
)

# Manage multiple concurrent sessions
_sessions_lock = Lock()
_sessions = {}  # session_id -> WebAgentTextEnv

def _make_env(observation_mode='text', session_prefix='api_'):
    # Use the shared_server to keep catalogs/goals common and fast
    env = WebAgentTextEnv(
        observation_mode=observation_mode,
        server=shared_server,
        session_prefix=session_prefix,
        show_attrs=False,  # flip True to show attributes on item pages
        num_products=None,
        human_goals=0,
    )
    return env

def _available_actions(env: WebAgentTextEnv):
    meta = env.get_available_actions()
    return {
        "has_search_bar": bool(meta.get("has_search_bar")),
        "clickables": sorted(list(meta.get("clickables", [])))
    }

def _snapshot(env: WebAgentTextEnv):
    # Current observation + lightweight state snapshot for clients
    st = env.state
    return {
        "session_id": env.session,
        "instruction_text": st["instruction_text"],
        "url": st["url"],
        "observation_mode": env.observation_mode,
        "observation": env.observation,
        "available_actions": _available_actions(env)
    }

# ====== API LAYER additions/changes ======

@app.route("/api/start", methods=["POST"])
def api_start():
    """
    Start a new session (optionally with a specific goal).
    JSON body (all optional):
      {
        "goal_idx": 123,                      # int, index into SimServer.goals
        "observation_mode": "text" | "text_rich" | "html" | "url",
        "session": "my-session-id"
      }
    """
    payload = request.get_json(force=True, silent=True) or {}

    observation_mode = payload.get("observation_mode", "text")
    if observation_mode not in {"text", "text_rich", "html", "url"}:
        return jsonify({"error": f"Unsupported observation_mode: {observation_mode}"}), 400

    requested_session = payload.get("session")  # optional
    goal_idx = payload.get("goal_idx", None)
    if goal_idx is not None:
        try:
            goal_idx = int(goal_idx)
        except Exception:
            return jsonify({"error": "goal_idx must be an integer"}), 400
        if not (0 <= goal_idx < len(shared_server.goals)):
            return jsonify({"error": f"goal_idx out of range [0, {len(shared_server.goals)-1}]"}), 400

    env = _make_env(observation_mode=observation_mode, session_prefix="api_")
    # NEW: pass session_int=goal_idx for deterministic goal selection
    obs, _ = env.reset(session=requested_session,
                       instruction_text=shared_server.assigned_instruction_text,
                       session_int=goal_idx)

    with _sessions_lock:
        _sessions[env.session] = env

    # Clear goal override hack
    shared_server.assigned_instruction_text = None

    snap = _snapshot(env)
    # It’s useful to echo back which goal index got bound
    return jsonify({
        "message": "session_started",
        "goal_idx": goal_idx if goal_idx is not None else None,
        **snap
    }), 200

@app.route("/api/replay", methods=["POST"])
def api_replay():
    """
    Stateless, deterministic reconstruction of a page by replaying actions from a given goal.
    JSON body:
      {
        "goal_idx": 123,                        # REQUIRED (int; index into SimServer.goals)
        "actions": ["search[...]", "click[...]", ...],   # REQUIRED (list[str])
        "observation_mode": "text" | "text_rich" | "html" | "url"   # optional; default "text"
      }
    Returns (ephemeral; session not persisted):
      {
        "message": "replay_ok",
        "session_id": "<temp_session_id>",     # informational only (already cleaned up server-side)
        "goal_idx": 123,
        "steps_replayed": N,
        "stopped_early": true|false,           # true if 'done' occurred before consuming all actions
        "reward": <float>,                     # last reward observed during replay
        "done": true|false,
        ... snapshot fields (instruction_text, url, observation, available_actions) ...
      }
    """
    payload = request.get_json(force=True, silent=True) or {}
    if "goal_idx" not in payload or "actions" not in payload:
        return jsonify({"error": "goal_idx (int) and actions (list[str]) are required"}), 400

    # Validate goal_idx
    try:
        goal_idx = int(payload["goal_idx"])
    except Exception:
        return jsonify({"error": "goal_idx must be an integer"}), 400
    if not (0 <= goal_idx < len(shared_server.goals)):
        return jsonify({"error": f"goal_idx out of range [0, {len(shared_server.goals)-1}]"}), 400

    # Validate actions
    actions = payload["actions"]
    if not isinstance(actions, list) or any(not isinstance(a, str) for a in actions):
        return jsonify({"error": "actions must be a list of strings"}), 400

    # Observation mode
    observation_mode = payload.get("observation_mode", "text")
    if observation_mode not in {"text", "text_rich", "html", "url"}:
        return jsonify({"error": f"Unsupported observation_mode: {observation_mode}"}), 400

    # Create a temporary env (NOT registered in _sessions)
    env = _make_env(observation_mode=observation_mode, session_prefix="sim_")
    obs, _ = env.reset(session=None, session_int=goal_idx)

    # Replay the action history to reach the desired state
    last_reward, last_done = 0.0, False
    steps_replayed = 0
    for a in actions:
        _, last_reward, last_done, _ = env.step(a)
        steps_replayed += 1
        if last_done:
            break

    # Snapshot BEFORE cleanup
    snap = _snapshot(env)
    response = {
        "message": "replay_ok",
        "session_id": snap["session_id"],   # informational; will be cleaned
        "goal_idx": goal_idx,
        "steps_replayed": steps_replayed,
        "stopped_early": bool(last_done) and (steps_replayed < len(actions)),
        "reward": float(last_reward),
        "done": bool(last_done),
        **snap
    }

    # CRITICAL: clean up SimServer's per-session bookkeeping to avoid memory growth
    try:
        shared_server.user_sessions.pop(env.session, None)
    except Exception:
        pass  # defensive; ignore if already gone

    return jsonify(response), 200


@app.route("/api/step", methods=["POST"])
def api_step():
    """
    Take one environment step.
    JSON body:
      {
        "session_id": "...",
        "action": "search[keyword one two]" | "click[BUY NOW]" | ...
      }
    Returns: new observation/state, reward (if any), done flag, and optional verbose info on done.
    """
    payload = request.get_json(force=True, silent=True) or {}
    sid = payload.get("session_id")
    action = payload.get("action")

    if not sid or not isinstance(sid, str):
        return jsonify({"error": "Missing or invalid 'session_id'"}), 400
    if not action or not isinstance(action, str):
        return jsonify({"error": "Missing or invalid 'action'"}), 400

    with _sessions_lock:
        env = _sessions.get(sid)

    if env is None:
        return jsonify({"error": f"Unknown session_id: {sid}. Start with /api/start."}), 404

    # Step the environment
    state, reward, done, _ = env.step(action)

    # Build response
    snap = _snapshot(env)
    resp = {
        "message": "step_ok",
        "session_id": sid,
        "action_taken": action,
        "state": state,                 # full [SEP]-joined state if num_prev_* enabled
        "reward": reward,
        "done": bool(done),
        "observation": snap["observation"],
        "url": snap["url"],
        "available_actions": snap["available_actions"],
        "instruction_text": snap["instruction_text"]
    }

    # If episode terminated (clicked END / purchase), attach verbose match info if available
    if done:
        verbose = shared_server.user_sessions.get(sid, {}).get("verbose_info")
        if verbose is not None:
            resp["match_explanation"] = verbose

    return jsonify(resp), 200


@app.route("/api/reset",methods=["POST"])
def api_reset():
    """
    Optional: reset an existing session (keeps the same session_id),
    optionally with a new goal.
    JSON body:
      {
        "session_id": "...",
        "goal": "New instruction text"   (optional)
      }
    """
    payload = request.get_json(force=True, silent=True) or {}
    sid = payload.get("session_id")
    new_goal = payload.get("goal")

    if not sid or not isinstance(sid, str):
        return jsonify({"error": "Missing or invalid 'session_id'"}), 400

    with _sessions_lock:
        env = _sessions.get(sid)

    if env is None:
        return jsonify({"error": f"Unknown session_id: {sid}"}), 404

    # If a new goal is provided, ensure server uses it for this reset
    if isinstance(new_goal, str) and new_goal.strip():
        shared_server.assigned_instruction_text = new_goal
        obs, _ = env.reset(session=sid, instruction_text=new_goal)
        shared_server.assigned_instruction_text = None
    else:
        obs, _ = env.reset(session=sid)

    snap = _snapshot(env)
    return jsonify({
        "message": "session_reset",
        **snap
    }), 200


@app.route("/api/end", methods=["POST"])
def api_end():
    """
    Optional: end and delete a session.
    JSON body:
      { "session_id": "..." }
    """
    payload = request.get_json(force=True, silent=True) or {}
    sid = payload.get("session_id")
    if not sid or not isinstance(sid, str):
        return jsonify({"error": "Missing or invalid 'session_id'"}), 400

    with _sessions_lock:
        existed = _sessions.pop(sid, None) is not None

    return jsonify({
        "message": "session_deleted" if existed else "session_not_found",
        "session_id": sid
    }), 200


# --- Entry point (optional) ---
if __name__ == "__main__":
    # IMPORTANT: this runs the API server for /api/* endpoints (not the simulated shop UI)
    # Adjust host/port as needed.
    app.run(host="0.0.0.0", port=5000, debug=False)
