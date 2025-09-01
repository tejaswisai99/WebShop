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
    human_goals=0,
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

@app.route("/api/start", methods=["POST"])
def api_start():
    """
    Start a new session with a caller-provided goal (instruction_text).
    JSON body:
      {
        "goal": "Find me ... under $20 ...",
        "observation_mode": "text" | "text_rich" | "html" | "url",  (optional; default "text")
        "session": "my-session-id"                                   (optional)
      }
    Returns: session snapshot with observation and available actions.
    """
    payload = request.get_json(force=True, silent=True) or {}
    goal = payload.get("goal")
    if not goal or not isinstance(goal, str):
        return jsonify({"error": "Missing or invalid 'goal' (instruction_text)"}), 400

    observation_mode = payload.get("observation_mode", "text")
    if observation_mode not in {"text", "text_rich", "html", "url"}:
        return jsonify({"error": f"Unsupported observation_mode: {observation_mode}"}), 400

    requested_session = payload.get("session")  # optional client-provided session id

    # Ensure the next env reset uses this exact instruction_text
    # (SimServer.receive respects assigned_instruction_text if present)
    shared_server.assigned_instruction_text = goal

    env = _make_env(observation_mode=observation_mode, session_prefix="api_")
    # Use client session if supplied; otherwise WebAgentTextEnv will generate one
    obs, _ = env.reset(session=requested_session, instruction_text=goal)

    # Persist this env handle by its session_id
    with _sessions_lock:
        _sessions[env.session] = env

    # Clear the hack after use so future sessions don’t inherit by accident
    shared_server.assigned_instruction_text = None

    snap = _snapshot(env)
    return jsonify({
        "message": "session_started",
        **snap
    }), 200


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
