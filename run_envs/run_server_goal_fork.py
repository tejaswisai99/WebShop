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

# --- add near the other globals ---
_session_histories = {}      # session_id -> list[str]
_session_goal_idx = {}       # session_id -> int (index into shared_server.goals)

# --- add helper to compute the index of the current session's goal ---
def _goal_idx_for_session(session_id: str) -> int:
    sess = shared_server.user_sessions.get(session_id)
    if not sess:
        return None
    goal = sess.get("goal", {})
    # Prefer explicit ids if present
    for key in ("goal_id", "id", "target_asin", "asin"):
        if key in goal:
            val = goal[key]
            for i, g in enumerate(shared_server.goals):
                if g.get(key) == val:
                    return i
    # Fallback: identity/equality scan
    try:
        return shared_server.goals.index(goal)
    except ValueError:
        # Last resort: match on a couple common fields
        for i, g in enumerate(shared_server.goals):
            if g.get("instruction_text") == goal.get("instruction_text"):
                return i
    return None


def _make_env(observation_mode='text', session_prefix='api_'):
    env = WebAgentTextEnv(
        observation_mode=observation_mode,
        server=shared_server,
        session_prefix=session_prefix,
        show_attrs=False,
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
    st = env.state
    return {
        "session_id": env.session,
        "instruction_text": st["instruction_text"],
        "url": st["url"],
        "observation_mode": env.observation_mode,
        "observation": env.observation,
        "available_actions": _available_actions(env)
    }

# NEW: helper to append to history
def _record_history(session_id: str, action: str):
    _session_histories.setdefault(session_id, []).append(action)

@app.route("/api/start", methods=["POST"])
def api_start():
    payload = request.get_json(force=True, silent=True) or {}

    observation_mode = payload.get("observation_mode", "text")
    if observation_mode not in {"text", "text_rich", "html", "url"}:
        return jsonify({"error": f"Unsupported observation_mode: {observation_mode}"}), 400

    requested_session = payload.get("session")

    env = _make_env(observation_mode=observation_mode, session_prefix="api_")
    # If you want to force a specific instruction, set shared_server.assigned_instruction_text first.
    obs, _ = env.reset(session=requested_session, instruction_text=shared_server.assigned_instruction_text)

    with _sessions_lock:
        _sessions[env.session] = env
    # ensure per-session history starts empty
    _session_histories[env.session] = []

    # NEW: record goal idx and include it in responses
    idx = _goal_idx_for_session(env.session)
    _session_goal_idx[env.session] = idx

    # clear assigned goal after use
    shared_server.assigned_instruction_text = None

    snap = _snapshot(env)
    return jsonify({"message": "session_started", "goal_idx":idx, **snap}), 200


@app.route("/api/step", methods=["POST"])
def api_step():
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
    # NEW: keep history so we can fork by replay
    _record_history(sid, action)

    snap = _snapshot(env)
    resp = {
        "message": "step_ok",
        "session_id": sid,
        "action_taken": action,
        "state": state,
        "reward": reward,
        "done": bool(done),
        "observation": snap["observation"],
        "url": snap["url"],
        "available_actions": snap["available_actions"],
        "instruction_text": snap["instruction_text"]
    }
    if done:
        verbose = shared_server.user_sessions.get(sid, {}).get("verbose_info")
        if verbose is not None:
            resp["match_explanation"] = verbose
    return jsonify(resp), 200


@app.route("/api/reset", methods=["POST"])
def api_reset():
    payload = request.get_json(force=True, silent=True) or {}
    sid = payload.get("session_id")
    new_goal = payload.get("goal")

    if not sid or not isinstance(sid, str):
        return jsonify({"error": "Missing or invalid 'session_id'"}), 400

    with _sessions_lock:
        env = _sessions.get(sid)
    if env is None:
        return jsonify({"error": f"Unknown session_id: {sid}"}), 404

    if isinstance(new_goal, str) and new_goal.strip():
        shared_server.assigned_instruction_text = new_goal
        obs, _ = env.reset(session=sid, instruction_text=new_goal)
        shared_server.assigned_instruction_text = None
    else:
        obs, _ = env.reset(session=sid)

    # NEW: clear history on reset
    _session_histories[sid] = []

    snap = _snapshot(env)
    return jsonify({"message": "session_reset", **snap}), 200


# NEW: Fork endpoint — recreate same goal and replay action history
@app.route("/api/fork", methods=["POST"])
def api_fork():
    """
    Create a NEW session at the exact same state as an existing one by:
      1) reading the base session's instruction_text and observation_mode,
      2) resetting a fresh env with that instruction_text,
      3) replaying the base session's actions.
    JSON body:
      { "session_id": "base_session_id" }
    Returns: the fork's current snapshot + last (reward, done) observed during replay.
    """
    payload = request.get_json(force=True, silent=True) or {}
    base_sid = payload.get("session_id")
    if not base_sid or not isinstance(base_sid, str):
        return jsonify({"error": "Missing or invalid 'session_id'"}), 400

    with _sessions_lock:
        base_env = _sessions.get(base_sid)
    if base_env is None:
        return jsonify({"error": f"Unknown base session_id: {base_sid}"}), 404

    base_instr = base_env.state["instruction_text"]
    base_mode  = base_env.observation_mode
    base_hist  = list(_session_histories.get(base_sid, []))

    # Create a new env and reset with the SAME instruction
    fork_env = _make_env(observation_mode=base_mode, session_prefix="fork_")
    # Some server implementations read from shared_server.assigned_instruction_text
    shared_server.assigned_instruction_text = base_instr
    obs, _ = fork_env.reset(session=None, instruction_text=base_instr)
    shared_server.assigned_instruction_text = None

    # Replay actions to land at the same state
    last_reward = 0.0
    last_done = False
    for a in base_hist:
        _, last_reward, last_done, _ = fork_env.step(a)
        if last_done:
            break

    # Register fork session + copy its history (so future forks-from-fork work too)
    with _sessions_lock:
        _sessions[fork_env.session] = fork_env
    _session_histories[fork_env.session] = list(base_hist)

    snap = _snapshot(fork_env)
    return jsonify({
        "message": "session_forked",
        "base_session_id": base_sid,
        "reward": last_reward,
        "done": bool(last_done),
        **snap
    }), 200


@app.route("/api/end", methods=["POST"])
def api_end():
    payload = request.get_json(force=True, silent=True) or {}
    sid = payload.get("session_id")
    if not sid or not isinstance(sid, str):
        return jsonify({"error": "Missing or invalid 'session_id'"}), 400

    with _sessions_lock:
        existed = _sessions.pop(sid, None) is not None
    # NEW: also drop history
    _session_histories.pop(sid, None)

    return jsonify({
        "message": "session_deleted" if existed else "session_not_found",
        "session_id": sid
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
