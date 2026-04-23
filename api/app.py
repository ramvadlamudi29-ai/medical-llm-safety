"""
AI Platform — Modern Chat UI (Streamlit)
========================================

Run locally:
    streamlit run ui/app.py

Requires the FastAPI backend running (default: http://localhost:8000).
This UI does NOT modify or depend on backend internals — it only calls /query.

Env vars:
    AI_PLATFORM_API_URL   default: http://localhost:8000
    AI_PLATFORM_API_KEY   optional, sent as x-api-key header
"""
from __future__ import annotations

import os
import time
import json
from typing import Any, Dict, List, Optional

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_URL = os.getenv("AI_PLATFORM_API_URL", "http://localhost:8000").rstrip("/")
API_KEY = os.getenv("AI_PLATFORM_API_KEY", "")
QUERY_ENDPOINT = f"{API_URL}/query"
HEALTH_ENDPOINT = f"{API_URL}/health"
REQUEST_TIMEOUT = 60

st.set_page_config(
    page_title="AI Platform — Chat",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
:root {
    --bubble-user: #2563eb;
    --bubble-user-text: #ffffff;
    --bubble-assistant: #f1f5f9;
    --bubble-assistant-text: #0f172a;
    --muted: #64748b;
    --card-bg: #ffffff;
    --card-border: #e2e8f0;
    --accent: #6366f1;
}
[data-theme="dark"] {
    --bubble-assistant: #1e293b;
    --bubble-assistant-text: #f1f5f9;
    --card-bg: #0f172a;
    --card-border: #1e293b;
    --muted: #94a3b8;
}
.block-container { padding-top: 1.2rem; padding-bottom: 6rem; max-width: 1100px; }

.chat-row { display: flex; margin: 8px 0; }
.chat-row.user { justify-content: flex-end; }
.chat-row.assistant { justify-content: flex-start; }

.bubble {
    max-width: 78%;
    padding: 12px 16px;
    border-radius: 16px;
    line-height: 1.55;
    font-size: 0.97rem;
    word-wrap: break-word;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    white-space: pre-wrap;
}
.bubble.user {
    background: var(--bubble-user);
    color: var(--bubble-user-text);
    border-bottom-right-radius: 4px;
}
.bubble.assistant {
    background: var(--bubble-assistant);
    color: var(--bubble-assistant-text);
    border-bottom-left-radius: 4px;
}
.role-label {
    font-size: 0.72rem;
    color: var(--muted);
    margin: 0 6px 2px 6px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.metric-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 12px;
    padding: 10px 14px;
    margin-bottom: 8px;
}
.metric-label { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
.metric-value { font-size: 1.05rem; font-weight: 600; margin-top: 2px; word-break: break-all; }

.citation-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-left: 3px solid var(--accent);
    border-radius: 8px;
    padding: 10px 12px;
    margin-bottom: 8px;
}
.cite-title { font-weight: 600; font-size: 0.95rem; }
.cite-meta { font-size: 0.78rem; color: var(--muted); margin-top: 4px; }
.score-pill {
    display: inline-block;
    background: var(--accent);
    color: white;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-left: 6px;
}

.typing { display: inline-flex; gap: 4px; padding: 6px 0; }
.typing span {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--muted);
    animation: blink 1.2s infinite ease-in-out;
}
.typing span:nth-child(2) { animation-delay: 0.2s; }
.typing span:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink { 0%, 80%, 100% { opacity: 0.3; } 40% { opacity: 1; } }

.status-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 6px; }
.status-ok { background: #10b981; }
.status-bad { background: #ef4444; }

.error-box {
    background: #fef2f2;
    border: 1px solid #fecaca;
    color: #991b1b;
    padding: 12px 14px;
    border-radius: 10px;
    font-size: 0.9rem;
}
[data-theme="dark"] .error-box {
    background: #450a0a; border-color: #7f1d1d; color: #fecaca;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def _init_state() -> None:
    ss = st.session_state
    ss.setdefault("messages", [])  # list of {role, content, meta?, citations?, provider?, route?, latency?, error?}
    ss.setdefault("is_loading", False)
    ss.setdefault("pending_query", None)

_init_state()

# ---------------------------------------------------------------------------
# Backend client
# ---------------------------------------------------------------------------
def check_health() -> Dict[str, Any]:
    try:
        r = requests.get(HEALTH_ENDPOINT, timeout=5,
                         headers={"x-api-key": API_KEY} if API_KEY else None)
        if r.status_code == 200:
            return {"ok": True, "data": r.json()}
        return {"ok": False, "error": f"HTTP {r.status_code}"}
    except requests.RequestException as e:
        return {"ok": False, "error": str(e)}


def call_query(query: str, top_k: int, offline_safe: bool) -> Dict[str, Any]:
    """Call backend /query. Returns dict with parsed response or error."""
    payload = {"query": query, "top_k": top_k, "offline_safe": offline_safe}
    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["x-api-key"] = API_KEY
    started = time.time()
    try:
        r = requests.post(QUERY_ENDPOINT, json=payload, headers=headers, timeout=REQUEST_TIMEOUT)
        latency = time.time() - started
    except requests.Timeout:
        return {"ok": False, "error": "Request timed out. The backend took too long to respond."}
    except requests.ConnectionError:
        return {"ok": False, "error": f"Could not connect to backend at {API_URL}. Is it running?"}
    except requests.RequestException as e:
        return {"ok": False, "error": f"Network error: {e}"}

    try:
        data = r.json()
    except json.JSONDecodeError:
        return {"ok": False, "error": f"Backend returned non-JSON response (HTTP {r.status_code})."}

    if r.status_code >= 400 or not data.get("ok", False):
        msg = data.get("error") or data.get("detail") or f"HTTP {r.status_code}"
        return {"ok": False, "error": str(msg), "raw": data, "latency": latency}

    data["_latency"] = latency
    return data

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🤖 AI Platform")
    st.caption("Production-grade RAG · Routing · Caching")

    health = check_health()
    if health["ok"]:
        st.markdown(
            f'<div><span class="status-dot status-ok"></span>'
            f'Backend online · <code>v{health["data"].get("version","?")}</code></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div><span class="status-dot status-bad"></span>Backend offline</div>',
            unsafe_allow_html=True,
        )
        st.caption(health["error"])

    st.divider()
    st.markdown("#### ⚙️ Query Settings")
    top_k = st.slider("Top-K retrieved docs", 1, 10, 3,
                      help="How many documents to retrieve from the knowledge base.")
    offline_safe = st.toggle("Offline-safe mode", value=True,
                             help="Use heuristic LLM (no external API calls).")

    st.divider()
    st.markdown("#### 🎨 Display")
    show_metadata = st.toggle("Show metadata panel", value=True)
    stream_effect = st.toggle("Typing animation", value=True)

    st.divider()
    st.markdown("#### 🔌 Connection")
    st.code(API_URL, language=None)

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🗑️ Clear", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col_b:
        if st.button("🔄 Reload", use_container_width=True):
            st.rerun()

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("## 💬 AI Platform Chat")
st.caption("Ask anything — answers are grounded in your indexed knowledge base with citations, routing, and caching.")

# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------
def render_message(msg: Dict[str, Any], idx: int) -> None:
    role = msg["role"]
    content = msg.get("content", "")
    if role == "user":
        st.markdown(
            f'<div class="chat-row user"><div>'
            f'<div class="role-label" style="text-align:right;">You</div>'
            f'<div class="bubble user">{_escape(content)}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        return

    # assistant
    if msg.get("error"):
        st.markdown(
            f'<div class="chat-row assistant"><div style="width:78%;">'
            f'<div class="role-label">Assistant · error</div>'
            f'<div class="error-box">⚠️ {_escape(msg["error"])}</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f'<div class="chat-row assistant"><div style="max-width:78%;">'
        f'<div class="role-label">Assistant</div>'
        f'<div class="bubble assistant">{_escape(content)}</div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    # Citations
    citations = msg.get("citations") or []
    if citations:
        with st.expander(f"📚 Citations ({len(citations)})", expanded=False):
            for c in citations:
                title = c.get("title", c.get("id", "Untitled"))
                score = c.get("score", 0.0)
                cid = c.get("id", "")
                st.markdown(
                    f'<div class="citation-card">'
                    f'<div class="cite-title">{_escape(str(title))}'
                    f'<span class="score-pill">{float(score):.3f}</span></div>'
                    f'<div class="cite-meta">id: <code>{_escape(str(cid))}</code></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Metadata panel
    if show_metadata and (msg.get("meta") or msg.get("provider")):
        with st.expander("📊 Run details", expanded=False):
            meta = msg.get("meta") or {}
            route = msg.get("route") or {}
            cols = st.columns(4)
            cards = [
                ("Latency", f'{msg.get("latency", 0):.2f}s'),
                ("Provider", msg.get("provider", "—")),
                ("Cache", "HIT ✅" if meta.get("cache_hit") else "MISS"),
                ("Intent", route.get("intent", "—")),
            ]
            for col, (label, value) in zip(cols, cards):
                with col:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-label">{label}</div>'
                        f'<div class="metric-value">{_escape(str(value))}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            trace_id = meta.get("request_id") or meta.get("trace_id")
            if trace_id:
                st.markdown(
                    f'<div class="metric-card"><div class="metric-label">Trace ID</div>'
                    f'<div class="metric-value"><code>{_escape(str(trace_id))}</code></div></div>',
                    unsafe_allow_html=True,
                )

            with st.expander("Raw meta JSON", expanded=False):
                st.json({"meta": meta, "route": route})

    # Copy button
    if content:
        with st.expander("📋 Copy answer", expanded=False):
            st.code(content, language=None)


def _escape(s: str) -> str:
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )


# Render all messages
for i, m in enumerate(st.session_state.messages):
    render_message(m, i)

# ---------------------------------------------------------------------------
# Pending request handler (typing indicator + streaming effect)
# ---------------------------------------------------------------------------
if st.session_state.pending_query is not None:
    user_q = st.session_state.pending_query
    st.session_state.pending_query = None

    # Typing indicator placeholder
    placeholder = st.empty()
    placeholder.markdown(
        '<div class="chat-row assistant"><div>'
        '<div class="role-label">Assistant</div>'
        '<div class="bubble assistant"><div class="typing">'
        '<span></span><span></span><span></span></div></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )

    result = call_query(user_q, top_k=top_k, offline_safe=offline_safe)

    if not result.get("ok"):
        placeholder.empty()
        st.session_state.messages.append({
            "role": "assistant",
            "content": "",
            "error": result.get("error", "Unknown error"),
        })
    else:
        answer = result.get("answer", "")
        # Streaming/typing effect
        if stream_effect and answer:
            tokens = answer.split(" ")
            shown = ""
            for i, tok in enumerate(tokens):
                shown = shown + (" " if i else "") + tok
                placeholder.markdown(
                    f'<div class="chat-row assistant"><div style="max-width:78%;">'
                    f'<div class="role-label">Assistant</div>'
                    f'<div class="bubble assistant">{_escape(shown)}▌</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
                # Keep total animation snappy regardless of length
                time.sleep(min(0.025, 1.5 / max(len(tokens), 1)))
        placeholder.empty()

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "citations": result.get("citations", []),
            "meta": result.get("meta", {}),
            "route": result.get("route", {}),
            "provider": result.get("provider", "—"),
            "latency": result.get("_latency", 0.0),
        })

    st.session_state.is_loading = False
    st.rerun()

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
prompt = st.chat_input(
    "Ask the AI platform anything…",
    disabled=st.session_state.is_loading,
)
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.pending_query = prompt
    st.session_state.is_loading = True
    st.rerun()

# Empty-state hint
if not st.session_state.messages:
    st.info(
        "👋 Try: *“What is the zero trust policy?”* — answers include citations, "
        "latency, cache status, and trace IDs.",
        icon="💡",
    )
