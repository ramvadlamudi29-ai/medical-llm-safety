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
import streamlit as st   # ✅ ADDED THIS LINE

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

# (rest of your code unchanged…)