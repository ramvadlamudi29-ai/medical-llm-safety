"""Minimal Streamlit dashboard for the AI Platform.

Run with::

    streamlit run ui/streamlit_app.py

Streamlit is an OPTIONAL dependency — the rest of the project does not require it.
"""
from __future__ import annotations
import asyncio
import sys
from pathlib import Path

# Ensure project root is importable when streamlit is launched from anywhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import streamlit as st  # type: ignore
except Exception as e:  # pragma: no cover - UI optional
    raise SystemExit(
        "Streamlit is not installed. Install with: pip install streamlit"
    ) from e

from core.monitor import metrics
from core.pipeline import AIPlatformPipeline


@st.cache_resource
def _pipeline() -> AIPlatformPipeline:
    return AIPlatformPipeline()


st.set_page_config(page_title="AI Platform", page_icon="🤖", layout="wide")
st.title("🤖 AI Platform Dashboard")

with st.sidebar:
    st.header("Settings")
    top_k = st.slider("top_k", 1, 10, 3)
    offline = st.checkbox("Offline-safe (heuristic LLM)", value=True)
    structured = st.checkbox("Structured JSON output", value=False)

query = st.text_area("Ask a question", "What is the zero trust policy?")
if st.button("Run", type="primary"):
    pipe = _pipeline()
    with st.spinner("Thinking..."):
        resp = asyncio.run(
            pipe.run(query, top_k=top_k, offline_safe=offline, structured=structured)
        )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Answer")
        st.write(resp.answer)
        st.caption(
            f"provider={resp.provider} · intent={resp.route.get('intent')}"
        )
    with col2:
        st.subheader("Citations")
        if resp.citations:
            for c in resp.citations:
                st.markdown(f"- **{c['title']}** (score={c['score']})")
        else:
            st.write("_(none)_")

    st.subheader("Run metadata")
    st.json(resp.meta)

st.divider()
st.subheader("System metrics")
st.json(metrics.snapshot())
