"""
Dual-Domain Agentic RAG Platform — Premium Streamlit UI
Phase 6: Midnight FinTech Glass Theme with Agentic Intelligence
"""

import streamlit as st
import time
from rag_engine import AgenticRAG
from ingestion import DocumentIngestor

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Agentic RAG Platform",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
# PREMIUM CSS — MIDNIGHT FINTECH GLASS THEME
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* ── Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global Reset ── */
    *, *::before, *::after { font-family: 'Inter', sans-serif !important; }

    /* ── Main Background ── */
    .stApp {
        background: linear-gradient(160deg, #0a0e27 0%, #0d1333 40%, #111b44 100%);
        color: #e0e6f0;
    }

    /* ── Hide Streamlit Defaults ── */
    #MainMenu, footer { visibility: hidden; }
    header { background: transparent !important; }
    .block-container { padding-top: 2rem; max-width: 1200px; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1333 0%, #0a0e27 100%) !important;
        border-right: 1px solid rgba(0, 255, 180, 0.15);
    }
    /* Hide broken Material Icon text in sidebar collapse button */
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"],
    section[data-testid="stSidebar"] [data-testid="collapsedControl"],
    section[data-testid="stSidebar"] > div:first-child > button {
        font-size: 0px !important;
        color: transparent !important;
        width: 32px !important;
        height: 32px !important;
        overflow: hidden !important;
        background: rgba(0, 255, 180, 0.08) !important;
        border: 1px solid rgba(0, 255, 180, 0.15) !important;
        border-radius: 6px !important;
        position: relative !important;
    }
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"] *,
    section[data-testid="stSidebar"] > div:first-child > button * {
        font-size: 0px !important;
        visibility: hidden !important;
    }
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"]::after,
    section[data-testid="stSidebar"] > div:first-child > button::after {
        content: "✕";
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 14px;
        color: #5a6b82;
        visibility: visible !important;
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li,
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #c8d6e5 !important;
    }

    /* ── Glassmorphism Cards ── */
    .glass-card {
        background: rgba(13, 19, 51, 0.6);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(0, 255, 180, 0.12);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(0, 255, 180, 0.35);
        box-shadow: 0 8px 32px rgba(0, 255, 180, 0.08);
    }

    /* ── Metric Cards ── */
    .metric-row {
        display: flex;
        gap: 16px;
        margin-bottom: 24px;
        flex-wrap: wrap;
    }
    .metric-card {
        flex: 1;
        min-width: 150px;
        background: rgba(13, 19, 51, 0.7);
        backdrop-filter: blur(16px);
        border: 1px solid rgba(0, 255, 180, 0.12);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(0, 255, 180, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 255, 180, 0.1);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        background: linear-gradient(135deg, #00ffb4, #00d4aa, #0af);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .metric-label {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #7a8ba8;
        font-weight: 600;
    }

    /* ── Chat Bubbles ── */
    .chat-user {
        background: linear-gradient(135deg, rgba(0, 255, 180, 0.12), rgba(0, 170, 255, 0.08));
        border: 1px solid rgba(0, 255, 180, 0.2);
        border-radius: 18px 18px 4px 18px;
        padding: 16px 20px;
        margin: 8px 0;
        margin-left: 15%;
        color: #e0e6f0;
        animation: fadeInUp 0.4s ease-out;
    }
    .chat-ai {
        background: rgba(13, 19, 51, 0.5);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(100, 120, 180, 0.15);
        border-radius: 18px 18px 18px 4px;
        padding: 16px 20px;
        margin: 8px 0;
        margin-right: 10%;
        color: #d0d8e8;
        animation: fadeInUp 0.4s ease-out;
    }

    /* ── Tool Badge ── */
    .tool-badge {
        display: inline-block;
        background: rgba(0, 255, 180, 0.1);
        border: 1px solid rgba(0, 255, 180, 0.3);
        color: #00ffb4;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        letter-spacing: 0.5px;
        margin-right: 6px;
        margin-top: 8px;
    }

    /* ── Response Time Badge ── */
    .time-badge {
        display: inline-block;
        background: rgba(0, 170, 255, 0.1);
        border: 1px solid rgba(0, 170, 255, 0.25);
        color: #0af;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        margin-top: 8px;
    }

    /* ── Source Citation Box ── */
    .source-box {
        background: rgba(20, 30, 60, 0.5);
        border-left: 3px solid #00ffb4;
        border-radius: 0 10px 10px 0;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 13px;
        color: #9aa8c0;
    }

    /* ── Header Gradient Text ── */
    .hero-title {
        font-size: 32px;
        font-weight: 800;
        background: linear-gradient(135deg, #00ffb4, #0af, #7b61ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
        line-height: 1.2;
    }
    .hero-sub {
        font-size: 14px;
        color: #6a7b96;
        font-weight: 400;
        margin-bottom: 24px;
    }

    /* ── Sidebar Logo Title ── */
    .sidebar-logo {
        font-size: 22px;
        font-weight: 800;
        background: linear-gradient(135deg, #00ffb4, #0af);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 12px 0;
        margin-bottom: 8px;
    }
    .sidebar-divider {
        border: none;
        border-top: 1px solid rgba(0, 255, 180, 0.1);
        margin: 16px 0;
    }

    /* ── Upload Zone ── */
    .upload-zone {
        background: rgba(0, 255, 180, 0.03);
        border: 2px dashed rgba(0, 255, 180, 0.2);
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
        margin: 12px 0;
    }
    .upload-zone:hover {
        border-color: rgba(0, 255, 180, 0.5);
        background: rgba(0, 255, 180, 0.06);
    }

    /* ── File Uploader Styling ── */
    section[data-testid="stSidebar"] .stFileUploader {
        background: rgba(0, 255, 180, 0.03) !important;
        border: 2px dashed rgba(0, 255, 180, 0.2) !important;
        border-radius: 14px !important;
        padding: 12px !important;
        transition: all 0.3s ease !important;
    }
    section[data-testid="stSidebar"] .stFileUploader:hover {
        border-color: rgba(0, 255, 180, 0.5) !important;
        background: rgba(0, 255, 180, 0.06) !important;
    }
    section[data-testid="stSidebar"] .stFileUploader > div {
        background: transparent !important;
        border: none !important;
    }
    /* Fix: Hide duplicate button text and restyle */
    section[data-testid="stSidebar"] .stFileUploader button {
        background: rgba(0, 255, 180, 0.15) !important;
        border: 1px solid rgba(0, 255, 180, 0.3) !important;
        color: transparent !important;
        border-radius: 8px !important;
        font-size: 0px !important;
        padding: 6px 14px !important;
        position: relative !important;
        min-height: 32px !important;
        overflow: hidden !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    section[data-testid="stSidebar"] .stFileUploader button * {
        font-size: 0px !important;
        visibility: hidden !important;
        width: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }
    section[data-testid="stSidebar"] .stFileUploader button::after {
        content: "Browse Files";
        color: #00ffb4;
        font-size: 13px;
        font-weight: 600;
        font-family: 'Inter', sans-serif;
        visibility: visible !important;
    }
    section[data-testid="stSidebar"] .stFileUploader button:hover {
        background: rgba(0, 255, 180, 0.25) !important;
    }
    section[data-testid="stSidebar"] .stFileUploader small {
        color: #5a6b82 !important;
    }

    /* ── Animations ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .processing { animation: pulse 1.5s ease-in-out infinite; color: #00ffb4; }

    /* ── Streamlit Overrides ── */
    .stChatInput > div { 
        background: rgba(13, 19, 51, 0.8) !important;
        border: 1px solid rgba(0, 255, 180, 0.15) !important;
        border-radius: 14px !important;
    }
    .stChatInput textarea {
        color: #e0e6f0 !important;
        background: transparent !important;
    }
    /* Fix: Hide 'keyboard_double' icon text in chat send button — all states */
    .stChatInput button {
        background: rgba(0, 255, 180, 0.1) !important;
        border: 1px solid rgba(0, 255, 180, 0.2) !important;
        border-radius: 8px !important;
        width: 36px !important;
        height: 36px !important;
        overflow: hidden !important;
        position: relative !important;
    }
    .stChatInput button:hover {
        background: rgba(0, 255, 180, 0.2) !important;
        border-color: rgba(0, 255, 180, 0.4) !important;
    }
    .stChatInput button * {
        font-size: 0px !important;
        color: transparent !important;
        visibility: hidden !important;
    }
    .stChatInput button::after {
        content: "➤";
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 16px;
        color: #00ffb4;
        visibility: visible !important;
    }
    .stRadio > div { gap: 8px; }
    .stRadio label {
        background: rgba(13, 19, 51, 0.5) !important;
        border: 1px solid rgba(100, 120, 180, 0.15) !important;
        border-radius: 10px !important;
        padding: 8px 16px !important;
        color: #c8d6e5 !important;
        transition: all 0.2s ease !important;
    }
    .stRadio label:hover {
        border-color: rgba(0, 255, 180, 0.3) !important;
    }
    .stExpander {
        background: rgba(13, 19, 51, 0.4) !important;
        border: 1px solid rgba(100, 120, 180, 0.1) !important;
        border-radius: 12px !important;
    }
    .stAlert { border-radius: 12px !important; }
    div[data-testid="stChatMessage"] { background: transparent !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# INITIALIZE ENGINE (cached)
# ══════════════════════════════════════════════════════════════
@st.cache_resource
def load_rag_engine():
    return AgenticRAG()

@st.cache_resource
def load_ingestor():
    return DocumentIngestor()

rag = load_rag_engine()
ingestor = load_ingestor()


# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sidebar-logo">Agentic RAG</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align:center; color:#5a6b82; font-size:12px; margin-top:-8px;">Enterprise AI Platform</p>', unsafe_allow_html=True)
    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ── PDF Upload Zone ──
    st.markdown("### Upload Documents")
    uploaded_file = st.file_uploader(
        "Drop a PDF to teach the AI",
        type=["pdf"],
    )

    if uploaded_file:
        upload_domain = st.selectbox(
            "Classify this document as:",
            ["healthcare", "finance"],
            key="upload_domain"
        )
        if st.button("Ingest Document", use_container_width=True):
            with st.spinner(""):
                st.markdown('<p class="processing">Processing document...</p>', unsafe_allow_html=True)
                file_bytes = uploaded_file.read()
                result = ingestor.ingest_uploaded_pdf(file_bytes, uploaded_file.name, upload_domain)

                if result["success"]:
                    st.success(f"Ingested: {result['pages']} pages, {result['chunks']} chunks")
                    st.cache_resource.clear()
                else:
                    st.error(result.get("error", "Ingestion failed."))

    st.markdown('<hr class="sidebar-divider">', unsafe_allow_html=True)

    # ── Clear Chat ──
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ══════════════════════════════════════════════════════════════
# MAIN CONTENT — HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-title">Dual-Domain Agentic RAG Platform</div>
<div class="hero-sub">Autonomous AI Agent with Medical DB, Finance DB & Live Web Search powered by Tavily AI</div>
""", unsafe_allow_html=True)

# ── Metric Cards ──
stats = rag.get_db_stats()
st.markdown(f"""
<div class="metric-row">
    <div class="metric-card">
        <div class="metric-value">{stats['total_chunks']}</div>
        <div class="metric-label">Chunks Indexed</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">3</div>
        <div class="metric-label">Agent Tools</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">2</div>
        <div class="metric-label">Domains Active</div>
    </div>
    <div class="metric-card">
        <div class="metric-value">Llama 3.1</div>
        <div class="metric-label">LLM Engine</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# CHAT INTERFACE
# ══════════════════════════════════════════════════════════════

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Render chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="chat-ai">{msg["content"]}</div>', unsafe_allow_html=True)
        # Render metadata badges
        if "tools_used" in msg:
            badges_html = ""
            for tool in msg["tools_used"]:
                badges_html += f'<span class="tool-badge">{tool}</span>'
            if "response_time" in msg:
                badges_html += f'<span class="time-badge">{msg["response_time"]}s</span>'
            st.markdown(badges_html, unsafe_allow_html=True)
        # Render source citations
        if "sources" in msg and msg["sources"]:
            with st.expander("View Source Citations", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"""
                    <div class="source-box">
                        <strong style="color: #00ffb4;">Tool:</strong> {src['tool']}<br>
                        <strong style="color: #0af;">Excerpt:</strong> {src['content']}
                    </div>
                    """, unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask anything... Medical, Finance, or Live Web"):
    # Add and render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f'<div class="chat-user">{prompt}</div>', unsafe_allow_html=True)

    # Generate AI response
    with st.spinner("Agent is thinking..."):
        result = rag.ask(prompt)

    # Render AI response
    st.markdown(f'<div class="chat-ai">{result["answer"]}</div>', unsafe_allow_html=True)

    # Render tool badges
    badges_html = ""
    for tool in result["tools_used"]:
        badges_html += f'<span class="tool-badge">{tool}</span>'
    badges_html += f'<span class="time-badge">{result["response_time"]}s</span>'
    st.markdown(badges_html, unsafe_allow_html=True)

    # Render source citations
    if result["sources"]:
        with st.expander("View Source Citations", expanded=False):
            for src in result["sources"]:
                st.markdown(f"""
                <div class="source-box">
                    <strong style="color: #00ffb4;">Tool:</strong> {src['tool']}<br>
                    <strong style="color: #0af;">Excerpt:</strong> {src['content']}
                </div>
                """, unsafe_allow_html=True)

    # Save to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "tools_used": result["tools_used"],
        "sources": result["sources"],
        "response_time": result["response_time"]
    })
