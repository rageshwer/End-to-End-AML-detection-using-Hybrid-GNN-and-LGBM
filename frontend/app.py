import io
import streamlit as st
import PIL.Image
PIL.Image.MAX_IMAGE_PIXELS = None
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import time
import os
import requests
# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="AML Detection Using Hybrid GNN and LGBM",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)
API_URL = os.getenv("API_URL", "http://localhost:8000")
# ============================================================
# 2. GLOBAL STYLE — Light, gradient-hero SaaS aesthetic
#    Palette: coral-red + violet accents on white/ink
# ============================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] { font-family: 'Manrope', -apple-system, sans-serif; }

    :root {
        --red: #FF4438;
        --red-dark: #E0301F;
        --violet: #8B5CF6;
        --ink: #14141A;
        --gray-600: #6B7280;
        --gray-500: #9096A2;
        --gray-100: #F5F5F7;
        --gray-200: #ECECF1;
        --border: rgba(15,15,20,0.09);
        --surface: #FFFFFF;
    }

    .stApp { background: var(--surface); }
    #MainMenu, footer, header {visibility: hidden;}
    .block-container { padding-top: 1.2rem; padding-bottom: 0; max-width: 100%; }

    /* ---------- HERO ---------- */
    .hero-wrap {
        position: relative;
        margin: -1.2rem -4rem 0 -4rem;
        padding: 4.5rem 4rem 5rem 4rem;
        text-align: center;
        overflow: hidden;
        background:
            radial-gradient(ellipse 55% 60% at 22% 15%, rgba(255,150,190,0.55), transparent 60%),
            radial-gradient(ellipse 50% 55% at 78% 5%, rgba(180,150,255,0.45), transparent 60%),
            radial-gradient(ellipse 60% 50% at 50% 100%, rgba(255,190,150,0.30), transparent 65%),
            #FFFFFF;
    }
    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255,255,255,0.7);
        border: 1px solid var(--border);
        padding: 6px 16px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 700;
        color: var(--ink);
        margin-bottom: 1.6rem;
        letter-spacing: 0.2px;
    }
    .hero-eyebrow .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--red); }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        line-height: 1.12;
        color: var(--ink);
        margin: 0 auto 1.2rem auto;
        max-width: 780px;
    }
    .hero-title .accent { color: var(--red); }
    .hero-subtitle {
        font-size: 1.12rem;
        color: var(--gray-600);
        max-width: 560px;
        margin: 0 auto 2.2rem auto;
        line-height: 1.6;
    }
    .hero-cta-row { display: flex; justify-content: center; gap: 1.4rem; align-items: center; margin-bottom: 3.2rem; }
    .cta-black {
        background: var(--ink); color: #fff; padding: 0.85rem 1.8rem; border-radius: 10px;
        font-weight: 700; font-size: 0.95rem; text-decoration: none; display: inline-block;
        box-shadow: 0 10px 24px -10px rgba(0,0,0,0.35);
    }
    .cta-link { color: var(--ink); font-weight: 700; font-size: 0.95rem; }

    /* ---------- FLOATING PREVIEW CARD ---------- */
    .preview-card {
        position: relative;
        background: #fff;
        border-radius: 20px;
        border: 1px solid var(--border);
        box-shadow: 0 40px 80px -30px rgba(20,20,26,0.25);
        padding: 1.8rem 2rem 2rem 2rem;
        max-width: 980px;
        margin: 0 auto;
        text-align: left;
    }
    .preview-label {
        font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1.6px;
        font-weight: 700; color: var(--gray-500); margin-bottom: 1rem;
    }

    /* ---------- Generic sections ---------- */
    .section {
        padding: 4.5rem 4rem;
        margin: 0 -4rem;
    }
    .section-gray { background: var(--gray-100); }
    .section-white { background: #fff; }
    .section-dark {
        background: linear-gradient(165deg, #0A0410 0%, #1A0B2A 55%, #170A22 100%);
        position: relative;
        overflow: hidden;
        color: #F1EEF8;
    }
    .section-dark::before {
        content: "";
        position: absolute; top: -20%; left: 10%;
        width: 600px; height: 600px;
        background: radial-gradient(circle, rgba(139,92,246,0.28), transparent 70%);
        pointer-events: none;
    }
    .section-dark::after {
        content: "";
        position: absolute; bottom: -30%; right: 5%;
        width: 500px; height: 500px;
        background: radial-gradient(circle, rgba(255,68,56,0.18), transparent 70%);
        pointer-events: none;
    }
    .section-header { text-align: center; margin-bottom: 3rem; position: relative; z-index: 1; }
    .kicker {
        text-transform: uppercase; letter-spacing: 2.2px; font-size: 0.76rem; font-weight: 800;
        color: var(--red); margin-bottom: 0.7rem;
    }
    .section-dark .kicker { color: #FF8A7D; }
    .section-h2 { font-size: 2rem; font-weight: 800; letter-spacing: -0.8px; color: var(--ink); }
    .section-dark .section-h2 { color: #F8F6FC; }
    .section-h2 .accent { color: var(--red); }
    .section-sub { color: var(--gray-600); font-size: 1rem; max-width: 520px; margin: 0.7rem auto 0 auto; }
    .section-dark .section-sub { color: #C4BBD6; }

    /* ---------- Stats ---------- */
    .stats-row { display: flex; justify-content: center; gap: 5rem; text-align: center; position: relative; z-index: 1; }
    .stat-num {
        font-size: 2.6rem; font-weight: 800; color: var(--ink); letter-spacing: -1px;
        font-family: 'JetBrains Mono', monospace;
    }
    .stat-label { color: var(--gray-500); font-size: 0.85rem; font-weight: 600; margin-top: 0.3rem; }

    /* ---------- Icon grid ---------- */
    .icon-grid {
        display: grid; grid-template-columns: repeat(3, 1fr); gap: 2.2rem 3rem;
        max-width: 980px; margin: 0 auto; position: relative; z-index: 1;
    }
    .icon-item { display: flex; gap: 14px; align-items: flex-start; }
    .icon-badge {
        width: 38px; height: 38px; min-width: 38px; border-radius: 10px;
        background: rgba(255,68,56,0.1); color: var(--red);
        display: flex; align-items: center; justify-content: center; font-size: 1.1rem;
    }
    .section-dark .icon-badge { background: rgba(139,92,246,0.18); color: #C9B8FF; }
    .icon-title { font-weight: 700; color: var(--ink); font-size: 0.98rem; margin-bottom: 0.2rem; }
    .section-dark .icon-title { color: #F1EEF8; }
    .icon-desc { color: var(--gray-600); font-size: 0.88rem; line-height: 1.5; }
    .section-dark .icon-desc { color: #B3A9C6; }

    /* ---------- Dark panel mock (graph) ---------- */
    .graph-panel {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.6rem;
        backdrop-filter: blur(6px);
        position: relative;
        z-index: 1;
    }

    /* ---------- Analysis / result cards (light) ---------- */
    .app-card {
        background: #fff;
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 2rem 2.2rem;
        box-shadow: 0 20px 50px -30px rgba(20,20,26,0.2);
        max-width: 980px;
        margin: 0 auto 2rem auto;
    }
    .app-card .kicker { text-align: left; }
    .app-card .card-title { font-size: 1.3rem; font-weight: 800; color: var(--ink); margin-bottom: 1rem; }

    .metric-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; }
    .metric-tile {
        background: var(--gray-100); border-radius: 12px; padding: 1rem 1.2rem; border: 1px solid var(--border);
    }
    .metric-tile .metric-label { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 1.1px; color: var(--gray-500); font-weight: 700; margin-bottom: 0.35rem; }
    .metric-tile .metric-value { font-size: 1.7rem; font-weight: 800; color: var(--ink); font-family: 'JetBrains Mono', monospace; }
    .metric-bar { height: 4px; background: var(--gray-200); border-radius: 2px; margin-top: 0.6rem; overflow: hidden; }
    .metric-bar-fill { height: 100%; background: linear-gradient(90deg, var(--red), var(--violet)); border-radius: 2px; }

    .stTextInput input {
        background: var(--gray-100) !important; border: 1px solid var(--border) !important;
        border-radius: 10px !important; color: var(--ink) !important;
        font-family: 'JetBrains Mono', monospace !important; padding: 0.7rem 1rem !important;
    }
    .stTextInput input:focus { border-color: var(--red) !important; box-shadow: 0 0 0 3px rgba(255,68,56,0.14) !important; }
    .stTextInput label { color: var(--gray-600) !important; font-weight: 700 !important; font-size: 0.85rem !important; }

    div.stButton > button {
        background: var(--ink) !important; color: #fff !important; border: none !important;
        border-radius: 10px !important; font-weight: 700 !important; padding: 0.75rem 1rem !important;
        box-shadow: 0 10px 22px -10px rgba(0,0,0,0.35); transition: all 0.2s ease !important;
    }
    div.stButton > button:hover { transform: translateY(-1px); background: var(--red-dark) !important; }

    .verdict-banner {
        border-radius: 12px; padding: 1rem 1.3rem; font-weight: 700; font-size: 1rem;
        display: flex; align-items: center; gap: 10px; margin-top: 1rem; border: 1px solid;
    }
    .verdict-flag { background: rgba(255,68,56,0.08); border-color: rgba(255,68,56,0.3); color: var(--red-dark); }
    .verdict-pass { background: rgba(139,92,246,0.08); border-color: rgba(139,92,246,0.3); color: #6D3FD1; }

    /* ---------- Footer ---------- */
    .footer-section {
        background: #0A0410; color: #B3A9C6; padding: 3rem 4rem 2rem 4rem; margin: 0 -4rem -3rem -4rem;
        font-size: 0.85rem;
    }
    .footer-brand { color: #fff; font-weight: 800; font-size: 1.05rem; margin-bottom: 0.5rem; }
    .footer-bottom { border-top: 1px solid rgba(255,255,255,0.08); margin-top: 2rem; padding-top: 1.2rem; text-align: center; color: #6B6479; font-size: 0.78rem; }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 3. HERO
# ============================================================
st.markdown("""
<div class="hero-wrap">
<div class="hero-eyebrow"><span class="dot"></span>Financial Crime Intelligence</div>
<div class="hero-title" style="max-width: 950px; margin: 0 auto 1.2rem auto;"><span class="accent">Graph-native</span> AML Detection, Explainability and Monitoring</div>
<div class="hero-subtitle" style="max-width: 800px; margin: 0 auto 2.2rem auto;">This prototype is based on Elliptic bitcoin datasets converted into graph. It consists of more than 2 Lakh transactions acting as nodes and more than 2.3 Lakh edges between them that correspond to the flow of amount. The data is highly imbalanced with only 22.8% data being labelled (2.2% being illicit).</div>
<div class="hero-cta-row">
<a href="#analysis-section" style="color: #1E3A8A !important; text-decoration: none; font-weight: 800;">Click to Analyze a Node ↓</a>
</div>
</div>
""", unsafe_allow_html=True)

# ---- Floating preview card (model metrics as a live "dashboard" mock) ----
metrics = [
    ("PR-AUC", "0.71", 71),
    ("Illicit F1-Score", "0.69", 69),
    ("Illicit Recall", "58.0%", 58.0),
    ("Illicit Precision", "87.0%", 87.0),
]
metric_html = '<div class="metric-grid">'
for label, value, pct in metrics:
    metric_html += f"""
    <div class="metric-tile">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-bar"><div class="metric-bar-fill" style="width:{pct}%;"></div></div>
    </div>"""
metric_html += "</div>"

st.markdown(f"""
<div style="margin-top:-3.6rem; padding: 0 4rem 3rem 4rem; position:relative; z-index:2;">
    <div class="preview-card">
        <div class="preview-label">Production Model · Metrics</div>
        {metric_html}
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 4. STATS BAND
# ============================================================
st.markdown("""
<div class="section section-white" style="padding-top:1rem;">
    <div class="section-header">
        <div class="kicker">Ready for production</div>
        <div class="section-h2">The GNN behind every score</div>
        <div class="section-sub">A single model evaluates every transaction against the full shape of the financial network — not just its own attributes.</div>
    </div>
    <div class="stats-row">
        <div><div class="stat-num">2 L+</div><div class="stat-label">Nodes scored</div></div>
        <div><div class="stat-num">2-hop</div><div class="stat-label">Neighborhood depth</div></div>
        <div><div class="stat-num">15-20ms</div><div class="stat-label">Median inference time</div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 5. GRAY SECTION — Why manual review doesn't scale
# ============================================================
st.markdown("""
<div class="section section-gray">
    <div class="section-header">
        <div class="kicker">Why GNN evaluations matter</div>
        <div class="section-h2">Manual review breaks <span class="accent">differently</span></div>
        <div class="section-sub">Laundering rings hide in relationships, not single transactions — which is exactly what a graph model is built to catch.</div>
    </div>
    <div class="icon-grid">
        <div class="icon-item"><div class="icon-badge">🕸️</div><div><div class="icon-title">Topological features</div><div class="icon-desc">Risk lives in the shape of the network, not one row of data.</div></div></div>
        <div class="icon-item"><div class="icon-badge">🔁</div><div><div class="icon-title">Multi-hop aggregation</div><div class="icon-desc">Signals propagate through 2nd and 3rd-degree connections.</div></div></div>
        <div class="icon-item"><div class="icon-badge">🧬</div><div><div class="icon-title">Node embeddings</div><div class="icon-desc">Precomputed vectors encode relationships humans can't map by hand.</div></div></div>
        <div class="icon-item"><div class="icon-badge">⚡</div><div><div class="icon-title">Real-time scoring</div><div class="icon-desc">Sub-second inference keeps pace with live transaction volume.</div></div></div>
        <div class="icon-item"><div class="icon-badge">🔍</div><div><div class="icon-title">Explainability</div><div class="icon-desc">Every score comes with a feature-level breakdown of the "why."</div></div></div>
        <div class="icon-item"><div class="icon-badge">📈</div><div><div class="icon-title">Performance monitoring</div><div class="icon-desc">Laundering patterns shift — the model's metrics are watched continuously.</div></div></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 6. DARK SECTION — How it works
# ============================================================
st.html("""
<div class="section section-dark">
    <div class="section-header">
        <div class="kicker">How it works</div>
        <div class="section-h2">From raw transactions to a risk score</div>
    </div>
    
    <div class="icon-grid" style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 3rem; max-width: 1000px; margin: 2rem auto 0 auto;">
        
        <div style="display: flex; gap: 1.5rem; align-items: flex-start;">
            <div class="icon-badge" style="width: 48px; height: 48px; font-size: 1.2rem; flex-shrink: 0;">①</div>
            <div>
                <div class="icon-title" style="font-size: 1.1rem; margin-bottom: 0.5rem; color: #fff;">Graph Topology & Embeddings</div>
                <div class="icon-desc" style="font-size: 0.9rem; line-height: 1.6; color: #B3A9C6;">
                    Every transaction is a node, edges define flows. A 2-layer Hybrid-GNN (GraphSAGE and GAT) reasons over this topology to generate 256-D vector embeddings for every transaction.
                </div>
            </div>
        </div>

        <div style="display: flex; gap: 1.5rem; align-items: flex-start;">
            <div class="icon-badge" style="width: 48px; height: 48px; font-size: 1.2rem; flex-shrink: 0;">②</div>
            <div>
                <div class="icon-title" style="font-size: 1.1rem; margin-bottom: 0.5rem; color: #fff;">Chronological Training</div>
                <div class="icon-desc" style="font-size: 0.9rem; line-height: 1.6; color: #B3A9C6;">
                    Embeddings are merged with 165-D raw features into a 421-D vector for the LGBMClassifier. This pipeline achieves a leak-proof illicit F1 score of 0.69.
                </div>
            </div>
        </div>

        <div style="display: flex; gap: 1.5rem; align-items: flex-start;">
            <div class="icon-badge" style="width: 48px; height: 48px; font-size: 1.2rem; flex-shrink: 0;">③</div>
            <div>
                <div class="icon-title" style="font-size: 1.1rem; margin-bottom: 0.5rem; color: #fff;">Inference & SHAP Explainability</div>
                <div class="icon-desc" style="font-size: 0.9rem; line-height: 1.6; color: #B3A9C6;">
                    The API calculates risk probability and SHAP values. A dynamic waterfall plot unpacks the score, attributing risk shifts to local, neighborhood, and structural features.
                </div>
            </div>
        </div>

        <div style="display: flex; gap: 1.5rem; align-items: flex-start;">
            <div class="icon-badge" style="width: 48px; height: 48px; font-size: 1.2rem; flex-shrink: 0;">④</div>
            <div>
                <div class="icon-title" style="font-size: 1.1rem; margin-bottom: 0.5rem; color: #fff;">Monitoring</div>
                <div class="icon-desc" style="font-size: 0.9rem; line-height: 1.6; color: #B3A9C6;">
                    Continuous performance validation. Trigger on-demand model audits against 5,000 sampled test transactions to ensure metrics hold against historical benchmarks.
                </div>
            </div>
        </div>

    </div>
</div>
""")

# ============================================================
# 7. INPUT SECTION
# ============================================================
st.markdown('<div class="section section-white" style="padding-bottom: 1rem;">', unsafe_allow_html=True)

st.markdown('<div class="kicker">Analyze</div><div class="card-title">Evaluate a Transaction Node</div>', unsafe_allow_html=True)
st.markdown('<div id="analysis-section"></div>', unsafe_allow_html=True)
col_input, col_btn, col_empty = st.columns([2, 1, 2])
with col_input:
    test_id = st.text_input("Enter test Node ID between 0 and 16669 :", value="1024")
with col_btn:
    st.markdown("<div style='height: 1.6rem;'></div>", unsafe_allow_html=True)
    analyze_btn = st.button("Analyze Risk", type="primary", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 8. RESULTS SECTION
# ============================================================
if analyze_btn:
    if not test_id:
        st.warning("Please enter a valid ID.")
    else:
        with st.spinner("Querying GNN API and calculating explainability..."):
            start_t=time.perf_counter()
            response=requests.post(f'{API_URL}/predict',params={'id':test_id})
            end_t=time.perf_counter()
            latency=(end_t-start_t)*1000
            if response.status_code == 200:
                result=response.json()
                prob=result['final_prob']
                is_high_risk = prob > 0.08

                st.markdown(f'<div class="kicker">Latency</div><div class="card-title">The inference and SHAP retrieval done in : {latency:.2f} ms</div>', unsafe_allow_html=True)
                st.markdown('<div class="section section-white" style="padding-bottom: 0.5rem;">', unsafe_allow_html=True)
                res_col1, res_col2 = st.columns([1, 1.6], gap="large")

                # -------- Risk Gauge --------
                with res_col1:
                    st.markdown('<div class="kicker">Assessment</div><div class="card-title">Risk Probability</div>', unsafe_allow_html=True)

                    fig_g, ax_g = plt.subplots(figsize=(4.2, 3.4), subplot_kw={'aspect': 'equal'})
                    fig_g.patch.set_alpha(0)
                    ax_g.set_facecolor('none')

                    gauge_color = '#FF4438' if is_high_risk else '#8B5CF6'
                    track_color = '#ECECF1'

                    theta_bg = np.linspace(np.pi, 0, 100)
                    ax_g.plot(np.cos(theta_bg), np.sin(theta_bg), color=track_color, linewidth=14, solid_capstyle='round')

                    gauge_max = 0.0001 
                    fill_ratio = min(prob / gauge_max, 1.0)
                    # Force a minimum fill of 2% just so a tiny dot is always visible at the start
                    fill_ratio = max(fill_ratio, 0.02)
                    
                    theta_val = np.linspace(np.pi, np.pi - (np.pi * fill_ratio), 100)
                    ax_g.plot(np.cos(theta_val), np.sin(theta_val), color=gauge_color, linewidth=14, solid_capstyle='round')

                    ax_g.text(0, -0.15, f"{prob*100:.4f}%", ha='center', va='center',
                            fontsize=30, fontweight='bold', color='#14141A', family='monospace')
                    ax_g.text(0, -0.5, "MONEY LAUNDERING RISK", ha='center', va='center',
                            fontsize=8.5, color='#6B7280', fontweight='bold')

                    ax_g.set_xlim(-1.3, 1.3)
                    ax_g.set_ylim(-0.7, 1.2)
                    ax_g.axis('off')
                    
                    # 3. FIXED: Removed st.pyplot() to prevent the DOS crash! 
                    buf_g = io.BytesIO()
                    fig_g.savefig(buf_g, format="png", bbox_inches="tight", dpi=300) 
                    st.image(buf_g, use_container_width=True)

                    if is_high_risk:
                        st.markdown('<div class="verdict-banner verdict-flag">🚨 DECISION: Looks illicit, mark for review.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="verdict-banner verdict-pass">✅ DECISION: Licit Transaction.</div>', unsafe_allow_html=True)

                # -------- Waterfall / Feature impact --------
                with res_col2:
                    st.markdown('<div class="kicker">Explainability</div><div class="card-title">True Feature Waterfall</div>', unsafe_allow_html=True)

                    fig, ax = plt.subplots(figsize=(7.5, 4.8))
                    fig.patch.set_alpha(0)
                    ax.set_facecolor('none')

                    features = ['Base Probability', 'GNN Embeddings', 'Local Features', 'One Hop Aggregation']
                    impacts = [result['base_prob'], result['gnn_shift'], result['local_shift'], result['one_hop_shift']]

                    starts = [
                        0,                                                             
                        result['base_prob'],                                           
                        result['base_prob'] + result['gnn_shift'],                     
                        result['base_prob'] + result['gnn_shift'] + result['local_shift'] 
                    ]

                    # Mathematically define exactly where each bar ends
                    endpoints = [s + i for s, i in zip(starts, impacts)]

                    risk_increase_color = '#FF4438'
                    risk_decrease_color = '#8B5CF6'
                    colors = [risk_increase_color if x > 0 else risk_decrease_color for x in impacts]

                    bars = ax.barh(features, impacts, left=starts, color=colors, height=0.45, zorder=3)

                    # --- THE FIX: Dynamic framing based on actual endpoints ---
                    min_x = min(0, min(endpoints))
                    max_x = max(0, max(endpoints))
                    x_span = max_x - min_x

                    # Add equal 30% padding on both sides to guarantee text fits perfectly
                    ax.set_xlim(min_x - (x_span * 0.30), max_x + (x_span * 0.30))

                    for i, (bar, val) in enumerate(zip(bars, impacts)):
                        end_x = endpoints[i]
                        
                        # Place text slightly outside the exact endpoint of the bar
                        if val > 0:
                            align = 'left'
                            offset = x_span * 0.02
                        else:
                            align = 'right'
                            offset = -x_span * 0.02
                            
                        ax.text(end_x + offset, bar.get_y() + bar.get_height()/2, f"{val * 100:+.4f}%", 
                                va='center', ha=align, color='#14141A', fontsize=9.5, 
                                fontweight='bold', family='monospace')

                    ax.axvline(0, color='#9096A2', linewidth=1.5, zorder=2)

                    for spine in ax.spines.values():
                        spine.set_visible(False)

                    ax.xaxis.grid(True, linestyle='--', alpha=0.4, color='#9096A2', zorder=1)
                    ax.set_axisbelow(True)
                    ax.set_xlabel("Cumulative AML Risk Probability", color='#6B7280', labelpad=12, fontweight='600', fontsize=9.5)

                    # --- THE FIX: Increased precision to .4f and limited number of ticks ---
                    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda val, pos: f"{val * 100:.4f}%"))
                    ax.xaxis.set_major_locator(plt.MaxNLocator(5)) # Prevents ticks from squishing together

                    ax.tick_params(axis='x', colors='#6B7280', labelsize=8.5)
                    ax.tick_params(axis='y', colors='#14141A', labelsize=10, pad=15) 

                    ax.invert_yaxis()

                    legend_handles = [
                        mpatches.Patch(color=risk_increase_color, label='Increases risk'),
                        mpatches.Patch(color=risk_decrease_color, label='Decreases risk')
                    ]
                    ax.legend(handles=legend_handles, loc='upper center', bbox_to_anchor=(0.5, 1.15),
                                ncol=2, frameon=False, fontsize=9.5, labelcolor='#14141A')

                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=300) 
                    st.image(buf, use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)

            else :
                st.error(f"API Error {response.status_code}: {response.text}")
st.markdown('</div>', unsafe_allow_html=True)  # close section-white


# ============================================================
# 9. PERFORMANCE AUDIT SECTION
# ============================================================
st.markdown('<div class="section section-white" style="padding-bottom: 1rem;">', unsafe_allow_html=True)

# Create two columns: Button on the left (narrow), Report on the right (wide)
col_aud_btn, col_aud_res = st.columns([1, 4], gap="large")

with col_aud_btn:
    st.markdown('<div class="kicker">Performance Monitoring</div><div class="card-title">Model Performance Audit</div>', unsafe_allow_html=True)
    st.markdown('<div class="section section-white" style="padding-bottom: 1rem;">', unsafe_allow_html=True)
    st.markdown("Generates an Evidently AI powered performance audit based on 5,000 randomly sampled test transactions. The metrics generated can be compared with the reported model metrics. This generates on demand performance check.")
    audit_btn = st.button("Run Performance Audit", key="perf_audit_btn")
    st.markdown('<div class="kicker">Contents</div><div class="card-title">1. Current Check Metrics</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">2. Confusion Matrix</div>', unsafe_allow_html=True)
    st.markdown('<div class="card-title">3. Classification Quality</div>', unsafe_allow_html=True)
    
if audit_btn:
    with st.spinner("Generating Evidently report..."):
        try:
            response = requests.post(f"{API_URL}/performance")
            if response.status_code == 200:
                html_data = response.text
                # Store in session state so it persists
                st.session_state['audit_html'] = html_data
            else:
                st.error("Audit failed. Check API logs.")
        except Exception as e:
            st.error(f"Could not connect to API: {e}")

with col_aud_res:
    if 'audit_html' in st.session_state:
        # Use an st.container to create a sandbox
        with st.container():
            # Inject CSS to override Evidently's internal widths
            st.markdown("""
                <style>
                    /* Force the Evidently report to be responsive within the column */
                    .evidently-report { 
                        max-width: 100% !important; 
                        margin: 0 !important; 
                        padding: 0 !important; 
                    }
                </style>
            """, unsafe_allow_html=True)
            
            st.components.v1.html(st.session_state['audit_html'], height=800, scrolling=True)
    else:
        st.info("Click the button on the left to generate the classification performance report.")

st.markdown('</div>', unsafe_allow_html=True)


# ============================================================
# 10. FOOTER
# ============================================================
st.markdown("""
<div class="footer-section">
    <div class="footer-brand">🛡️ Flagging bitcoin transactions as illicit/licit.</div>
    <div>Graph-native AML evaluation, observability and monitoring for compliance teams.</div>
    <div class="footer-bottom"> 2026 AML Scoring Portal · Internal Compliance Tool Prototype · Elliptic Bitcoin Dataset </div>
    <div class="footer-bottom">
        LinkedIn: <a href="https://www.linkedin.com/in/rageshwer-singh-06a178384" target="_blank" rel="noopener noreferrer" style="color: inherit; text-decoration: underline;">Rageshwer Singh</a> · 
        GitHub: <a href="https://github.com/rageshwer?tab=repositories" target="_blank" rel="noopener noreferrer" style="color: inherit; text-decoration: underline;">Repositories</a>
    </div>
</div>
""", unsafe_allow_html=True)
