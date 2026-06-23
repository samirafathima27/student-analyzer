import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from analyzer import (
    load_and_clean, get_overview, get_gender_analysis,
    get_test_prep_analysis, get_parent_education_analysis,
    get_performance_distribution, train_model, predict_student
)
from report import generate_report

# ── PAGE CONFIG ────────────────────────────────────────────
st.set_page_config(
    page_title="EduLens — Student Intelligence Platform",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── GLOBAL CSS ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: #0A0E1A;
    font-family: 'Inter', sans-serif;
}

/* Hide streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 4rem !important; max-width: 1400px !important; }

/* ── Hero Header ── */
.hero {
    background: linear-gradient(135deg, #0A0E1A 0%, #0D1B3E 50%, #0A0E1A 100%);
    border-bottom: 1px solid rgba(99, 179, 237, 0.15);
    padding: 3rem 2rem 2rem;
    margin: -1rem -2rem 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at 60% 40%, rgba(99, 179, 237, 0.06) 0%, transparent 60%),
                radial-gradient(ellipse at 20% 80%, rgba(159, 122, 234, 0.05) 0%, transparent 50%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #63B3ED;
    margin-bottom: 0.75rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2rem, 4vw, 3.2rem);
    font-weight: 700;
    color: #F0F4FF;
    line-height: 1.1;
    margin: 0 0 0.75rem;
    letter-spacing: -0.02em;
}
.hero-title span {
    background: linear-gradient(135deg, #63B3ED, #9F7AEA);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub {
    font-size: 15px;
    color: #8892A4;
    font-weight: 400;
    max-width: 500px;
    line-height: 1.6;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(99, 179, 237, 0.08);
    border: 1px solid rgba(99, 179, 237, 0.2);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    color: #63B3ED;
    margin-top: 1.25rem;
}

/* ── Upload Zone ── */
.upload-zone {
    background: linear-gradient(135deg, rgba(13, 27, 62, 0.6), rgba(10, 14, 26, 0.8));
    border: 1.5px dashed rgba(99, 179, 237, 0.25);
    border-radius: 16px;
    padding: 2.5rem;
    text-align: center;
    margin: 1.5rem 0;
    transition: border-color 0.2s;
}
.upload-zone:hover { border-color: rgba(99, 179, 237, 0.5); }
.upload-icon { font-size: 2.5rem; margin-bottom: 0.75rem; }
.upload-title { font-size: 18px; font-weight: 600; color: #F0F4FF; margin-bottom: 0.5rem; }
.upload-sub { font-size: 13px; color: #8892A4; }

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
}
.kpi-card {
    background: linear-gradient(135deg, rgba(13, 27, 62, 0.8), rgba(10, 14, 26, 0.9));
    border: 1px solid rgba(99, 179, 237, 0.1);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s, transform 0.2s;
}
.kpi-card:hover {
    border-color: rgba(99, 179, 237, 0.25);
    transform: translateY(-2px);
}
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    border-radius: 14px 14px 0 0;
}
.kpi-card.blue::before { background: linear-gradient(90deg, #63B3ED, #4299E1); }
.kpi-card.green::before { background: linear-gradient(90deg, #68D391, #48BB78); }
.kpi-card.red::before { background: linear-gradient(90deg, #FC8181, #F56565); }
.kpi-card.purple::before { background: linear-gradient(90deg, #B794F4, #9F7AEA); }
.kpi-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #8892A4;
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #F0F4FF;
    line-height: 1;
    margin-bottom: 0.25rem;
}
.kpi-delta {
    font-size: 12px;
    color: #8892A4;
}

/* ── Score Row ── */
.score-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin: 1rem 0;
}
.score-card {
    background: rgba(13, 27, 62, 0.6);
    border: 1px solid rgba(99, 179, 237, 0.08);
    border-radius: 12px;
    padding: 1.25rem;
}
.score-subject {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8892A4;
    margin-bottom: 0.5rem;
}
.score-num {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #F0F4FF;
}
.score-bar-bg {
    height: 4px;
    background: rgba(99, 179, 237, 0.1);
    border-radius: 4px;
    margin-top: 0.75rem;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #63B3ED, #9F7AEA);
}

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 2rem 0 1rem;
}
.section-tag {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #63B3ED;
    background: rgba(99, 179, 237, 0.08);
    border: 1px solid rgba(99, 179, 237, 0.15);
    padding: 3px 10px;
    border-radius: 20px;
}
.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 18px;
    font-weight: 600;
    color: #F0F4FF;
    margin: 0;
}

/* ── Insight Cards ── */
.insight-row {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 1rem;
    margin: 1rem 0;
}
.insight-card {
    background: rgba(13, 27, 62, 0.5);
    border: 1px solid rgba(99, 179, 237, 0.08);
    border-radius: 12px;
    padding: 1.25rem;
}
.insight-label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #8892A4;
    margin-bottom: 0.75rem;
}
.insight-highlight {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #63B3ED;
}

/* ── Predict Form ── */
.predict-form {
    background: linear-gradient(135deg, rgba(13, 27, 62, 0.7), rgba(10, 14, 26, 0.8));
    border: 1px solid rgba(99, 179, 237, 0.12);
    border-radius: 16px;
    padding: 2rem;
    margin: 1rem 0;
}
.predict-result {
    background: linear-gradient(135deg, rgba(72, 187, 120, 0.1), rgba(56, 178, 172, 0.1));
    border: 1px solid rgba(72, 187, 120, 0.3);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    margin-top: 1rem;
}
.predict-result.fail {
    background: linear-gradient(135deg, rgba(245, 101, 101, 0.1), rgba(252, 129, 129, 0.08));
    border-color: rgba(245, 101, 101, 0.3);
}
.predict-label { font-size: 12px; color: #8892A4; margin-bottom: 0.5rem; }
.predict-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #68D391;
}
.predict-value.fail { color: #FC8181; }

/* ── Report ── */
.report-container {
    background: rgba(13, 27, 62, 0.4);
    border: 1px solid rgba(99, 179, 237, 0.1);
    border-radius: 16px;
    padding: 2rem;
    margin: 1rem 0;
    color: #C8D3E8;
    font-size: 15px;
    line-height: 1.8;
}

/* ── Accuracy badge ── */
.accuracy-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(72, 187, 120, 0.1);
    border: 1px solid rgba(72, 187, 120, 0.25);
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 14px;
    font-weight: 600;
    color: #68D391;
    margin-bottom: 1rem;
}

/* ── At risk table ── */
.risk-header {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 0.75rem 1rem;
    background: rgba(245, 101, 101, 0.06);
    border: 1px solid rgba(245, 101, 101, 0.15);
    border-radius: 8px;
    margin-bottom: 0.5rem;
    font-size: 13px;
    color: #FC8181;
    font-weight: 500;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(13, 27, 62, 0.5);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(99, 179, 237, 0.1);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 20px;
    font-size: 13px;
    font-weight: 500;
    color: #8892A4;
    background: transparent;
    border: none;
}
.stTabs [aria-selected="true"] {
    background: rgba(99, 179, 237, 0.12) !important;
    color: #63B3ED !important;
}

/* ── Streamlit overrides ── */
.stFileUploader {
    background: transparent !important;
}
.stFileUploader > div {
    background: rgba(13, 27, 62, 0.5) !important;
    border: 1.5px dashed rgba(99, 179, 237, 0.25) !important;
    border-radius: 12px !important;
}
.stDataFrame { border-radius: 12px; overflow: hidden; }
.stButton > button {
    background: linear-gradient(135deg, #63B3ED, #9F7AEA) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    font-size: 14px !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }
.stSelectbox > div > div {
    background: rgba(13, 27, 62, 0.8) !important;
    border: 1px solid rgba(99, 179, 237, 0.2) !important;
    color: #F0F4FF !important;
    border-radius: 8px !important;
}
.stSlider > div > div > div { background: #63B3ED !important; }
label { color: #8892A4 !important; font-size: 13px !important; }
p { color: #C8D3E8; }
h1, h2, h3 { color: #F0F4FF !important; }
.stSuccess {
    background: rgba(72, 187, 120, 0.08) !important;
    border: 1px solid rgba(72, 187, 120, 0.2) !important;
    border-radius: 8px !important;
    color: #68D391 !important;
}
</style>
""", unsafe_allow_html=True)

# ── HERO ───────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">AI · Machine Learning · Data Analytics</div>
    <h1 class="hero-title">EduLens — <span>Student Intelligence</span><br>Platform</h1>
    <p class="hero-sub">Upload class data and get instant AI-powered insights, risk predictions, and a full performance report.</p>
    <div class="hero-badge">⚡ Powered by Groq LLM + Random Forest ML</div>
</div>
""", unsafe_allow_html=True)

# ── FILE UPLOAD ────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Drop your student CSV here",
    type=["csv"],
    help="Upload a CSV with columns: gender, math score, reading score, writing score, etc."
)

if not uploaded_file:
    st.markdown("""
    <div class="upload-zone">
        <div class="upload-icon">📂</div>
        <div class="upload-title">Upload Student Performance CSV</div>
        <div class="upload-sub">Supports the standard Kaggle Student Performance dataset format</div>
    </div>
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:1rem;margin-top:1.5rem;">
        <div style="background:rgba(13,27,62,0.5);border:1px solid rgba(99,179,237,0.08);border-radius:12px;padding:1.25rem;">
            <div style="font-size:1.5rem;margin-bottom:0.5rem;">📊</div>
            <div style="font-size:14px;font-weight:600;color:#F0F4FF;margin-bottom:0.25rem;">Deep Analysis</div>
            <div style="font-size:12px;color:#8892A4;">Gender, test prep, parental education impact</div>
        </div>
        <div style="background:rgba(13,27,62,0.5);border:1px solid rgba(99,179,237,0.08);border-radius:12px;padding:1.25rem;">
            <div style="font-size:1.5rem;margin-bottom:0.5rem;">🤖</div>
            <div style="font-size:14px;font-weight:600;color:#F0F4FF;margin-bottom:0.25rem;">ML Predictions</div>
            <div style="font-size:12px;color:#8892A4;">Random Forest model predicts pass/fail risk</div>
        </div>
        <div style="background:rgba(13,27,62,0.5);border:1px solid rgba(99,179,237,0.08);border-radius:12px;padding:1.25rem;">
            <div style="font-size:1.5rem;margin-bottom:0.5rem;">📝</div>
            <div style="font-size:14px;font-weight:600;color:#F0F4FF;margin-bottom:0.25rem;">AI Report</div>
            <div style="font-size:12px;color:#8892A4;">Llama 3.3 70B writes your full class report</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── LOAD DATA ──────────────────────────────────────────────
df = load_and_clean(uploaded_file)
overview = get_overview(df)
pass_rate = round(overview['pass_count'] / overview['total_students'] * 100, 1)

st.markdown(f"""
<div style="background:rgba(72,187,120,0.08);border:1px solid rgba(72,187,120,0.2);border-radius:8px;padding:10px 16px;margin:1rem 0;font-size:13px;color:#68D391;font-weight:500;">
    ✅ Successfully loaded <strong>{overview['total_students']} student records</strong> — ready to analyze
</div>
""", unsafe_allow_html=True)

# ── TABS ───────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "  📊  Overview  ",
    "  📈  Charts  ",
    "  🔍  Deep Analysis  ",
    "  🤖  Predict  ",
    "  📝  AI Report  "
])

# ────────────────────────────────────────────────────────────
# TAB 1 — OVERVIEW
# ────────────────────────────────────────────────────────────
with tab1:
    # KPI Row
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card blue">
            <div class="kpi-label">Total Students</div>
            <div class="kpi-value">{overview['total_students']}</div>
            <div class="kpi-delta">records analyzed</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-label">Passed</div>
            <div class="kpi-value">{overview['pass_count']}</div>
            <div class="kpi-delta">{pass_rate}% pass rate</div>
        </div>
        <div class="kpi-card red">
            <div class="kpi-label">At Risk</div>
            <div class="kpi-value">{overview['fail_count']}</div>
            <div class="kpi-delta">need intervention</div>
        </div>
        <div class="kpi-card purple">
            <div class="kpi-label">Class Average</div>
            <div class="kpi-value">{overview['avg_total']}</div>
            <div class="kpi-delta">out of 100</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Subject scores
    st.markdown(f"""
    <div class="score-row">
        <div class="score-card">
            <div class="score-subject">Math Score</div>
            <div class="score-num">{overview['avg_math']}</div>
            <div class="score-bar-bg"><div class="score-bar-fill" style="width:{overview['avg_math']}%"></div></div>
        </div>
        <div class="score-card">
            <div class="score-subject">Reading Score</div>
            <div class="score-num">{overview['avg_reading']}</div>
            <div class="score-bar-bg"><div class="score-bar-fill" style="width:{overview['avg_reading']}%"></div></div>
        </div>
        <div class="score-card">
            <div class="score-subject">Writing Score</div>
            <div class="score-num">{overview['avg_writing']}</div>
            <div class="score-bar-bg"><div class="score-bar-fill" style="width:{overview['avg_writing']}%"></div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick insights
    gender_df = get_gender_analysis(df)
    test_prep_df = get_test_prep_analysis(df)
    try:
        prep_lift = round(test_prep_df.loc["completed", "average"] - test_prep_df.loc["none", "average"], 1)
    except:
        prep_lift = "N/A"

    try:
        top_gender = gender_df["average"].idxmax()
        top_gender_score = round(gender_df.loc[top_gender, "average"], 1)
    except:
        top_gender = "N/A"
        top_gender_score = "N/A"

    st.markdown(f"""
    <div class="insight-row">
        <div class="insight-card">
            <div class="insight-label">🏆 Highest Average Score</div>
            <div class="insight-highlight">{overview['top_student']} / 100</div>
            <div style="font-size:12px;color:#8892A4;margin-top:4px;">Top performer in class</div>
        </div>
        <div class="insight-card">
            <div class="insight-label">⚠️ Lowest Average Score</div>
            <div class="insight-highlight" style="color:#FC8181">{overview['lowest_student']} / 100</div>
            <div style="font-size:12px;color:#8892A4;margin-top:4px;">Needs immediate support</div>
        </div>
        <div class="insight-card">
            <div class="insight-label">📚 Test Prep Score Lift</div>
            <div class="insight-highlight" style="color:#68D391">+{prep_lift} pts</div>
            <div style="font-size:12px;color:#8892A4;margin-top:4px;">Students who completed test prep</div>
        </div>
        <div class="insight-card">
            <div class="insight-label">👥 Top Performing Gender</div>
            <div class="insight-highlight">{top_gender.title()} — {top_gender_score}</div>
            <div style="font-size:12px;color:#8892A4;margin-top:4px;">Average score across all subjects</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Raw data
    st.markdown('<div class="section-header"><span class="section-tag">Data</span><p class="section-title">Full Dataset</p></div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, height=350)

# ────────────────────────────────────────────────────────────
# TAB 2 — CHARTS
# ────────────────────────────────────────────────────────────
with tab2:
    # Chart theme
    chart_theme = dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#8892A4", size=12),
        xaxis=dict(gridcolor="rgba(99,179,237,0.06)", tickcolor="#8892A4", linecolor="rgba(99,179,237,0.1)"),
        yaxis=dict(gridcolor="rgba(99,179,237,0.06)", tickcolor="#8892A4", linecolor="rgba(99,179,237,0.1)"),
        title_font=dict(color="#F0F4FF", size=15, family="Space Grotesk"),
        margin=dict(t=50, b=30, l=10, r=10)
    )

    col1, col2 = st.columns(2)

    with col1:
        fig1 = px.histogram(
            df, x="average", nbins=25,
            title="Score Distribution Across Class",
            color_discrete_sequence=["#63B3ED"]
        )
        fig1.update_layout(**chart_theme)
        fig1.update_traces(marker_line_color="rgba(0,0,0,0)", opacity=0.85)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        result_counts = df["result"].value_counts()
        fig2 = go.Figure(go.Pie(
            values=result_counts.values,
            labels=result_counts.index,
            hole=0.65,
            marker_colors=["#68D391", "#FC8181"],
            textfont=dict(color="#F0F4FF"),
        ))
        fig2.update_layout(
            title="Pass vs Fail Breakdown",
            **{k: v for k, v in chart_theme.items() if k not in ['xaxis', 'yaxis']},
            annotations=[dict(text=f"{pass_rate}%<br>Pass", x=0.5, y=0.5, font_size=16,
                            font_color="#F0F4FF", showarrow=False)]
        )
        st.plotly_chart(fig2, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        gender_df_chart = get_gender_analysis(df).reset_index()
        gender_melted = gender_df_chart.melt(id_vars="gender", value_vars=["math", "reading", "writing"])
        fig3 = px.bar(
            gender_melted, x="variable", y="value", color="gender", barmode="group",
            title="Subject Scores by Gender",
            color_discrete_sequence=["#9F7AEA", "#63B3ED"],
            labels={"variable": "Subject", "value": "Average Score", "gender": "Gender"}
        )
        fig3.update_layout(**chart_theme)
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        prep_chart = get_test_prep_analysis(df).reset_index()
        prep_melted = prep_chart.melt(id_vars="test_prep", value_vars=["math", "reading", "writing"])
        fig4 = px.bar(
            prep_melted, x="variable", y="value", color="test_prep", barmode="group",
            title="Test Prep Impact by Subject",
            color_discrete_sequence=["#FC8181", "#68D391"],
            labels={"variable": "Subject", "value": "Average Score", "test_prep": "Test Prep"}
        )
        fig4.update_layout(**chart_theme)
        st.plotly_chart(fig4, use_container_width=True)

    # Performance distribution full width
    perf_counts = df["performance"].value_counts().reset_index()
    perf_counts.columns = ["Performance", "Count"]
    colors = {"Fail": "#FC8181", "Below Average": "#F6AD55", "Average": "#F6E05E",
              "Good": "#68D391", "Excellent": "#63B3ED"}
    fig5 = px.bar(
        perf_counts, x="Performance", y="Count",
        title="Performance Category Distribution",
        color="Performance",
        color_discrete_map=colors
    )
    fig5.update_layout(**chart_theme)
    st.plotly_chart(fig5, use_container_width=True)

    # Scatter — math vs reading
    fig6 = px.scatter(
        df, x="math", y="reading", color="result",
        title="Math vs Reading Score Correlation",
        color_discrete_map={"Pass": "#68D391", "Fail": "#FC8181"},
        opacity=0.6,
        trendline="ols"
    )
    fig6.update_layout(**chart_theme)
    st.plotly_chart(fig6, use_container_width=True)

# ────────────────────────────────────────────────────────────
# TAB 3 — DEEP ANALYSIS
# ────────────────────────────────────────────────────────────
with tab3:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header"><span class="section-tag">Gender</span><p class="section-title">Performance by Gender</p></div>', unsafe_allow_html=True)
        st.dataframe(get_gender_analysis(df), use_container_width=True)

        st.markdown('<div class="section-header"><span class="section-tag">Test Prep</span><p class="section-title">Preparation Impact</p></div>', unsafe_allow_html=True)
        st.dataframe(get_test_prep_analysis(df), use_container_width=True)

    with col2:
        st.markdown('<div class="section-header"><span class="section-tag">Parents</span><p class="section-title">Parental Education Impact</p></div>', unsafe_allow_html=True)
        parent_df = get_parent_education_analysis(df)
        st.dataframe(parent_df, use_container_width=True)

        st.markdown('<div class="section-header"><span class="section-tag">Distribution</span><p class="section-title">Performance Spread</p></div>', unsafe_allow_html=True)
        st.dataframe(get_performance_distribution(df), use_container_width=True)

    # Top performers
    st.markdown('<div class="section-header"><span class="section-tag">Top 10</span><p class="section-title">🏆 Highest Performers</p></div>', unsafe_allow_html=True)
    top10 = df.nlargest(10, "average")[["gender", "race", "test_prep", "math", "reading", "writing", "average", "result"]]
    st.dataframe(top10, use_container_width=True)

    # At risk
    st.markdown("""
    <div class="risk-header">
        ⚠️ Students At Risk — Lowest 10 performers needing intervention
    </div>
    """, unsafe_allow_html=True)
    bottom10 = df.nsmallest(10, "average")[["gender", "race", "test_prep", "math", "reading", "writing", "average", "result"]]
    st.dataframe(bottom10, use_container_width=True)

# ────────────────────────────────────────────────────────────
# TAB 4 — PREDICT
# ────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header"><span class="section-tag">ML Model</span><p class="section-title">Predict Student Pass / Fail</p></div>', unsafe_allow_html=True)

    with st.spinner("Training Random Forest model on your data..."):
        model, accuracy, le = train_model(df)

    st.markdown(f"""
    <div class="accuracy-badge">
        ✅ Model Ready — Accuracy: {accuracy}% on test data
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="predict-form">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["female", "male"])
        lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
        test_prep = st.selectbox("Test Preparation", ["none", "completed"])

    with col2:
        math = st.slider("Math Score", 0, 100, 65)
        reading = st.slider("Reading Score", 0, 100, 65)
        writing = st.slider("Writing Score", 0, 100, 65)

    with col3:
        avg = round((math + reading + writing) / 3, 1)
        total = math + reading + writing
        st.markdown(f"""
        <div style="padding:1rem;background:rgba(99,179,237,0.06);border-radius:10px;border:1px solid rgba(99,179,237,0.12);margin-top:1.5rem;">
            <div style="font-size:11px;color:#8892A4;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.5rem">Live Preview</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:700;color:#F0F4FF">{avg}</div>
            <div style="font-size:12px;color:#8892A4">Average score</div>
            <div style="margin-top:0.75rem;font-size:13px;color:#C8D3E8">Total: <strong>{total}</strong> / 300</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("🔍 Run Prediction", type="primary", use_container_width=True):
        prediction = predict_student(model, le, gender, lunch, test_prep, math, reading, writing)
        is_pass = "✅" in prediction
        result_class = "" if is_pass else "fail"
        result_text = "Pass" if is_pass else "Fail"
        result_color = "#68D391" if is_pass else "#FC8181"
        msg = "This student is likely to pass based on the ML model." if is_pass else "This student is at risk of failing. Consider additional support."

        st.markdown(f"""
        <div class="predict-result {result_class}">
            <div class="predict-label">ML Prediction Result</div>
            <div class="predict-value {result_class}">{result_text}</div>
            <div style="font-size:13px;color:#8892A4;margin-top:0.5rem">{msg}</div>
        </div>
        """, unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────
# TAB 5 — AI REPORT
# ────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-header"><span class="section-tag">Groq LLM</span><p class="section-title">AI Generated Class Report</p></div>', unsafe_allow_html=True)
    st.markdown('<p style="color:#8892A4;font-size:14px">Llama 3.3 70B analyzes your class data and writes a professional educational report with findings and recommendations.</p>', unsafe_allow_html=True)

    if st.button("✨ Generate AI Report", type="primary", use_container_width=True):
        with st.spinner("Llama 3.3 70B is analyzing your class... (10-20 seconds)"):
            model, accuracy, le = train_model(df)
            report = generate_report(
                overview,
                get_gender_analysis(df),
                get_test_prep_analysis(df),
                accuracy
            )

        st.markdown(f'<div class="report-container">{report}</div>', unsafe_allow_html=True)
        st.download_button(
            label="📥 Download Report as .txt",
            data=report,
            file_name="student_performance_report.txt",
            mime="text/plain",
            use_container_width=True
        )

        