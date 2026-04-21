import streamlit as st
import json
import pandas as pd
import os
import subprocess
import sys

st.set_page_config(
    page_title="AI Hiring Intelligence System",
    page_icon="brain",
    layout="wide"
)

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/ats_mock_data.csv")
RESULTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/last_run_results.json")


def load_results():
    if not os.path.exists(RESULTS_PATH):
        return None
    with open(RESULTS_PATH) as f:
        return json.load(f)


def load_ats():
    df = pd.read_csv(DATA_PATH)
    df["apply_date"] = pd.to_datetime(df["apply_date"])
    return df


st.title("AI-Powered Hiring Intelligence System")
st.caption("Multi-agent pipeline for engineering talent acquisition insights")

col1, col2, col3 = st.columns([1, 1, 4])
with col1:
    run_clicked = st.button("Run Pipeline", type="primary", use_container_width=True)
with col2:
    refresh_clicked = st.button("Refresh", use_container_width=True)

if run_clicked:
    with st.spinner("Running all 5 insight agents..."):
        result = subprocess.run(
            [sys.executable, "../orchestrator.py"],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.abspath(__file__))
        )
        if result.returncode == 0:
            st.success("Pipeline complete!")
        else:
            st.error(f"Pipeline error: {result.stderr[:500]}")

df = load_ats()
results = load_results()

st.divider()
st.subheader("Pipeline Overview")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Candidates", len(df))
k2.metric("Hired", len(df[df["current_stage"] == "Hired"]))
k3.metric("Active Offers", len(df[df["current_stage"] == "Offer"]))
k4.metric("Avg Days in Pipeline", f"{df['days_in_pipeline'].mean():.0f}")
k5.metric("SLA Breach Rate", f"{df['sla_breached'].eq('Yes').mean()*100:.0f}%")

st.divider()
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("Hiring Funnel")
    stage_order = ["Applied", "Phone Screen", "Technical Screen", "Onsite", "Offer", "Hired", "Rejected"]
    stage_counts = df["current_stage"].value_counts().reindex(stage_order, fill_value=0)
    st.bar_chart(stage_counts)

with col_right:
    st.subheader("Source Distribution")
    source_counts = df["source"].value_counts()
    st.bar_chart(source_counts)

st.divider()
st.subheader("Agent Insights")

if results:
    insights = results["pipeline_run"]["insights"]
    evaluations = results["evaluations"]
    eval_map = {e["evaluated_agent"]: e for e in evaluations}

    for insight in insights:
        agent = insight["agent_name"]
        ev = eval_map.get(agent, {})
        score = ev.get("overall_score", 0)
        passed = ev.get("passed", False)
        status_icon = "PASS" if passed else "FAIL"

        with st.expander(f"{status_icon} {agent} - Score: {score:.2f}", expanded=passed):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown("**Recommendation**")
                st.info(insight["recommendation"])
                st.markdown("**Evidence**")
                st.write(insight["evidence"])
                st.markdown("**Alternative**")
                st.write(insight["alternative"])
            with c2:
                st.metric("Confidence", f"{insight['confidence_score']:.0%}")
                st.metric("Eval Score", f"{score:.2f}")
                cost = insight["cost_of_insight"]
                st.metric("Cost", f"${cost['estimated_usd']:.6f}")
                st.metric("Model", cost["model"])

            if ev.get("flags"):
                st.warning("Flags: " + " | ".join(ev["flags"]))

    st.divider()
    st.subheader("Cost and Performance Summary")
    pr = results["pipeline_run"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Cost", f"${pr['total_estimated_usd']:.6f}")
    m2.metric("Total Tokens", f"{pr['total_input_tokens'] + pr['total_output_tokens']:,}")
    m3.metric("Total Latency", f"{pr['total_latency_seconds']:.1f}s")
    m4.metric("Success Rate", f"{pr['successful_agents']}/{pr['total_agents']}")

else:
    st.info("No pipeline results yet. Click Run Pipeline to generate insights.")

st.divider()
st.caption("Lamonte Smith - Interview Kickstart Applied Agentic AI - Capstone 2 - April 2026")
