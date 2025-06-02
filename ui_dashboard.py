import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(
    page_title="RAG-LLM Evaluation",
    layout="wide"
)
st.title("🔍 RAG-LLM Multi-Hop QA Dashboard")

# Sidebar information
with st.sidebar:
    st.header("About")
    st.markdown("""
    This dashboard shows metrics for a Retrieval-Augmented Generation (RAG) pipeline:
    - **Multi-Hop QA**: Supports questions needing more than one context
    - **Retrieval Metrics**: Recall@5, MRR@5, MAP@5
    - **Generation Metrics**: EM, F1, End-to-End EM/F1
    - **RAGAs**: Context recall via Non-LLM or LLM
    - **Latencies**: Retrieval and Generation
    """)
    st.markdown("**Usage**:")
    st.code("""
    python eval.py
    streamlit run ui_dashboard.py
    """, language="bash")
    st.markdown("**Repository:** [GitHub Link](https://github.com/GiannisKarampas/rag_llm_evaluation.git)")

# Main content
try:
    df = pd.read_json("eval_results.json")

    # 1. At-a-glance KPIs
    st.subheader("Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Mean Recall@5", f"{df['ctx_recall@5'].mean():.2f}")
        st.metric("Mean MRR@5", f"{df['mrr@5'].mean():.2f}")
    with col2:
        st.metric("Mean MAP@5", f"{df['map@5'].mean():.2f}")
        st.metric("Mean Gen EM", f"{df['gen_em'].mean():.2f}")
    with col3:
        st.metric("Mean Gen F1", f"{df['gen_f1'].mean():.2f}")
        st.metric("Mean E2E F1", f"{df['e2e_f1'].mean():.2f}")
    with col4:
        st.metric("Retrieval_latency", f"{df['retrieval_latency'].mean():.2f}")
        st.metric("Generation_latency", f"{df['generation_latency'].mean():.2f}")

    st.markdown("---")

    # 2. Filters
    with st.sidebar:
        st.header("Filters")
        min_recall = st.slider("Min Recall@5", 0.0, 1.0, 0.0, 0.01)
        min_mrr = st.slider("Min MRR@5", 0.0, 1.0, 0.0, 0.01)
        min_f1 = st.slider("Min Gen F1", 0.0, 1.0, 0.0, 0.01)
        show_errors = st.checkbox("Show only EM = 0", value=False)

    filtered_df = df[
        (df["ctx_recall@5"] >= min_recall) &
        (df["mrr@5"] >= min_mrr) &
        (df["gen_f1"] >= min_f1)
    ]
    if show_errors:
        filtered_df = filtered_df[filtered_df["gen_em"] == 0]

    st.subheader("Filtered Results")
    st.dataframe(filtered_df)

    # 3. Distributions
    st.markdown("---")
    st.subheader("Metric Distributions")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Recall@5")
        recall_counts = df["ctx_recall@5"].value_counts().sort_index().reset_index()
        recall_counts.columns = ["recall", "count"]
        st.bar_chart(recall_counts.set_index("recall"))
    with col2:
        st.write("MRR@5")
        mrr_counts = df["mrr@5"].value_counts().sort_index().reset_index()
        mrr_counts.columns = ["mrr", "count"]
        st.bar_chart(mrr_counts.set_index("mrr"))

    col3, col4 = st.columns(2)
    with col3:
        st.write("Gen F1 Distribution")
        chart = alt.Chart(df).mark_bar().encode(
            alt.X("gen_f1:Q", bin=alt.Bin(maxbins=20)),
            y="count()"
        )
        st.altair_chart(chart, use_container_width=True)
    with col4:
        st.write("Retrieval Latency (ms)")
        box = alt.Chart(df).mark_boxplot().encode(
            y=alt.Y("retrieval_latency:Q", title="Latency (ms)")
        )
        st.altair_chart(box, use_container_width=True)

    # 4. Question-level Drill-down
    st.markdown("---")
    st.subheader("Inspect a Single Question")
    choice = st.selectbox("Select a question", df["question"].tolist())
    row = df[df["question"] == choice].iloc[0]

    st.markdown(f"**Question:** {row['question']}")
    st.markdown("**Retrieved IDs:** " + ", ".join(row["retrieved_ids"]))
    st.markdown(f"**Predicted Answer:** {row['pred_answer']}")
    st.markdown(f"**Ground Truth:** {', '.join(row['ground_truth_answers'])}")
    st.markdown(f"**Metrics:** EM={row['gen_em']}, F1={row['gen_f1']:.2f}, Recall={row['ctx_recall@5']:.2f}, MRR={row['mrr@5']:.2f}, MAP={row['map@5']:.2f}")

except FileNotFoundError:
    st.error("`eval_results.json` not found. Run `python eval.py` first.")
