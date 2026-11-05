from uuid import UUID

import pandas as pd
import streamlit as st

from rie.config import get_settings
from rie.graph.workflow import ResearchWorkflow
from rie.retrieval.semantic_scholar import SemanticScholarClient
from rie.state import ResearchState


st.set_page_config(
    page_title="RIE | Research Intelligence Engine",
    page_icon="R",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root { color-scheme: light; --rie-ink: #15243a; --rie-muted: #65758b; --rie-accent: #087e8b; --rie-line: #e2e8f0; --rie-wash: #f5f8fb; }
      .stApp, [data-testid="stAppViewContainer"] { background: #f5f7fa !important; color: var(--rie-ink) !important; }
      [data-testid="stHeader"] { background: rgba(245,247,250,.9) !important; }
      [data-testid="stMain"] { background: transparent !important; }
      .stApp [data-testid="stMarkdownContainer"], .stApp [data-testid="stMarkdownContainer"] * { color: var(--rie-ink); }
      .stApp [data-testid="stMarkdownContainer"] a { color: var(--rie-accent); }
      .stApp code { color:#284b63; background:#edf3f7; }
      .block-container { max-width: 1320px; padding: 2.4rem 3rem 4rem; }
      [data-testid="stSidebar"] { background: #fff !important; border-right: 1px solid var(--rie-line); }
      [data-testid="stSidebar"] * { color: #26364b; }
      h1, h2, h3, p, label, [data-testid="stMetricLabel"], [data-testid="stMetricValue"] { color: var(--rie-ink) !important; }
      h1, h2, h3 { letter-spacing: -.035em; }
      .rie-brand { display: inline-flex; align-items: center; gap: 9px; color: var(--rie-accent); font-size: .76rem; font-weight: 750; letter-spacing: .13em; text-transform: uppercase; }
      .rie-brand-mark { display:inline-grid; place-items:center; width:27px; height:27px; border-radius:9px; background:#dff3f0; color:#087e8b; font-size:.85rem; letter-spacing:0; }
      .rie-title { color: var(--rie-ink); font-size: clamp(2.2rem, 4vw, 3.2rem); font-weight: 760; line-height: 1.08; letter-spacing: -.045em; margin: .8rem 0 .65rem; }
      .rie-lede { color: var(--rie-muted); font-size: 1.06rem; max-width: 760px; line-height: 1.7; }
      .rie-step { background: #fff; border: 1px solid var(--rie-line); border-radius: 15px; padding: 17px 18px; min-height: 91px; box-shadow: 0 3px 12px rgba(24,39,60,.035); }
      .rie-step b { display: block; color: var(--rie-ink); margin-bottom: 6px; font-size: .96rem; }
      .rie-step span { color: var(--rie-muted); font-size: .88rem; line-height: 1.5; }
      [data-testid="stForm"] { background: #fff; border: 1px solid var(--rie-line); border-radius: 18px; padding: 22px 24px 20px; box-shadow: 0 10px 30px rgba(24,39,60,.055); }
      [data-testid="stTextArea"] textarea { background: #fbfcfe !important; color: var(--rie-ink) !important; border: 1px solid #dbe3ec !important; border-radius: 12px !important; }
      [data-testid="stTextArea"] textarea:focus { border-color: #49a6a4 !important; box-shadow: 0 0 0 3px rgba(8,126,139,.12) !important; }
      div.stButton > button[kind="primaryFormSubmit"] { min-height: 3.1rem; border-radius: 11px; background: #087e8b; border: 0; font-weight: 650; }
      div.stButton > button[kind="primaryFormSubmit"] p { color: white !important; }
      div.stButton > button[kind="primaryFormSubmit"]:hover { background: #066d78; border: 0; }
      [data-testid="stMetric"] { background:#fff; border:1px solid var(--rie-line); border-radius:14px; padding:16px 18px; box-shadow:0 3px 12px rgba(24,39,60,.035); }
      [data-testid="stMetricLabel"] { color:var(--rie-muted) !important; font-size:.85rem; }
      [data-testid="stMetricValue"] { font-weight:720; }
      [data-testid="stTabs"] [role="tab"] { color:#607187; font-weight:600; }
      [data-testid="stTabs"] [role="tab"][aria-selected="true"] { color:var(--rie-accent); }
      [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background:var(--rie-accent); }
      [data-testid="stExpander"] { background:#fff; border:1px solid var(--rie-line); border-radius:12px; overflow:hidden; }
      [data-testid="stExpander"] summary { color:var(--rie-ink); }
      [data-testid="stDataFrame"] { border:1px solid var(--rie-line); border-radius:12px; overflow:hidden; }
      .rie-empty { background:#fff; border:1px solid var(--rie-line); border-radius:16px; padding:24px 26px; margin:1rem 0; }
      .rie-empty h3 { margin:.1rem 0 .45rem; font-size:1.15rem; }
      .rie-empty p { color:var(--rie-muted) !important; margin:0; line-height:1.65; }
      .rie-note { color: var(--rie-muted); font-size: .85rem; border-top:1px solid var(--rie-line); padding-top:1rem; }
      @media (max-width: 760px) { .block-container { padding:1.5rem 1rem 3rem; } [data-testid="stForm"] { padding:17px; } }
    </style>
    """,
    unsafe_allow_html=True,
)

settings = get_settings()
try:
    project_id = settings.watsonx_project_id.strip()
    parsed_project_id = UUID(project_id)
    project_uuid_valid = parsed_project_id.version == 4 and str(parsed_project_id) == project_id.lower()
except ValueError:
    project_uuid_valid = False
granite_ready = bool(settings.watsonx_api_key.strip()) and project_uuid_valid

with st.sidebar:
    st.markdown("## Research setup")
    results_per_source = st.slider("Papers per source", min_value=5, max_value=25, value=10)
    top_k = st.slider("Evidence passages", min_value=3, max_value=15, value=5)
    st.caption("Results are retrieved from Semantic Scholar, with Crossref as a fallback.")
    st.divider()
    st.markdown("### Required AI service")
    if not settings.watsonx_api_key:
        st.error("IBM Granite is not configured")
        st.caption("Add your IBM Cloud API key and project UUID to `.env`, then restart the app.")
    elif not settings.watsonx_project_id.strip():
        st.error("WatsonX project ID is missing")
        st.caption("Set `WATSONX_PROJECT_ID` in `.env` and restart the app.")
    else:
        if project_uuid_valid:
            st.success("IBM Granite settings are present")
            st.caption(f"Model: `{settings.watsonx_model_id}`")
            st.caption("Cloud access is verified when a research run starts.")
        else:
            st.error("Project ID must be an IBM Cloud UUIDv4")
    with st.expander("Configure IBM Cloud Lite"):
        st.markdown(
            "1. Create or open a Watsonx.ai project in IBM Cloud.\n"
            "2. Associate an active Watson Machine Learning service with that project.\n"
            "3. Put the API key, project UUID, and regional service URL in `.env`.\n"
            "4. Restart Streamlit so it reloads the settings.\n\n"
            "Keep credentials in `.env`; never commit them."
        )

st.markdown('<div class="rie-brand"><span class="rie-brand-mark">R</span> IBM SkillsBuild · Research workspace</div>', unsafe_allow_html=True)
st.markdown('<div class="rie-title">Research Intelligence Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="rie-lede">Turn a research question into a concise, source-linked literature review. '
    'IBM Granite plans the search, extracts evidence, and drafts a report from retrieved academic papers.</div>',
    unsafe_allow_html=True,
)

st.write("")
step_columns = st.columns(4)
for column, title, description in zip(
    step_columns,
    ("01 · Plan", "02 · Retrieve", "03 · Ground", "04 · Report"),
    ("Granite frames focused searches", "Search OpenAlex and Semantic Scholar", "Rank abstracts and extract evidence", "Review findings with linked sources"),
):
    column.markdown(
        f'<div class="rie-step"><b>{title}</b><span>{description}</span></div>',
        unsafe_allow_html=True,
    )

st.write("")
with st.form("research_question_form"):
    question = st.text_area(
        "Your research question",
        placeholder="Example: How does retrieval-augmented generation affect factual accuracy in large language models?",
        height=115,
        help="A focused question usually produces a more useful search and evidence summary.",
    )
    submitted = st.form_submit_button(
        "Start literature review",
        type="primary",
        disabled=not granite_ready,
        use_container_width=True,
    )

if not granite_ready:
    st.info("Configure a valid IBM Cloud API key and project UUID in `.env` to enable research. This MVP requires IBM Granite.")

if submitted:
    if not question.strip():
        st.warning("Enter a research question to start the review.")
        st.stop()
    state = ResearchState(research_question=question.strip())
    progress = st.progress(0, text="Preparing the research plan…")
    try:
        workflow = ResearchWorkflow(
            semantic_scholar=SemanticScholarClient(settings.semantic_scholar_api_key),
            top_k=top_k,
        )
        progress.progress(15, text="Planning focused searches with IBM Granite…")
        progress.progress(30, text="Searching academic literature…")
        state = workflow.run(state, results_per_source=results_per_source)
        progress.progress(100, text="Literature review complete")
        st.session_state["rie_last_state"] = state.model_dump()
    except Exception as exc:
        progress.empty()
        st.error("The literature review could not be completed.")
        message = str(exc)
        if "Provider details:" in message:
            explanation, provider_details = message.split("Provider details:", 1)
            st.markdown(explanation.strip())
            with st.expander("Technical details from WatsonX"):
                st.code(provider_details.strip(), language="json")
        else:
            st.markdown(message)
        if "not associated with a Watson Machine Learning" in message:
            st.info("In IBM Cloud, associate this project with an active Watson Machine Learning service in the same account and region, then restart the research run.")
        elif "UUIDv4" in message or "project UUIDv4" in message:
            st.info("Copy the project UUID from the IBM Cloud project Manage page into `WATSONX_PROJECT_ID`, then restart Streamlit.")
        elif "WatsonX" in message:
            st.info("Review the API key, project UUID, Watsonx.ai regional URL, and model ID in `.env`.")
        else:
            st.info("Check source connectivity and the terminal log, then retry. A source may be temporarily unavailable.")
        st.stop()

if "rie_last_state" not in st.session_state:
    st.info("Enter a research question to begin. The report will include evidence extracted from paper abstracts and links to the retrieved sources.")
else:
    state = ResearchState.model_validate(st.session_state["rie_last_state"])
    st.divider()
    st.markdown("## Review workspace")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Papers retrieved", len(state.papers))
    metric_columns[1].metric("Evidence passages", len(state.chunks))
    metric_columns[2].metric("Extracted claims", len(state.evidence))

    if not state.papers:
        rate_limited = any("429" in error or "Too Many Requests" in error for error in state.errors)
        empty_title = "The paper sources are temporarily busy" if rate_limited else "No papers came back for this search"
        empty_copy = (
            "OpenAlex or Semantic Scholar throttled this run. Give the services a little time, then retry; "
            "you can also broaden the question to reduce the number of searches."
            if rate_limited
            else "Try a broader question or check the source notices below. A report will appear here once papers are retrieved."
        )
        st.markdown(
            f'<div class="rie-empty"><h3>{empty_title}</h3><p>{empty_copy}</p></div>',
            unsafe_allow_html=True,
        )
        if state.subqueries:
            with st.expander("Search strategy"):
                for index, subquery in enumerate(state.subqueries, 1):
                    st.markdown(f"{index}. {subquery}")
        if state.errors:
            with st.expander(f"Source details · {len(state.errors)} notices"):
                st.caption("Only the first few source errors are shown to keep this view readable.")
                for error in list(dict.fromkeys(state.errors))[:4]:
                    st.warning(error)
    else:
        tabs = st.tabs(["Summary", "Evidence", "Papers", "Report"])
        with tabs[0]:
            st.markdown(f"**Research question**\n\n{state.research_question}")
            if state.subqueries:
                with st.expander("Search strategy"):
                    for index, subquery in enumerate(state.subqueries, 1):
                        st.markdown(f"{index}. {subquery}")
            st.markdown(state.report or "No report was produced for this run.")

        with tabs[1]:
            if state.evidence:
                paper_by_id = {paper.paper_id: paper for paper in state.papers}
                for index, item in enumerate(state.evidence, 1):
                    paper = paper_by_id.get(item.chunk.paper_id)
                    with st.container(border=True):
                        st.markdown(f"**Finding {index}: {item.claim}**")
                        st.write(item.summary)
                        st.caption(f"{item.stance.title()} · extraction confidence {item.confidence:.0%}")
                        if paper:
                            source_label = f"{paper.title}" + (f" ({paper.year})" if paper.year else "")
                            if paper.url:
                                st.markdown(f"Source: [{source_label}]({paper.url})")
                            else:
                                st.markdown(f"Source: {source_label}")
            else:
                st.info("No evidence claims were extracted. Review the Papers tab or try another question.")

        with tabs[2]:
            st.dataframe(
                pd.DataFrame([
                    {
                        "Title": paper.title,
                        "Authors": ", ".join(paper.authors[:3]) or "Unknown",
                        "Year": paper.year or "—",
                        "Source": paper.source.replace("_", " ").title(),
                        "Citations": paper.cited_by_count,
                        "URL": paper.url or (f"https://doi.org/{paper.doi.removeprefix('https://doi.org/')}" if paper.doi else ""),
                    }
                    for paper in state.papers
                ]),
                use_container_width=True,
                hide_index=True,
                column_config={"URL": st.column_config.LinkColumn("Paper link", display_text="Open paper")},
            )

        with tabs[3]:
            st.markdown(state.report or "No report is available.")
            st.download_button(
                "Download report as Markdown",
                data=state.report or "",
                file_name="research_intelligence_report.md",
                mime="text/markdown",
                disabled=not state.report,
            )

        if state.errors:
            with st.expander(f"Source notices · {len(state.errors)}"):
                for error in list(dict.fromkeys(state.errors))[:6]:
                    st.warning(error)

st.markdown(
    '<div class="rie-note">Research assistance, not a substitute for reading and verifying cited papers. '
    'The current MVP indexes paper abstracts; candidate findings may be incomplete.</div>',
    unsafe_allow_html=True,
)
