"""Home Credit Data Analytics Agent Streamlit app."""

import streamlit as st

from src.data_loader import load_main_dataset, load_column_descriptions
from src.precomputed import load_precomputed
from src.llm_provider import get_provider, LocalModelProvider, LOCAL_MODELS
from src.prompt_templates import build_system_prompt
from src.code_executor import extract_code, run_with_retries

st.set_page_config(
    page_title="Home Credit Analytics Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1f77b4;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    df = load_main_dataset()
    col_desc = load_column_descriptions()
    precomputed = load_precomputed()
    return df, col_desc, precomputed

import base64
def get_base64_of_bin_file(bin_file):
    """Converts a binary file to a base64 string."""
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

df, col_desc_df, precomputed = load_data()

with st.sidebar:
    st.markdown(f'<div class="main-header"><img src="data:image/png;base64,{get_base64_of_bin_file("src/assets/data-analytics.png")}" alt="Data Analytics" width="30"> Analytics Agent</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.subheader("LLM Provider")
    provider_choice = st.radio(
        "Choose your AI backend:",
        [
            "Gemini (Cloud)",
            "Qwen2.5-Coder 7B  (Local, ~4.7 GB)",
            "Qwen2.5-Coder 32B (Local, ~20 GB) — Best",
        ],
        index=0,
        help="Gemini requires GOOGLE_API_KEY. Local models auto-download on first use — no API key needed.",
    )

    if provider_choice.startswith("Gemini"):
        model_name = st.text_input("Model", value="gemini-2.5-flash")
        provider_key = "gemini"
        provider_kwargs = {"model_name": model_name}
        local_model_key = None
    else:
        local_model_key = "32b" if "32B" in provider_choice else "7b"
        cfg = LOCAL_MODELS[local_model_key]
        provider_key = "local"
        provider_kwargs = {"model_key": local_model_key}
        if not LocalModelProvider.is_model_downloaded(local_model_key):
            st.info(f" {cfg['label']} ({cfg['size_label']}) will be downloaded the first time you ask a question.")
        else:
            st.success(f"{cfg['label']} ready <3")

    show_code = st.checkbox("Show generated code", value=False)

    st.markdown("---")

    st.subheader("Dataset Overview")
    ds_info = precomputed.get("dataset_info", {})
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Rows", f"{ds_info.get('rows', len(df)):,}")
    with col2:
        st.metric("Columns", f"{ds_info.get('columns', len(df.columns))}")

    default_rate = ds_info.get("default_rate", df["TARGET"].mean() if "TARGET" in df.columns else 0)
    st.metric("Default Rate", f"{default_rate * 100:.2f}%")

    model_auc = precomputed.get("feature_importance", {}).get("model_auc", "N/A")
    st.metric("Model AUC", model_auc)

    st.markdown("---")

    st.subheader("Sample Questions")
    sample_questions = [
        "What are the top features predicting default?",
        "Show the distribution of loan amounts by contract type",
        "What is the default rate by education level?"
    ]

    for q in sample_questions:
        if st.button(q, key=f"sample_{q}", use_container_width=True):
            st.session_state["pending_question"] = q


st.markdown('<div class="main-header">Home Credit Data Analytics Agent</div>', unsafe_allow_html=True)
st.markdown(
    "Ask questions about the Home Credit Default Risk dataset in plain English. "
    "The agent will analyze the data and respond with charts, tables, and insights."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            if msg.get("text"):
                st.markdown(msg["text"])
            if msg.get("code") and show_code:
                with st.expander("Generated Code"):
                    st.code(msg["code"], language="python")
            for fig in msg.get("figures", []):
                st.plotly_chart(fig, use_container_width=True)
            for tbl in msg.get("dataframes", []):
                st.dataframe(tbl, use_container_width=True)
            if msg.get("error"):
                st.error(msg["error"])
            if msg.get("result") is not None and not msg.get("figures") and not msg.get("dataframes"):
                st.markdown(str(msg["result"]))
        else:
            st.markdown(msg["content"])


def process_question(question: str):
    """Send question to the LLM, execute code, display results."""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    system_prompt = build_system_prompt(df, col_desc_df, precomputed)

    with st.chat_message("assistant"):
        try:
            status_placeholder = st.empty()

            provider = get_provider(provider_key, **provider_kwargs)

            if provider_key == "local":
                if not LocalModelProvider.is_model_downloaded(local_model_key):
                    cfg = LOCAL_MODELS[local_model_key]
                    status_placeholder.warning(
                        f"Downloading {cfg['label']} ({cfg['size_label']}). "
                        "Please wait — this only happens once..."
                    )
                    provider.ensure_ready()
                    status_placeholder.empty()
                elif not provider.is_loaded():
                    status_placeholder.info("🔄 Loading model into memory (first load takes a moment)...")
                    provider.ensure_ready()
                    status_placeholder.empty()
            status_placeholder.info(f"Thinking with {provider.name()} ...")

            llm_response = provider.generate(system_prompt, question)
            code = extract_code(llm_response)

            status_placeholder.info("Executing analysis ...")

            result = run_with_retries(
                code, df, precomputed, provider, system_prompt, question, max_retries=2
            )

            status_placeholder.empty()

            msg = {"role": "assistant", "figures": [], "dataframes": [], "text": "", "code": code}

            if show_code:
                with st.expander("Generated Code", expanded=False):
                    st.code(code, language="python")
                msg["code"] = code

            if result["text"]:
                st.markdown(f"```\n{result['text']}\n```")
                msg["text"] = f"```\n{result['text']}\n```"

            for fig in result["figures"]:
                st.plotly_chart(fig, use_container_width=True)
                msg["figures"].append(fig)

            for tbl in result["dataframes"]:
                st.dataframe(tbl, use_container_width=True)
                msg["dataframes"].append(tbl)

            if result["result"] is not None and not result["figures"] and not result["dataframes"]:
                st.markdown(str(result["result"]))
                msg["result"] = result["result"]

            if result["error"]:
                st.error(f"Execution error:\n{result['error']}")
                msg["error"] = result["error"]

            st.session_state.messages.append(msg)

        except Exception as exc:
            st.error(f"Error: {exc}")
            st.session_state.messages.append({"role": "assistant", "text": "", "error": str(exc)})

if "pending_question" in st.session_state:
    q = st.session_state.pop("pending_question")
    process_question(q)

user_input = st.chat_input("Ask a question about the Home Credit dataset...")
if user_input:
    process_question(user_input)
