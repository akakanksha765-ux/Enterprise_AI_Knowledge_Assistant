"""
Enterprise AI Knowledge Assistant
---------------------------------

Run:
    uv run streamlit run streamlit_app.py
"""

import requests
import streamlit as st

API_URL = "http://localhost:8000/chat"

# -----------------------------------------------------------------------------
# Page
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Enterprise AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

with st.sidebar:

    st.title("System")

    st.markdown("### LLM")
    st.code("llama3.2:3b")

    st.markdown("### Embedding Model")
    st.code("BAAI/bge-small-en-v1.5")

    st.markdown("### Vector Store")
    st.code("FAISS")

    st.markdown("### Backend")
    st.code("FastAPI")

    st.markdown("### Tracing")
    st.code("Opik")

    st.markdown("### Monitoring")
    st.code("Prometheus")

    st.markdown("### Visualization")
    st.code("Grafana")

    st.subheader("Observability")

    st.link_button(
        "📊 Open Grafana",
        "http://localhost:3000",
        use_container_width=True,
    )

    st.link_button(
        "📈 Open Prometheus",
        "http://localhost:9090",
        use_container_width=True,
    )

    st.link_button(
        "🔍 Open Opik",
        "https://www.comet.com/opik",
        use_container_width=True,
    )
    
# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------

st.title("🤖 Enterprise AI Knowledge Assistant")

st.write(
    "Ask questions about enterprise documents using a Retrieval-Augmented "
    "Generation (RAG) pipeline."
)

st.divider()

# -----------------------------------------------------------------------------
# Question
# -----------------------------------------------------------------------------

# -------------------------------------------------------------------
# Session State Initialization
# -------------------------------------------------------------------

if "question" not in st.session_state:
    st.session_state.question = ""

if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = ""


# -------------------------------------------------------------------
# Sample Questions
# -------------------------------------------------------------------

sample_questions = [
    "Summarize the company overview.",
    "What are the key financial highlights?",
    "Explain the business glossary.",
    "What products does Northwind sell?",
]


# -------------------------------------------------------------------
# Callbacks
# -------------------------------------------------------------------

def select_sample_question():
    """Copy selected sample question into the question text area."""

    selected = st.session_state.selected_sample

    if selected:
        st.session_state.question = selected


def clear_question():
    """Clear question and selected sample question."""

    st.session_state.question = ""
    st.session_state.selected_sample = ""


# -------------------------------------------------------------------
# Question
# -------------------------------------------------------------------

question = st.text_area(
    "Question",
    placeholder="Example: What are the company's revenue highlights?",
    height=150,
    key="question",
)


# -------------------------------------------------------------------
# Sample Questions
# -------------------------------------------------------------------

st.caption("Sample Questions")

st.selectbox(
    "Choose a sample question",
    [""] + sample_questions,
    key="selected_sample",
    on_change=select_sample_question,
)


# -------------------------------------------------------------------
# Buttons
# -------------------------------------------------------------------

c1, c2, c3 = st.columns([1, 1, 8])

with c1:
    ask = st.button(
        "🚀 Ask",
        use_container_width=True,
    )

with c2:
    st.button(
        "🗑 Clear",
        use_container_width=True,
        on_click=clear_question,
    )



# -----------------------------------------------------------------------------
# Ask
# -----------------------------------------------------------------------------

if ask:

    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    payload = {
        "question": question
    }

    try:

        with st.spinner(
            "Searching enterprise documents...\nGenerating response..."
        ):

            response = requests.post(
                API_URL,
                json=payload,
                timeout=180,
            )

    except requests.exceptions.ConnectionError:

        st.error(
            """
Unable to connect to the FastAPI backend.

Start it first:

docker compose up
"""
        )
        st.stop()

    except requests.exceptions.Timeout:

        st.error("Request timed out.")
        st.stop()

    except Exception as ex:

        st.exception(ex)
        st.stop()

    if response.status_code != 200:

        st.error(response.text)
        st.stop()

    data = response.json()
    meta = data["metadata"]

    # -------------------------------------------------------------------------
    # Answer
    # -------------------------------------------------------------------------

    st.divider()

    st.subheader("Answer")

    with st.container(border=True):
        st.markdown(data["answer"])

    # -------------------------------------------------------------------------
    # Sources
    # -------------------------------------------------------------------------

    if data["sources"]:

        st.subheader("Sources")

        with st.container(border=True):

            for source in data["sources"]:
                st.markdown(f"📄 **{source}**")
    # -------------------------------------------------------------------------
    # Metadata
    # -------------------------------------------------------------------------

    with st.expander("Metadata", expanded=False):

        st.write(
            f"**Response Time:** {meta['response_time_ms']} ms"
        )

        st.write(
            f"**Retrieved Chunks:** {meta['retrieved_chunks']}"
        )

        st.write(
            f"**Source Documents:** {meta['source_documents']}"
        )

        st.write(
            f"**LLM Model:** {meta['llm_model']}"
        )

        st.write(
            f"**Embedding Model:** {meta['embedding_model']}"
        )

        st.write(
            f"**Vector Store:** {meta['vector_store']}"
        )

        st.divider()

        st.subheader("Raw JSON Response")

        st.json(data)
# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------

st.divider()

st.caption(
    "Enterprise AI Knowledge Assistant v1.0\n\n"
    "FastAPI • LangChain • FAISS • Ollama • "
    "Opik • Prometheus • Grafana"
)