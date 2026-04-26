"""Streamlit dashboard for the RAG search assistant.

Hybrid retrieval (BM25 + sentence-transformer embeddings) followed by an
LLM-grounded answer with source citations. Sliders expose the BM25/dense
weight and top-K so the user can see the retrieval mix shift in real time.
"""

import os

import streamlit as st

from rag import RAGEngine

st.set_page_config(page_title="RAG Search Assistant", layout="wide")


@st.cache_resource(show_spinner="Loading retrieval indices and embedding model...")
def load_engine(corpus_path: str = "corpus") -> RAGEngine:
    return RAGEngine(corpus_path=corpus_path)


st.title("RAG Search Assistant")
st.caption(
    "Hybrid **BM25** (sparse, term-overlap) + **dense embeddings** "
    "(semantic) retrieval, with an **LLM-grounded answer** that cites its sources."
)

engine = load_engine("corpus")
stats = engine.stats()

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Retrieval settings")
    alpha = st.slider(
        "BM25 weight (α)", 0.0, 1.0, 0.5, 0.05,
        help="α = 1 → BM25 only · α = 0 → dense only · 0.5 → equal weight",
    )
    top_k = st.slider("Top-K retrieved docs", 1, 10, 5)

    st.divider()
    st.subheader("LLM provider")
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    if has_anthropic:
        st.success("Anthropic key detected (Claude Haiku 4.5)")
    elif has_openai:
        st.success("OpenAI key detected (GPT-4o-mini)")
    else:
        st.warning("No API key set — running in extractive-fallback mode")
        st.code(
            "ANTHROPIC_API_KEY=sk-ant-...\nOPENAI_API_KEY=sk-...",
            language="bash",
        )
        st.caption(
            "Set either env var before launching Streamlit to enable "
            "LLM-grounded answers."
        )

    st.divider()
    st.subheader("Corpus stats")
    st.markdown(
        f"""
- **{stats['documents']}** documents
- **{stats['vocabulary']}** unique terms
- **{stats['total_tokens']}** total tokens
- avg **{stats['avg_doc_length']}** tokens/doc
- embedding dim: **{stats['embedding_dim']}**
- model: `{stats['embedding_model']}`
"""
    )

# ---------------------------------------------------------------- main panel
EXAMPLES = [
    "How do I request paid time off?",
    "What's the process for code review?",
    "How does on-call rotation work?",
    "What's the dress code in the office?",
    "Who do I contact for IT issues on my first day?",
    "What benefits do I get and when do they start?",
    "How are security incidents reported?",
    "What's our remote work policy?",
]
selected = st.selectbox("Example questions", [""] + EXAMPLES, index=0)
default_q = selected if selected else ""

q = st.text_input(
    "Your question",
    value=default_q,
    placeholder="Ask anything about the company knowledge base...",
)

if st.button("Ask", type="primary") and q.strip():
    with st.spinner("Retrieving + generating..."):
        result = engine.answer(q.strip(), top_k=top_k, alpha=alpha)

    st.subheader("Answer")
    st.info(result.answer)
    st.caption(f"Provider: **{result.provider}** · α (BM25 weight) = {alpha}")

    st.subheader("Retrieved contexts")
    for i, c in enumerate(result.contexts, 1):
        with st.expander(
            f"[{i}] {c.headline}  ·  hybrid score = {c.hybrid_score:.3f}"
        ):
            cols = st.columns(3)
            cols[0].metric("Hybrid", f"{c.hybrid_score:.3f}")
            cols[1].metric("BM25 (norm)", f"{c.bm25_score:.3f}")
            cols[2].metric("Dense (norm)", f"{c.dense_score:.3f}")
            st.markdown(f"**{c.headline}**")
            st.write(c.body)

    with st.expander("Show prompt sent to LLM"):
        st.code(result.prompt, language="text")
else:
    st.markdown(
        """
        **Try one of the example questions in the dropdown** or write your
        own — the assistant will:

        1. **Retrieve** the top-K most relevant docs using a weighted mix of
           BM25 (sparse) and dense embeddings (semantic).
        2. **Build a grounded prompt** that includes the retrieved chunks
           with numbered citations.
        3. **Call the LLM** with strict instructions to use only the
           provided context, and to cite sources by their bracket number.

        Adjust the **α slider** to see how the retrieval mix changes between
        keyword matching (α = 1) and semantic similarity (α = 0).
        """
    )
