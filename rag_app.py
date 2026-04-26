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

with st.expander("How to use this app", expanded=False):
    st.markdown("""
**What this app does in plain English.**
Imagine a chatbot that can only answer using a fixed set of documents —
it can't make stuff up. This is called **RAG** (retrieval-augmented
generation). Type a question, the app searches its mini knowledge base
of 53 short documents (work policies, cooking, household tips, health,
finance, travel), picks the most relevant ones, and asks an AI model
(Google Gemini) to write the answer using *only* those documents.

**Quick start (30 seconds).**
1. Pick a question from the **Example questions** dropdown — try
   "How do I cook pasta properly?"
2. Click **Ask**.
3. Read the **Answer** at the top, then expand the **Retrieved
   contexts** below to see exactly which documents the AI used.

**What the sliders mean.**
- **BM25 weight (α)** — how the app finds relevant documents. There are
  two methods:
    - **Keyword matching** (BM25) — finds documents that share exact
      words with your question. Great for specific terms.
    - **Meaning matching** (dense embeddings) — finds documents that
      mean the same thing even if they use different words. Great for
      paraphrases ("How do I take time off?" still finds the PTO doc).
    - **α = 1.0** → only keyword. **α = 0.0** → only meaning. **0.5** →
      equal blend (recommended).
- **Top-K retrieved docs** — how many documents to feed to the AI.
  More = more context but slower.

**What you'll see in the answer.**
The AI cites its sources with bracketed numbers like `[1]` and `[2]`.
Those numbers match the **Retrieved contexts** expanders below — click
to read the actual source. The "Show prompt sent to LLM" panel shows
exactly what we asked the AI, for transparency.

**Try this scenario.** Ask "How do I sleep better?" and slide α from
1.0 down to 0.0 — at α=1.0 the keyword-only search might miss the
sleep-tips doc if it uses different words; at α=0.0 the meaning-based
search will find it.
""")

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
    has_gemini = bool(os.getenv("GEMINI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    if has_gemini:
        st.success("Gemini key detected (gemini-2.5-flash)")
    elif has_anthropic:
        st.success("Anthropic key detected (Claude Haiku 4.5)")
    elif has_openai:
        st.success("OpenAI key detected (GPT-4o-mini)")
    else:
        st.warning("No API key set — running in extractive-fallback mode")
        st.code(
            "GEMINI_API_KEY=...    # free tier at aistudio.google.com\n"
            "ANTHROPIC_API_KEY=sk-ant-...\n"
            "OPENAI_API_KEY=sk-...",
            language="bash",
        )
        st.caption(
            "Set any one of these env vars before launching Streamlit to "
            "enable LLM-grounded answers."
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
    # Everyday life
    "How do I cook pasta properly?",
    "How long should I boil an egg for a runny yolk?",
    "What can I do to sleep better?",
    "How much water should I drink each day?",
    "How do I unclog a drain without using harsh chemicals?",
    "What's a good way to start exercising?",
    "How does compound interest actually work?",
    "What should I do if my flight gets delayed?",
    "Why should I use a password manager?",
    # Work / company knowledge base
    "How do I request paid time off?",
    "What's the process for code review?",
    "How does on-call rotation work?",
    "What's our remote work policy?",
    "How are security incidents reported?",
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
