import streamlit as st

import config
from rag import retrieve, answer, get_collection, page_label

st.set_page_config(page_title="Paper Q&A", layout="wide")
st.title("Paper Q&A")
st.caption("Ask questions across your research papers. Every answer cites the passages it used.")

# Sidebar settings
with st.sidebar:
    st.header("Settings")
    collection_name = st.text_input("Collection", config.COLLECTION)
    k = st.slider("Passages to retrieve (k)", 1, 10, 5)
    rerank = st.toggle("Rerank with cross-encoder", value=False, help="Slower, but usually more accurate")

    n_chunks = get_collection(collection_name).count()
    st.metric("Indexed chunks", n_chunks)
    st.caption(f"Embeddings: `{config.EMBED_MODEL}`  \nLLM: `{config.LLM_MODEL}`")

    if st.button("Clear chat"):
        st.session_state.history = []

if n_chunks==0:
    st.info("No papers indexed yet. Put PDFs in `papers/` (or run `fetch_papers.py`), then run `python ingest.py`.")
    st.stop()


def show_sources(hits):

    with st.expander(f"Sources ({len(hits)})"):
        for i,h in enumerate(hits, 1):

            title = h["title"]
            if h.get("url"):
                title = f"[{title}]({h['url']}#page={h['page_start']})"   # opens the pdf at that page

            st.markdown(f"**[{i}]** {title}, {page_label(h)} · score `{round(h['score'],3)}`")

            text = h["text"]
            if len(text)>700:
                text = text[:700] + "…"
            st.caption(text)


if "history" not in st.session_state:
    st.session_state.history = []

# Showing the previous questions and answers
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        show_sources(turn["hits"])

question = st.chat_input("e.g. Which methods detect poisoned samples using frozen feature extractors?")

if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching papers..."):
            hits = retrieve(question, collection_name, k=k, rerank=rerank)
        try:
            text = st.write_stream(answer(question, hits, stream=True))
        except Exception as e:
            text = f"Couldn't reach the LLM at `{config.LLM_BASE_URL}`: {e}"
            st.error(text)
        show_sources(hits)

    st.session_state.history.append({"question": question, "answer": text, "hits": hits})
