# app.py
# The chat interface for StudyRAG.
# Run with: streamlit run app.py

import streamlit as st
from chain import ask

st.set_page_config(
    page_title="StudyRAG",
    page_icon="📚",
    layout="centered",
)

with st.sidebar:
    st.title("📚 StudyRAG")
    st.caption("Ask questions about your study materials")
    st.divider()
    st.markdown("""
**How to use:**
1. Add PDFs or .txt files to ./docs/
2. Run python ingest.py once
3. Ask questions here!
""")
    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()

st.title("📚 Ask Your Study Notes")
st.caption("Powered by hybrid retrieval + citation enforcement")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources used"):
                for s in msg["sources"]:
                    st.write(f"- `{s}`")

if query := st.chat_input("Ask a question about your study materials..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching your notes..."):
            result = ask(query)

        if result["declined"]:
            st.warning(result["answer"])
        else:
            st.write(result["answer"])

        if result["sources"]:
            with st.expander("📎 Sources used"):
                for s in result["sources"]:
                    st.write(f"- `{s}`")

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
    })