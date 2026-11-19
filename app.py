"""Project VITAL chatbot with optional RAG retrieval."""
import streamlit as st
from llm import ask_llm
from rag import retrieve, build_context

st.title("Project VITAL — LLM + RAG Lab")
use_rag = st.sidebar.toggle("Enable RAG", value=False)
if "messages" not in st.session_state:
    st.session_state.messages = []
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
question = st.chat_input("Ask a question")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    try:
        if use_rag:
            hits = retrieve(question, k=2)
            context = build_context(hits)
            answer = ask_llm(question, context)
        else:
            hits = []
            answer = ask_llm(question)
    except Exception as exc:
        hits = []
        answer = f"Ollama unavailable: {exc}"
    with st.chat_message("assistant"):
        st.write(answer)
        if use_rag and hits:
            st.caption("Retrieved sources: " + ", ".join(filename for filename, text, score in hits))

    st.session_state.messages.append({"role": "assistant", "content": answer})
