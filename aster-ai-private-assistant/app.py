import streamlit as st
from src.assistant import answer_question, check_ollama, list_models
from src.database import init_db, save_message, load_messages, clear_messages
from src.documents import extract_text, split_text, index_chunks, search_chunks

st.set_page_config(page_title="Aster AI", page_icon="✨", layout="wide")
init_db()

st.title("✨ Aster AI")
st.caption("A private, local-first knowledge assistant. Your documents stay on this computer.")
with st.sidebar:
    st.header("Settings")
    ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
    available, status = check_ollama(ollama_url)
    if available:
        st.success("Ollama is reachable")
        models = list_models(ollama_url)
        if models:
            default_model = "llama3.2:1b" if "llama3.2:1b" in models else models[0]
            model = st.selectbox("Chat model", models, index=models.index(default_model))
        else:
            model = st.text_input("Chat model", value="llama3.2:1b")
            st.warning("No models found. Pull one with: ollama pull llama3.2:1b")
    else:
        model = st.text_input("Chat model", value="llama3.2:1b")
        st.error(status)
        st.markdown("Install and start Ollama, then run `ollama pull llama3.2:1b`.")
    st.divider()
    st.subheader("Your documents")
    uploads = st.file_uploader("Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True)
    if st.button("Index uploaded documents", use_container_width=True):
        if not uploads:
            st.warning("Choose at least one PDF or TXT file first.")
        else:
            all_chunks = []
            errors = []
            for uploaded in uploads:
                try:
                    raw = extract_text(uploaded.name, uploaded.getvalue())
                    chunks = split_text(raw, source=uploaded.name)
                    all_chunks.extend(chunks)
                except Exception as exc:
                    errors.append(f"{uploaded.name}: {exc}")
            st.session_state["chunks"] = all_chunks
            st.session_state["indexed_files"] = [u.name for u in uploads]
            if all_chunks:
                st.success(f"Indexed {len(all_chunks)} text sections from {len(uploads)} file(s).")
            else:
                st.warning("No readable text was found. Scanned PDFs need OCR before they can be searched.")
            for err in errors:
                st.error(err)
    if st.session_state.get("indexed_files"):
        st.caption("Current session: " + ", ".join(st.session_state["indexed_files"]))
        if st.button("Clear indexed documents", use_container_width=True):
            st.session_state.pop("chunks", None)
            st.session_state.pop("indexed_files", None)
            st.rerun()
    st.divider()
    if st.button("Clear chat history", use_container_width=True):
        clear_messages()
        st.session_state.pop("chat_messages", None)
        st.rerun()
    st.caption("Privacy note: chat and document contents are processed locally. Chat history is stored in a local SQLite database.")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = load_messages(limit=100)

for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask a question about your notes...")
if prompt:
    st.session_state.chat_messages.append({"role": "user", "content": prompt})
    save_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    chunks = st.session_state.get("chunks", [])
    retrieved = search_chunks(prompt, chunks, top_k=4) if chunks else []
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = answer_question(
                question=prompt,
                context_chunks=retrieved,
                history=st.session_state.chat_messages[-8:],
                base_url=ollama_url,
                model=model,
            )
        st.markdown(result["answer"])
        if result.get("sources"):
            with st.expander("Sources used"):
                for source in result["sources"]:
                    st.markdown(f"- **{source['source']}**, section {source['chunk']}: {source['preview']}")
        if result.get("error"):
            st.warning(result["error"])
    st.session_state.chat_messages.append({"role": "assistant", "content": result["answer"]})
    save_message("assistant", result["answer"])
