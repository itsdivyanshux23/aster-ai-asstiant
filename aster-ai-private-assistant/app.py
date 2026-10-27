
import streamlit as st

from src.assistant import answer_question, MODEL
from src.database import (
    init_db,
    save_message,
    load_messages,
    clear_messages,
)
from src.documents import extract_text, split_text, search_chunks


st.set_page_config(
    page_title="Aster AI",
    page_icon="✨",
    layout="wide",
)

init_db()

st.title("✨ Aster AI")
st.caption(
    "Your personal study assistant powered by Gemini AI "
    "and document search."
)

with st.sidebar:
    st.header("Settings")
    st.subheader("AI Model")
    st.info(f"Model: {MODEL}")

    if st.button("Check API configuration", use_container_width=True):
        if st.secrets.get("GEMINI_API_KEY", ""):
            st.success("API key is configured.")
        else:
            st.error("GEMINI_API_KEY is missing from Streamlit Secrets.")

    st.divider()
    st.subheader("Your documents")

    uploads = st.file_uploader(
        "Upload PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button("Index uploaded documents", use_container_width=True):
        if not uploads:
            st.warning("Choose at least one PDF or TXT file first.")
        else:
            all_chunks = []
            errors = []

            for uploaded in uploads:
                try:
                    raw = extract_text(
                        uploaded.name,
                        uploaded.getvalue(),
                    )
                    chunks = split_text(raw, source=uploaded.name)
                    all_chunks.extend(chunks)
                except Exception as exc:
                    errors.append(f"{uploaded.name}: {exc}")

            st.session_state["chunks"] = all_chunks
            st.session_state["indexed_files"] = [
                uploaded.name for uploaded in uploads
            ]

            if all_chunks:
                st.success(
                    f"Indexed {len(all_chunks)} sections "
                    f"from {len(uploads)} file(s)."
                )
            else:
                st.warning(
                    "No readable text was found. Scanned PDFs "
                    "may require OCR."
                )

            for error in errors:
                st.error(error)

    if st.session_state.get("indexed_files"):
        st.caption(
            "Indexed: "
            + ", ".join(st.session_state["indexed_files"])
        )

        if st.button("Clear indexed documents", use_container_width=True):
            st.session_state.pop("chunks", None)
            st.session_state.pop("indexed_files", None)
            st.rerun()

    st.divider()

    if st.button("Clear chat history", use_container_width=True):
        clear_messages()
        st.session_state.pop("chat_messages", None)
        st.rerun()

    st.caption(
        "Privacy: Questions and relevant document excerpts are sent "
        "to Google's Gemini API. Avoid uploading sensitive information. "
        "Cloud SQLite history may not persist across app restarts."
    )


# Load saved chat history
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = load_messages(limit=100)


# Display previous messages
for msg in st.session_state.chat_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# Chat input
prompt = st.chat_input("Ask a question about your notes...")

if prompt:
    st.session_state.chat_messages.append(
        {"role": "user", "content": prompt}
    )
    save_message("user", prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    chunks = st.session_state.get("chunks", [])
    retrieved = search_chunks(prompt, chunks, top_k=4) if chunks else []

    with st.chat_message("assistant"):
        with st.spinner("Aster AI is thinking..."):
            result = answer_question(
                question=prompt,
                context_chunks=retrieved,
                history=st.session_state.chat_messages[-8:],
                model=MODEL,
            )

        st.markdown(result["answer"])

        if result.get("sources"):
            with st.expander("Sources used"):
                for source in result["sources"]:
                    st.markdown(
                        f"- **{source['source']}**, "
                        f"section {source['chunk']}: "
                        f"{source['preview']}"
                    )

        if result.get("error"):
            with st.expander("Technical error details"):
                st.code(result["error"])

    st.session_state.chat_messages.append(
        {"role": "assistant", "content": result["answer"]}
    )
    save_message("assistant", result["answer"])
