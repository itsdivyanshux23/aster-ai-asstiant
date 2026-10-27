
import streamlit as st
from google import genai
from google.genai import types


MODEL = "gemini-3.8-flash"


def check_ollama(base_url=None):
    # Kept so older code importing this function will not break.
    if st.secrets.get("GEMINI_API_KEY"):
        return True, "Gemini API is configured."
    return False, "Add GEMINI_API_KEY in Streamlit Secrets."


def list_models(base_url=None):
    # Gemini model is configured directly in this file.
    return [MODEL]


def answer_question(
    question,
    context_chunks=None,
    history=None,
    base_url=None,
    model=MODEL,
):
    context_chunks = context_chunks or []
    history = history or []

    sources = [
        {
            "source": c["source"],
            "chunk": c["chunk"],
            "preview": c["text"][:220].replace("\n", " ")
            + ("…" if len(c["text"]) > 220 else ""),
        }
        for c in context_chunks
    ]

    context = "\n\n".join(
        f"[Source: {c['source']}, section {c['chunk']}]\n{c['text']}"
        for c in context_chunks
    )

    if context:
        system = (
            "You are Aster AI, a careful study assistant. "
            "Use supplied document excerpts for document questions. "
            "Cite evidence inline as [filename, section 2]. "
            "Never invent citations. If the documents do not contain "
            "the answer, say so clearly. Separate general knowledge "
            "from information supported by documents. "
            "Be clear and beginner-friendly."
        )
        prompt = f"{system}\n\nDOCUMENT EXCERPTS:\n{context}\n\n"
    else:
        prompt = (
            "You are Aster AI, a helpful and accurate assistant. "
            "Be concise and beginner-friendly. If unsure, say so.\n\n"
        )

    for item in history[-6:]:
        role = item.get("role")
        content = item.get("content", "")
        if role in ("user", "assistant") and content:
            prompt += f"{role.upper()}: {content[:3000]}\n\n"

    prompt += f"USER: {question}"

    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if not api_key:
            return {
                "answer": "Gemini API key is missing. Add GEMINI_API_KEY in your Streamlit app's Secrets settings.",
                "sources": [],
                "error": "Missing API key.",
            }

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=model or MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                system_instruction=(
                    "You are Aster AI. Follow the instructions in the user's "
                    "prompt, answer accurately, and never invent document citations."
                ),
            ),
        )

        answer = (response.text or "").strip()
        if not answer:
            raise ValueError("Gemini returned an empty response.")

        return {
            "answer": answer,
            "sources": sources,
            "error": None,
        }

    except Exception as exc:
        return {
            "answer": (
                "I couldn't get a response from Gemini. Check your API key, "
                "internet connection, model access, and API quota, then try again."
            ),
            "sources": [],
            "error": str(exc),
        }
