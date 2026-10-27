
import streamlit as st
from google import genai
from google.genai import types

MODEL = "gemini-3.5-flash-lite"


def check_ollama(base_url=None):
    # Compatibility with older code.
    if st.secrets.get("GEMINI_API_KEY", ""):
        return True, "Gemini API is configured."
    return False, "GEMINI_API_KEY is missing from Streamlit Secrets."


def list_models(base_url=None):
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

    instructions = (
        "You are Aster AI, a helpful and accurate study assistant. "
        "Be clear and beginner-friendly. Never invent facts or citations. "
        "When document excerpts are supplied, use them for document questions. "
        "Cite document evidence as [filename, section 2]. "
        "If the documents do not contain the answer, say so clearly."
    )

    prompt_parts = [instructions]

    if context:
        prompt_parts.append("DOCUMENT EXCERPTS:\n" + context)

    # Include recent conversation for context.
    for item in history[-6:]:
        role = item.get("role")
        content = item.get("content", "")

        if role in ("user", "assistant") and content:
            prompt_parts.append(f"{role.upper()}: {content[:3000]}")

    prompt_parts.append("USER: " + question)
    prompt = "\n\n".join(prompt_parts)

    try:
        api_key = st.secrets.get("GEMINI_API_KEY", "")

        if not api_key:
            return {
                "answer": (
                    "Gemini API key is missing. Add GEMINI_API_KEY "
                    "in your Streamlit Cloud Secrets settings."
                ),
                "sources": [],
                "error": "Missing API key.",
            }

        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=model or MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=2048,
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
        error = str(exc)
        lower_error = error.lower()

        if "429" in error or "resource_exhausted" in lower_error:
            message = (
                "Aster AI has reached the Gemini API quota. "
                "Check your project's quota and reset time. "
                "Changing models does not guarantee more free requests."
            )
        elif "401" in error or "403" in error:
            message = (
                "Gemini authentication or access failed. "
                "Check your API key and project permissions."
            )
        elif "400" in error or "invalid_argument" in lower_error:
            message = (
                "Gemini rejected this request. Check that the selected "
                "model is available to your API project."
            )
        else:
            message = (
                "Aster AI could not get a response from Gemini. "
                "Check your API access and app logs."
            )

        return {
            "answer": message,
            "sources": [],
            "error": error,
        }
