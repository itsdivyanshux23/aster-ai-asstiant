import os
import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import APIError

# Preferred models in order of fallback
FALLBACK_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
]

MODEL = FALLBACK_MODELS[0]


def check_ollama(base_url=None):
    """Compatibility stub so any legacy calls won't crash."""
    api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if api_key:
        return True, "Gemini API is configured."
    return False, "Add GEMINI_API_KEY in Streamlit Secrets."


def list_models(base_url=None):
    return FALLBACK_MODELS


def answer_question(
    question,
    context_chunks=None,
    history=None,
    base_url=None,
    model=MODEL,
):
    context_chunks = context_chunks or []
    history = history or []

    api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return {
            "answer": "Error: GEMINI_API_KEY was not found in Streamlit Secrets.",
            "sources": [],
            "error": "Missing GEMINI_API_KEY",
        }

    sources = [
        {
            "source": c["source"],
            "chunk": c["chunk"],
            "preview": c["text"][:220].replace("\n", " ") + ("…" if len(c["text"]) > 220 else ""),
        }
        for c in context_chunks
    ]

    context = "\n\n".join(
        f"}, section {c['chunk']}]\n{c['text']}"
        for c in context_chunks
    )

    if context:
        system_instruction = (
            "You are Aster AI, a careful study assistant. Answer using the supplied document excerpts "
            "when the question concerns the user's documents. Cite sources inline like [filename, section 2]. "
            "If the excerpts do not contain the answer, clearly say the uploaded documents do not provide "
            "enough evidence, then optionally provide general knowledge separately. Do not invent citations. "
            "Be clear and beginner-friendly."
        )
        user_content = f"DOCUMENT EXCERPTS:\n{context}\n\nQUESTION:\n{question}"
    else:
        system_instruction = (
            "You are Aster AI, a helpful study assistant. Be accurate, concise, and beginner-friendly. "
            "If unsure, say so."
        )
        user_content = question

    # Prepare chat history for GenAI SDK
    formatted_contents = []
    for item in history[-6:]:
        role = item.get("role")
        content = item.get("content", "")
        if role == "user":
            formatted_contents.append(types.Content(role="user", parts=[types.Part.from_text(text=content[:3000])]))
        elif role == "assistant":
            formatted_contents.append(types.Content(role="model", parts=[types.Part.from_text(text=content[:3000])]))

    formatted_contents.append(types.Content(role="user", parts=[types.Part.from_text(text=user_content)]))

    client = genai.Client(api_key=api_key)

    # Try requested model first, then fall back if overloaded
    models_to_try = [model] + [m for m in FALLBACK_MODELS if m != model]
    last_error = None

    for target_model in models_to_try:
        try:
            response = client.models.generate_content(
                model=target_model,
                contents=formatted_contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                ),
            )
            answer = response.text.strip() if response.text else "No response generated."
            return {
                "answer": answer,
                "sources": sources,
                "error": None,
            }
        except APIError as exc:
            last_error = exc
            if exc.code in (503, 429):
                continue
            break
        except Exception as exc:
            last_error = exc
            break

    return {
        "answer": "Gemini models are currently experiencing high demand. Please try asking again in a few seconds.",
        "sources": sources,
        "error": str(last_error),
    }
