import os
import streamlit as st
from google import genai
from google.genai import types

DEFAULT_MODEL = "gemini-flash-latest"


@st.cache_resource
def get_genai_client():
    """Cache the client connection so we don't recreate it on every message."""
    api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def check_ollama(base_url=None):
    api_key = st.secrets.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if api_key:
        return True, "Gemini API is configured."
    return False, "Add GEMINI_API_KEY in Streamlit Secrets."


@st.cache_data(ttl=3600)
def list_models(base_url=None):
    """Dynamically fetch valid text generation models only."""
    client = get_genai_client()
    if not client:
        return ["gemini-flash-latest", "gemini-3.8-flash"]

    try:
        valid_models = []
        exclude_tags = ["tts", "image", "embedding", "audio", "realtime", "2.0", "2.5"]

        for m in client.models.list():
            methods = getattr(m, "supported_generation_methods", []) or getattr(m, "supported_actions", [])
            name = m.name.replace("models/", "") if hasattr(m, "name") else ""

            if "generateContent" in methods and not any(tag in name.lower() for tag in exclude_tags):
                valid_models.append(name)

        for priority in ["gemini-flash-latest", "gemini-3.8-flash"]:
            if priority in valid_models:
                valid_models.remove(priority)
                valid_models.insert(0, priority)

        return valid_models if valid_models else ["gemini-flash-latest", "gemini-3.8-flash"]
    except Exception:
        return ["gemini-flash-latest", "gemini-3.8-flash"]


def stream_answer(
    question,
    context_chunks=None,
    history=None,
    model=DEFAULT_MODEL,
):
    """Generator that yields text chunks in real time as they arrive."""
    client = get_genai_client()
    if not client:
        yield "Error: GEMINI_API_KEY not found in Streamlit Secrets."
        return

    context_chunks = context_chunks or []
    history = history or []

    formatted_context_list = []
    for c in context_chunks:
        chunk_text = str(c.get("text", ""))
        formatted_context_list.append(f"\n{chunk_text}")

    context = "\n\n".join(formatted_context_list)

    if context:
        system_instruction = (
            "You are Aster AI, a careful study assistant. Answer using the supplied document excerpts "
            "when the question concerns the user's documents. Cite sources inline like [filename, section 2]. "
            "If the excerpts do not contain the answer, clearly say the uploaded documents do not provide "
            "enough evidence, then optionally provide general knowledge separately. Do not invent citations. "
            "Be clear and beginner-friendly."
        )
        user_content = "DOCUMENT EXCERPTS:\n" + context + "\n\nQUESTION:\n" + str(question)
    else:
        system_instruction = (
            "You are Aster AI, a helpful study assistant. Be accurate, concise, and beginner-friendly. "
            "If unsure, say so."
        )
        user_content = str(question)

    formatted_contents = []
    for item in history[-6:]:
        role = item.get("role")
        content = item.get("content", "")
        if role == "user":
            formatted_contents.append(
                types.Content(role="user", parts=[types.Part.from_text(text=content[:3000])])
            )
        elif role == "assistant":
            formatted_contents.append(
                types.Content(role="model", parts=[types.Part.from_text(text=content[:3000])])
            )

    formatted_contents.append(
        types.Content(role="user", parts=[types.Part.from_text(text=user_content)])
    )

    try:
        response_stream = client.models.generate_content_stream(
            model=model,
            contents=formatted_contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
            ),
        )
        for chunk in response_stream:
            if chunk.text:
                yield chunk.text
    except Exception as exc:
        yield f"\n\n*(Error generating response: {exc})*"
