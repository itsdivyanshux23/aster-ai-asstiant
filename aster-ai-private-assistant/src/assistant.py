import os
import time
import streamlit as st
from google import genai
from google.genai import types
from google.genai.errors import APIError

FALLBACK_MODELS = [
    "gemini-flash-latest",
    model = "gemini-3.5-flash-lite",
]

DEFAULT_MODEL = FALLBACK_MODELS[0]


@st.cache_resource
def get_genai_client():
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
    client = get_genai_client()
    if not client:
        return FALLBACK_MODELS

    try:
        valid_models = []
        exclude_tags = ["tts", "image", "embedding", "audio", "realtime", "2.0", "2.5"]

        for m in client.models.list():
            methods = getattr(m, "supported_generation_methods", []) or getattr(m, "supported_actions", [])
            name = m.name.replace("models/", "") if hasattr(m, "name") else ""

            if "generateContent" in methods and not any(tag in name.lower() for tag in exclude_tags):
                valid_models.append(name)

        for priority in FALLBACK_MODELS:
            if priority in valid_models:
                valid_models.remove(priority)
                valid_models.insert(0, priority)

        return valid_models if valid_models else FALLBACK_MODELS
    except Exception:
        return FALLBACK_MODELS


def stream_answer(
    question,
    context_chunks=None,
    history=None,
    model=DEFAULT_MODEL,
):
    client = get_genai_client()
    if not client:
        yield "Error: `GEMINI_API_KEY` was not found in Streamlit Secrets."
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

    # Build fallback queue: requested model first, followed by remaining valid models
    candidates = [model] + [m for m in FALLBACK_MODELS if m != model]
    last_error = None

    for target_model in candidates:
        try:
            response_stream = client.models.generate_content_stream(
                model=target_model,
                contents=formatted_contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.2,
                ),
            )
            
            # Verify stream validity before committing
            has_yielded = False
            for chunk in response_stream:
                if chunk.text:
                    has_yielded = True
                    yield chunk.text

            if has_yielded:
                return

        except APIError as exc:
            last_error = exc
            # If 503 (high demand) or 429 (rate limit), jump immediately to the fallback model
            if exc.code in (503, 429):
                time.sleep(1)
                continue
            break
        except Exception as exc:
            last_error = exc
            break

    yield f"The AI servers are currently experiencing high demand. Please try again in a few moments. *(Details: {last_error})*"
