import requests

def check_ollama(base_url="http://localhost:11434"):
    try:
        response = requests.get(base_url.rstrip("/") + "/api/tags", timeout=3)
        response.raise_for_status()
        return True, "Ollama is running."
    except requests.RequestException:
        return False, "Cannot connect to Ollama. Install it, open Ollama, and try again."

def list_models(base_url="http://localhost:11434"):
    try:
        response = requests.get(base_url.rstrip("/") + "/api/tags", timeout=5)
        response.raise_for_status()
        return [m.get("name", "") for m in response.json().get("models", []) if m.get("name")]
    except (requests.RequestException, ValueError):
        return []

def answer_question(question, context_chunks=None, history=None,
                    base_url="http://localhost:11434", model="llama3.2:1b"):
    context_chunks = context_chunks or []
    history = history or []
    context = "\n\n".join(
        f"[Source: {c['source']}, section {c['chunk']}]\n{c['text']}"
        for c in context_chunks
    )
    if context:
        system = (
            "You are Aster AI, a careful study assistant. Answer using the supplied document excerpts "
            "when the question concerns the user's documents. Cite sources inline like [filename, section 2]. "
            "If the excerpts do not contain the answer, clearly say the uploaded documents do not provide "
            "enough evidence, then optionally give general knowledge separately. Do not invent citations. "
            "Be clear and beginner-friendly."
        )
        user_content = f"DOCUMENT EXCERPTS:\n{context}\n\nQUESTION:\n{question}"
    else:
        system = (
            "You are Aster AI, a helpful assistant running locally. Be accurate, concise, and beginner-friendly. "
            "If unsure, say so."
        )
        user_content = question
    messages = [{"role": "system", "content": system}]
    for item in history[-6:]:
        if item.get("role") in ("user", "assistant") and item.get("content"):
            messages.append({"role": item["role"], "content": item["content"][:3000]})
    messages.append({"role": "user", "content": user_content})
    try:
        response = requests.post(
            base_url.rstrip("/") + "/api/chat",
            json={"model": model, "messages": messages, "stream": False,
                  "options": {"temperature": 0.2}},
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        answer = data.get("message", {}).get("content", "").strip()
        if not answer:
            raise ValueError("Ollama returned an empty response.")
        return {
            "answer": answer,
            "sources": [
                {"source": c["source"], "chunk": c["chunk"],
                 "preview": c["text"][:220].replace("\n", " ") + ("…" if len(c["text"]) > 220 else "")}
                for c in context_chunks
            ],
            "error": None,
        }
    except requests.Timeout:
        return {"answer": "The local model took too long to respond. Try a smaller model or a shorter question.",
                "sources": [], "error": "Request timed out."}
    except requests.ConnectionError:
        return {"answer": "I cannot reach Ollama right now. Start Ollama and make sure the selected model is installed.",
                "sources": [], "error": "Ollama connection failed."}
    except (requests.RequestException, ValueError) as exc:
        return {"answer": f"I couldn't get a valid response from Ollama. Check the model name and Ollama status.",
                "sources": [], "error": str(exc)}
