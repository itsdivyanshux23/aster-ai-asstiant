# Aster AI — Private Local Knowledge Assistant

A beginner-friendly, local-first AI assistant that chats with a locally running Ollama model and can answer questions using uploaded PDF/TXT documents.

## Features

- Local LLM responses through Ollama; no paid AI API key.
- Upload PDF and TXT documents.
- Extract text and split it into overlapping chunks.
- Retrieve relevant passages with dependency-free TF-IDF cosine similarity.
- Prompt the model to cite document and section sources and admit when evidence is missing.
- Save chat history locally in SQLite.
- Basic automated tests with pytest.
- Streamlit chat interface.

## Requirements

- Windows, macOS, or Linux
- Python 3.10+
- [Ollama](https://ollama.com/download)
- A small local model. Start with `llama3.2:1b`; use a larger model only if your machine handles it comfortably.

## Setup (Windows PowerShell)

```powershell
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd aster-ai-private-assistant

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

ollama pull llama3.2:1b
ollama serve
```

If Ollama is already running as a desktop application, you may not need to run `ollama serve` separately. Open a second terminal, activate the environment, and run:

```powershell
python -m streamlit run app.py
```

Then open the local address Streamlit prints, usually `http://localhost:8501`.

## Tests

```powershell
python -m pytest -q
```

## How document Q&A works

1. The app extracts text from uploaded PDF/TXT files.
2. Text is split into overlapping sections.
3. A TF-IDF cosine-similarity retriever selects relevant sections.
4. The selected excerpts are added to the local model prompt.
5. The assistant is asked to cite sources and avoid claiming the documents support information they do not contain.

This first version uses lexical TF-IDF retrieval, not embedding-based semantic search. It is lightweight and easy to inspect, but may miss passages that use different words from the question.

## Privacy and limitations

- Chat history is saved in `aster_history.db` on your computer.
- Uploaded document contents are held in app memory for the current session and are not automatically persisted.
- Ollama processes prompts locally when configured to use the local server.
- Local model quality and speed depend on your hardware.
- Scanned/image-only PDFs need OCR; this version does not perform OCR.
- Source references identify retrieved excerpts; they do not guarantee that the generated answer is correct.
- Do not upload confidential documents to systems you do not control, and do not commit personal files, databases, model weights, or secrets to GitHub.

## Suggested next improvements

- Add embedding-based retrieval using a local embedding model.
- Add per-document delete controls and conversation export.
- Add evaluation examples to measure retrieval quality and hallucination rate.
- Add tests for model errors and PDF edge cases.
- Add a screenshot and short demo video.

## Tech stack

Python, Streamlit, Ollama, PyMuPDF, SQLite, pytest, Git.
