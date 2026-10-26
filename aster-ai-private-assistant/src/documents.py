
import math
import re
from collections import Counter


def extract_text(filename, content):
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if suffix == "txt":
        return content.decode("utf-8", errors="replace").strip()

    if suffix == "pdf":
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError(
                "PyMuPDF is missing. Install requirements.txt."
            ) from exc

        with fitz.open(stream=content, filetype="pdf") as doc:
            pages = [page.get_text("text") for page in doc]

        return "\n".join(pages).strip()

    raise ValueError("Only PDF and TXT files are supported.")


def split_text(text, source="document", chunk_words=180, overlap=35):
    words = text.split()

    if not words:
        return []

    chunks = []
    start = 0
    number = 1

    while start < len(words):
        end = min(start + chunk_words, len(words))
        piece = " ".join(words[start:end]).strip()

        if piece:
            chunks.append({
                "source": source,
                "chunk": number,
                "text": piece
            })
            number += 1

        if end == len(words):
            break

        start = max(end - overlap, start + 1)

    return chunks


def index_chunks(text, source="document"):
    return split_text(text, source=source)


def _tokens(text):
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def search_chunks(query, chunks, top_k=4):
    """Simple TF-IDF cosine search for document retrieval."""

    if not chunks:
        return []

    query_tokens = _tokens(query)

    if not query_tokens:
        return []

    docs = [_tokens(chunk["text"]) for chunk in chunks]

    df = Counter()

    for tokens in docs:
        df.update(set(tokens))

    n = len(docs)

    def vector(tokens):
        counts = Counter(tokens)
        total = max(len(tokens), 1)
        result = {}

        for term, count in counts.items():
            idf = math.log(1 + n / (1 + df[term]))
            result[term] = (count / total) * idf

        return result

    qv = vector(query_tokens)
    qnorm = math.sqrt(sum(v * v for v in qv.values())) or 1.0
    scored = []

    for chunk, tokens in zip(chunks, docs):
        dv = vector(tokens)

        dot = sum(
            value * dv.get(term, 0.0)
            for term, value in qv.items()
        )

        dnorm = math.sqrt(sum(v * v for v in dv.values())) or 1.0
        score = dot / (qnorm * dnorm)

        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    return [
        dict(chunk, score=round(score, 4))
        for score, chunk in scored[:top_k]
    ]
