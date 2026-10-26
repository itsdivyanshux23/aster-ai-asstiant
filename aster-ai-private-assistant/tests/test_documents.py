from src.documents import split_text, search_chunks, extract_text

def test_split_empty_text():
    assert split_text("") == []

def test_split_creates_chunks():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = split_text(text, source="notes.txt", chunk_words=100, overlap=10)
    assert len(chunks) > 1
    assert chunks[0]["source"] == "notes.txt"

def test_search_returns_relevant_chunk():
    chunks = [
        {"source": "os.txt", "chunk": 1, "text": "Process synchronization uses locks and semaphores."},
        {"source": "db.txt", "chunk": 1, "text": "SQL databases store records in tables."},
    ]
    results = search_chunks("semaphores synchronization", chunks)
    assert results
    assert results[0]["source"] == "os.txt"

def test_extract_txt():
    assert extract_text("notes.txt", b"hello world") == "hello world"
