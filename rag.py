"""Student exercise: implement retrieval over the fictional clinic policies."""
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DOCS = Path(__file__).parent / "knowledge_base"

def load_documents(directory=DOCS):
    """Return list of (filename, text) for *.txt files."""
    documents = []

    for file in sorted(Path(directory).glob("*.txt")):
        text = file.read_text(encoding="utf-8")
        documents.append((file.name, text))

    return documents

def retrieve(question: str, k: int = 2, documents=None):
    """Return top-k (filename, text, score) matches using TF-IDF."""
    # Exclude zero-score matches; break ties by filename.
    docs = load_documents() if documents is None else documents

    if not docs:
        return []

    vectorizer = TfidfVectorizer()
    texts = [text for filename, text in docs]
    vectors = vectorizer.fit_transform(texts + [question])
    scores = cosine_similarity(vectors[-1], vectors[:-1]).ravel()

    # Filter out weak, incidental keyword matches (e.g., shared words like "clinic") using an empirically chosen minimum TF-IDF cosine similarity threshold.
    min_score = 0.10
    matches = [
        (docs[i][0], docs[i][1], scores[i])
        for i in range(len(docs))
        if scores[i] >= min_score
    ]

    matches.sort(key=lambda x: (-x[2], x[0]))  # Sort by score (descending) then filename (ascending)
    return matches[:k]

def build_context(results):
    """Format hits with [Source: filename] headers."""
    # Prompt context.
    sections = []
    for filename, text, score in results:
        sections.append(f"[Source: {filename}]\n{text}")
    return "\n\n".join(sections)
    
