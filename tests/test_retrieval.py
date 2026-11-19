from rag import load_documents, retrieve, build_context

def test_load_documents():
    assert len(load_documents()) == 6

def test_cancellation_policy_retrieved():
    hits = retrieve("How many hours before an appointment must I cancel?", k=2)
    assert "cancellations.txt" in [name for name, _, _ in hits]

def test_context_includes_source():
    assert "[Source: cancellations.txt]" in build_context([("cancellations.txt", "18 hours", 0.9)])

def test_empty_corpus():
    assert retrieve("question", documents=[]) == []

# Ensure queries about topics outside the knowledge base return no documents
    # rather than weak, incidental keyword matches (e.g., the shared word "clinic").
def test_unsupported_question_returns_no_documents():
    hits = retrieve("Does the clinic provide free transportation?", k=2)
    assert hits == []