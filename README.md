# Project VITAL — LLM + RAG Testing Tutorial (Student Starter)

**CSCI3364 — Software Testing, Quality and Security | Boston College**  

In this tutorial you will build a local LLM chatbot, extend it with retrieval-augmented generation (RAG), and design tests that distinguish retrieval failures from generation failures.

## Before class

1. Install [Ollama](https://ollama.com/download).
2. Run `ollama pull qwen2.5:1.5b` (approximately 1 GB).
3. Check that `ollama run qwen2.5:1.5b` answers a question; type `/bye` to exit.
4. Install Python 3.10 or later.

## Run the baseline chatbot

```bash
git clone https://github.com/CSCI3364-Fall2026/VITAL-RAG-Starter.git
cd VITAL-RAG-Starter
python3 -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

The baseline chatbot works without RAG. If Ollama is unavailable, check that its service is running and the model is downloaded.

## Lab schedule

| Activity |
| --- | --- |
| LLMs, hallucinations, RAG and test oracles |
| Run chatbot and establish baseline responses |
| Load policy documents and implement TF-IDF retrieval |
| Add source context to model calls and a RAG toggle |
| Design and run automated tests; compare answers |
| Interpret results and report conclusions |

## Tasks (write the code yourself)

1. **Baseline**: Ask three policy questions before implementing RAG. Record the answers.
2. **TODO 1 — load documents** in `rag.py`. Return sorted `(filename, text)` tuples for `.txt` files.
3. **TODO 2 — retrieve** in `rag.py`. Use `TfidfVectorizer` and cosine similarity to rank documents. Return at most `k` `(filename, text, score)` tuples. Exclude zero-score results and break ties by filename.
4. **TODO 3 — format context** in `rag.py`. Prefix each selected document with `[Source: filename]`.
5. **TODO 4 — integrate RAG** in `app.py`. Add a sidebar **Enable RAG** toggle and send context to `ask_llm(question, context)` when enabled. Display retrieved filenames to the user.
6. **Test and improve**: run `pytest -q`, add at least one test, investigate a failure, make one improvement, and rerun.

The included `tests/test_retrieval.py` intentionally **fails initially** because the retrieval functions are incomplete. Passing those tests is a milestone, not a guarantee that generated answers are correct.

## Suggested questions

- How many hours before an appointment must I cancel?
- Can I request a same-day appointment?
- How are laboratory results communicated?
- Does the clinic provide free transportation? (The documents do not answer this.)
- What happens when I ask about an unsupported policy?

## Evidence to submit

For at least four questions, record: (a) question, (b) expected source, (c) retrieved filenames, (d) response without RAG, (e) response with RAG, and (f) whether the response is **supported by the source**. Explain one failure, your change, and the outcome of rerunning the test. Distinguish failures in retrieval from failures in generation.


## RAG Evaluation and Testing

### 1. RAG OFF vs. RAG ON

I tested four questions with RAG disabled and enabled to evaluate whether retrieving clinic policy documents improved the chatbot's responses.

| Question | Expected Source | Retrieved Filenames (RAG ON) | Response Without RAG | Response With RAG | Supported by Source? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **How many hours before an appointment must I cancel?** | `cancellations.txt` | `cancellations.txt`, `scheduling.txt` | "I'm sorry, but I can't answer this question as it doesn't provide enough context. Could you please provide more information or clarify the question?" | "You must cancel your appointment at least 18 hours before the scheduled time." | **Yes.** The cancellation policy requires at least 18 hours' notice. |
| **Can I request a same-day appointment?** | `scheduling.txt` | `scheduling.txt` | "I'm sorry, but I can't assist with that." | "Yes, you can request a same-day appointment, but it must be available." | **Yes.** Same-day requests depend on availability. |
| **How are laboratory results communicated?** | `lab_results.txt` | `lab_results.txt`, `portal.txt` | "Laboratory results are typically communicated in several ways, depending on the context and the nature of the results. Here are some common methods: 1. Written Reports: Laboratory results are often reported in written form, either as a report or as part of a larger report. This can be sent to the patient, the referring physician, or the patient's insurance company. 2. Email or Electronic Health Records (EHR): Results can be sent electronically via email or integrated into electronic health records (EHR) systems. This is particularly useful for sharing results with multiple healthcare providers. 3. Phone Calls: Results can be communicated over the phone, especially if the results are urgent or require immediate action. 4. In-Person Consultations: In some cases, results may be discussed in person, either at the patient's request or as part of a follow-up appointment. 5. Notifications: Some laboratories may send notifications via text message or mobile app, especially for results that are not immediately actionable. 6. Patient Education: In some cases, the laboratory may provide educational materials or information about the results, especially if the results are complex or require further explanation. 7. Follow-Up Appointments: Results may be discussed during follow-up appointments, either in person or via telehealth. The method of communication is often determined by the urgency of the results, the preferences of the patient or healthcare provider, and the specific requirements of the laboratory and the healthcare system." | "Laboratory results are communicated through the secure patient portal." | **Yes.** The laboratory results policy specifies communication through the secure portal. |
| **Does the clinic provide free transportation?** | None | None (`[]`) | "I'm sorry, but I can't answer this question because I don't have any specific information about a particular clinic. The answer would depend on the clinic's policies and the location. If you know the name of the clinic or the location, you can search online or contact them directly to find out if they provide free transportation." | "I do not have enough information in the provided documents." | **N/A.** No policy covers transportation, and the chatbot appropriately declined to provide an unsupported answer. |

### 2. Test, Failure, and Improvement

**New test:** I added `test_unsupported_question_returns_no_documents()` to `tests/test_retrieval.py`. The test checks whether the retriever returns an empty list when asked about free transportation, which is not addressed in the clinic's knowledge base.

**Failure:** The new test initially failed (1 failed, 4 passed). Rather than returning [], the retriever returned unrelated documents, including billing.txt and lab_results.txt. The highest similarity score was approximately 0.071. This was a retrieval failure because TF-IDF matched shared words such as "clinic" even though the documents did not answer the question. A generation failure would occur if the language model produced an incorrect or unsupported answer using the retrieved context. In this case, the chatbot appropriately stated that it did not have enough information, so the problem was in document retrieval rather than the generated response.

**Improvement:** I compared the similarity scores for supported questions (0.164 for cancellations, 0.404 for laboratory results, and 0.441 for same-day appointments) against the unsupported transportation question (0.071). Based on these results, I introduced a minimum similarity threshold of 0.10 in `rag.py`, replacing the original condition that accepted any positive similarity score.

**Retesting:** After making this change, I reran `python3 -m pytest -q` and all five tests passed. I also manually confirmed that the transportation question now retrieves no documents, while the cancellation and same-day appointment questions still retrieve the correct policies and generate supported answers.

**Limitations:** Although the threshold reduced irrelevant retrievals for the transportation question, it could also exclude relevant documents with low similarity scores. Additionally, passing the retrieval tests does not guarantee that the chatbot will generate accurate answers, so I evaluated its responses separately against the clinic policies.


[Project VITAL course repository](https://github.com/CSCI3364-Fall2026/Project_VITAL)
