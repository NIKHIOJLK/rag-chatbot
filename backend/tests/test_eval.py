from eval.evaluate import evaluate, is_relevant, load_questions, tune_threshold


def test_question_file_is_well_formed():
    in_scope, out_scope = load_questions()
    assert len(in_scope) >= 50 and len(out_scope) >= 10
    assert all(q["page"].endswith(".html") and q["keywords"] for q in in_scope)


def test_relevance_needs_right_page_and_keyword():
    chunk = {"source": "https://docs.python.org/3/tutorial/datastructures.html#tuples", "text": "A tuple is ..."}
    q = {"page": "tutorial/datastructures.html", "keywords": ["tuple"]}
    assert is_relevant(chunk, q)
    assert not is_relevant({**chunk, "text": "nothing here"}, q)
    assert not is_relevant({**chunk, "source": "https://docs.python.org/3/tutorial/classes.html"}, q)


def test_threshold_separates_real_and_off_topic_scores():
    t, acc = tune_threshold([0.8, 0.75, 0.9], [0.4, 0.5, 0.45])
    assert 0.5 < t <= 0.75 and acc == 1.0


def test_evaluate_reports_metrics(retriever):
    in_scope = [
        {"question": "use a list as a stack", "page": "tutorial/datastructures.html", "keywords": ["stack"]},
        {"question": "tuples immutable", "page": "tutorial/datastructures.html", "keywords": ["tuple"]},
    ]
    out = evaluate(retriever, in_scope, [{"question": "kubernetes ingress"}])
    assert out["hit3"] == 1.0 and out["mrr"] > 0.5
    assert len(out["out_best"]) == 1 and out["p50"] >= 0
