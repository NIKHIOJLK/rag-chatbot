from app.retriever import reciprocal_rank_fusion, tokenize


def test_rrf_rewards_items_ranked_high_in_both_lists():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]])
    assert fused[0] == "b"


def test_rrf_keeps_items_found_by_only_one_method():
    assert set(reciprocal_rank_fusion([["a"], ["z"]])) == {"a", "z"}


def test_tokenize_keeps_python_identifiers():
    assert tokenize("What does __init__ do?") == ["what", "does", "__init__", "do"]


def test_search_finds_the_right_section(retriever):
    hits, best = retriever.search("use a list as a stack with pop", k=2)
    assert hits[0].chunk["source"].endswith("#using-lists-as-stacks")
    assert 0 < best <= 1


def test_keyword_search_finds_exact_terms(retriever):
    hits, _ = retriever.search("tuple immutable", k=1)
    assert hits[0].chunk["source"].endswith("#tuples-and-sequences")


def test_unrelated_question_scores_low(retriever):
    _, best_related = retriever.search("list comprehensions create lists", k=1)
    _, best_unrelated = retriever.search("kubernetes ingress controller yaml", k=1)
    assert best_unrelated < best_related
