from app.ingest import MAX_WORDS, chunk_by_sections, chunk_fixed, extract_sections
from conftest import SAMPLE_HTML, SAMPLE_URL


def test_sections_follow_headings_and_keep_anchors():
    sections = extract_sections(SAMPLE_HTML, SAMPLE_URL)
    by_heading = {s.heading: s for s in sections}
    assert "5.1.1. Using Lists as Stacks" in by_heading
    assert by_heading["5.1.1. Using Lists as Stacks"].url == SAMPLE_URL + "#using-lists-as-stacks"
    assert all("¶" not in s.heading for s in sections)  # permalink symbols are removed


def test_a_section_does_not_swallow_its_subsections():
    lists = next(s for s in extract_sections(SAMPLE_HTML, SAMPLE_URL) if s.heading == "5.1. More on Lists")
    assert not any("stack" in b for b in lists.blocks)


def test_code_blocks_stay_whole_and_fenced(chunks):
    code = next(c for c in chunks if "fruits.append('grape')" in c["text"])
    assert "```\n>>> fruits = ['orange', 'apple']\n>>> fruits.append('grape')" in code["text"]


def test_chunks_start_with_page_and_section_context(chunks):
    stack = next(c for c in chunks if c["source"].endswith("#using-lists-as-stacks"))
    assert stack["text"].startswith("5. Data Structures > 5.1.1. Using Lists as Stacks")


def test_every_chunk_fits_the_embedding_model(chunks):
    assert max(len(c["text"].split()) for c in chunks) <= MAX_WORDS


def test_long_sections_are_split_with_unique_ids(chunks):
    comprehension = [c for c in chunks if c["source"].endswith("#list-comprehensions")]
    assert len(comprehension) >= 2
    assert len({c["id"] for c in chunks}) == len(chunks)


def test_fixed_chunking_is_the_old_500_word_windows():
    html = "<div class='body'>" + " ".join(f"w{i}" for i in range(1200)) + "</div>"
    chunks = chunk_fixed([(SAMPLE_URL, html)])
    assert [len(c["text"].split()) for c in chunks] == [500, 500, 300]
