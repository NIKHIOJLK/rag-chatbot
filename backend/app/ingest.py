"""Step 1 of the pipeline: download the docs and split them into chunks.

    python -m app.ingest            # download missing pages, write data/chunks.json
    python -m app.ingest --refresh  # download every page again

Two chunking strategies live here so the evaluation can compare them:
  * chunk_by_sections - v2: follows the page's own headings, keeps code examples whole,
    and keeps each chunk small enough for the embedding model to read all of it.
  * chunk_fixed       - v1 (original): 500-word windows that ignore the page structure.
"""
import argparse
import json
import re
import time
from dataclasses import dataclass, field

import requests
from bs4 import BeautifulSoup, Tag

from .config import DATA_DIR, RAW_DIR

BASE_URL = "https://docs.python.org/3/"

# The whole official tutorial plus the built-in functions and exceptions reference.
DOC_PAGES = [
    "tutorial/appetite.html",
    "tutorial/interpreter.html",
    "tutorial/introduction.html",
    "tutorial/controlflow.html",
    "tutorial/datastructures.html",
    "tutorial/modules.html",
    "tutorial/inputoutput.html",
    "tutorial/errors.html",
    "tutorial/classes.html",
    "tutorial/stdlib.html",
    "tutorial/stdlib2.html",
    "tutorial/venv.html",
    "tutorial/whatnow.html",
    "tutorial/floatingpoint.html",
    "library/functions.html",
    "library/exceptions.html",
]

# all-MiniLM-L6-v2 reads at most 256 tokens (roughly 180 words); anything longer is silently
# cut off. Staying under 150 words means the embedding sees the whole chunk.
MAX_WORDS = 150


@dataclass
class Section:
    page_title: str
    heading: str
    url: str  # page URL + #anchor, so a citation opens the exact section
    blocks: list[str] = field(default_factory=list)  # paragraphs, lists, code blocks


# ---------------------------------------------------------------- download

def raw_path(page: str):
    return RAW_DIR / page.replace("/", "__")


def download(pages: list[str], refresh: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for page in pages:
        path = raw_path(page)
        if path.exists() and not refresh:
            continue
        print(f"Downloading {page}")
        res = requests.get(BASE_URL + page, headers={"User-Agent": "rag-chatbot/2.0 (learning project)"}, timeout=30)
        res.raise_for_status()
        path.write_text(res.text, encoding="utf-8")
        time.sleep(0.5)  # be polite to docs.python.org


def load_pages(pages: list[str] = DOC_PAGES) -> list[tuple[str, str]]:
    """(url, html) for every downloaded page."""
    return [(BASE_URL + p, raw_path(p).read_text(encoding="utf-8")) for p in pages if raw_path(p).exists()]


# ---------------------------------------------------------------- parsing

def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _is_section(tag) -> bool:
    if not isinstance(tag, Tag):
        return False
    return tag.name == "section" or (tag.name == "div" and "section" in (tag.get("class") or []))


def _is_code(tag: Tag) -> bool:
    classes = " ".join(tag.get("class") or [])
    return tag.name == "pre" or "highlight" in classes


def _heading_text(heading: Tag) -> str:
    for link in heading.select("a.headerlink"):
        link.decompose()  # the ¶ permalink symbol
    return _clean(heading.get_text(" "))


def extract_sections(html: str, url: str) -> list[Section]:
    """Walk the page's <section> tree. Each section keeps only its own content (not its children's)."""
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("div", class_="body") or soup.body
    if body is None:
        return []
    h1 = body.find("h1")
    page_title = _heading_text(h1) if h1 else url

    sections: list[Section] = []

    def walk(node: Tag) -> None:
        heading = node.find(re.compile(r"^h[1-6]$"), recursive=False)
        title = _heading_text(heading) if heading else page_title
        anchor = node.get("id")
        section = Section(page_title, title, f"{url}#{anchor}" if anchor else url)
        for child in node.children:
            if not isinstance(child, Tag) or child is heading:
                continue
            if _is_section(child):
                walk(child)
            elif _is_code(child):
                pre = child.find("pre") or child
                section.blocks.append("```\n" + pre.get_text().rstrip() + "\n```")
            else:
                text = _clean(child.get_text(" "))
                if text:
                    section.blocks.append(text)
        if section.blocks:
            sections.append(section)

    top = [c for c in body.children if _is_section(c)]
    for node in top or [body]:
        walk(node)
    return sections


# ---------------------------------------------------------------- chunking

def _words(text: str) -> int:
    return len(text.split())


def _split_long(block: str, max_words: int, overlap: int = 20) -> list[str]:
    words = block.split()
    step = max_words - overlap
    return [" ".join(words[i : i + max_words]) for i in range(0, len(words), step)]


def chunk_by_sections(pages: list[tuple[str, str]], max_words: int = MAX_WORDS) -> list[dict]:
    """v2 chunking: group a section's blocks into chunks of at most max_words.

    Every chunk starts with "Page title > Section heading" so a chunk still makes sense on
    its own (a chunk saying "it returns a new list" is useless without knowing it's about sorted()).
    """
    chunks: list[dict] = []
    for url, html in pages:
        for section in extract_sections(html, url):
            header = f"{section.page_title} > {section.heading}" if section.heading != section.page_title else section.page_title
            budget = max_words - _words(header)
            pieces: list[str] = []
            current: list[str] = []
            for block in section.blocks:
                parts = [block] if _words(block) <= budget else _split_long(block, budget)
                for part in parts:
                    if current and _words(" ".join(current)) + _words(part) > budget:
                        pieces.append("\n\n".join(current))
                        current = []
                    current.append(part)
            if current:
                pieces.append("\n\n".join(current))
            for n, piece in enumerate(pieces):
                chunks.append({
                    "id": f"{section.url}::{n}",
                    "text": f"{header}\n\n{piece}",
                    "source": section.url,
                    "title": section.heading,
                    "page": section.page_title,
                })
    return chunks


def chunk_fixed(pages: list[tuple[str, str]], size: int = 500, overlap: int = 50) -> list[dict]:
    """v1 chunking (the original version), kept so the evaluation can measure the difference."""
    chunks: list[dict] = []
    for url, html in pages:
        soup = BeautifulSoup(html, "html.parser")
        body = soup.find("div", class_="body") or soup.body
        words = body.get_text("\n").split() if body else []
        for n, start in enumerate(range(0, len(words), size - overlap)):
            chunks.append({
                "id": f"{url}::fixed{n}",
                "text": " ".join(words[start : start + size]),
                "source": url,
                "title": url.rsplit("/", 1)[-1],
                "page": url,
            })
    return chunks


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the docs and write data/chunks.json")
    parser.add_argument("--refresh", action="store_true", help="download every page again")
    args = parser.parse_args()

    download(DOC_PAGES, refresh=args.refresh)
    pages = load_pages()
    chunks = chunk_by_sections(pages)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / "chunks.json").write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    sizes = sorted(_words(c["text"]) for c in chunks)
    print(f"\n{len(pages)} pages -> {len(chunks)} chunks (median {sizes[len(sizes) // 2]} words, max {sizes[-1]})")
    print("Saved data/chunks.json - open it to see what the model will search.")


if __name__ == "__main__":
    main()
