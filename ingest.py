import requests
from bs4 import BeautifulSoup
import json
import time

# List of Python doc pages we want to scrape.
# Start small so we can test the pipeline before scaling up.
DOC_URLS = [
    "https://docs.python.org/3/tutorial/introduction.html",
    "https://docs.python.org/3/tutorial/controlflow.html",
    "https://docs.python.org/3/tutorial/datastructures.html",
    "https://docs.python.org/3/library/functions.html",
    "https://docs.python.org/3/tutorial/errors.html",
]

def scrape_page(url):
    """Download a page and extract clean readable text from it."""
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    response.raise_for_status()  # crashes loudly if the request failed

    soup = BeautifulSoup(response.text, "html.parser")

    # Python docs put the main content inside a <div> with class "body"
    content_div = soup.find("div", class_="body")
    if not content_div:
        return ""

    # Get all paragraph and code text, joined with newlines
    text = content_div.get_text(separator="\n")
    return text

def chunk_text(text, source_url, chunk_size=500, overlap=50):
    """Split long text into overlapping chunks (~chunk_size words each)."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "text": chunk_text,
            "source": source_url
        })
        start += chunk_size - overlap  # overlap keeps context between chunks
    return chunks

def main():
    all_chunks = []

    for url in DOC_URLS:
        print(f"Scraping: {url}")
        text = scrape_page(url)
        if text:
            chunks = chunk_text(text, url)
            all_chunks.extend(chunks)
            print(f"  -> {len(chunks)} chunks")
        time.sleep(1)  # be polite to the server, don't hammer it with requests

    # Save all chunks to a JSON file so ingest and embedding are separate steps
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\nDone. Total chunks: {len(all_chunks)}")
    print("Saved to chunks.json")

if __name__ == "__main__":
    main()