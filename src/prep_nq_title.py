"""Prepare NQ data with Wikipedia page title as doc_id.

Input : data/NQ/train.jsonl   (160k passages, text = "document: {title} - wikipedia {passage}")
        data/NQ/test.jsonl    (20k queries,   doc_id = structured tokens)
        data/NQ/icl_test.jsonl
        HuggingFace natural_questions train split (real NQ training queries)

Output: data/nq_text_title/train.jsonl    -- unique docs (indexing) + real NQ queries (query)
        data/nq_text_title/test.jsonl     -- queries with title doc_ids
        data/nq_text_title/icl_test.jsonl -- queries with title doc_ids
"""

import argparse
import json
from pathlib import Path


_PREFIX = "document: "
_WIKI_SEP = " - wikipedia "  # lowercase for matching


def extract_title_and_passage(text: str):
    """Return (lowercase_title, clean_passage) from NQ document text.

    Handles both ' - wikipedia ' and ' - Wikipedia ' variants.
    """
    if not text.startswith(_PREFIX):
        return None, text
    rest = text[len(_PREFIX):]
    rest_lower = rest.lower()
    idx = rest_lower.find(_WIKI_SEP)
    if idx == -1:
        return None, text
    title = rest[:idx].strip().lower()
    passage = rest[idx + len(_WIKI_SEP):].strip()
    return title, passage


def build_token_to_title_map(train_path: Path):
    """Build {structured_docid → title} from train.jsonl.

    Test docids use 3 tokens; train docids may have 3 or 4 tokens.
    We index by both the full token string and the 3-token prefix so
    test lookups always find a match.
    """
    token_to_title: dict[str, str] = {}
    with open(train_path) as f:
        for line in f:
            d = json.loads(line)
            title, _ = extract_title_and_passage(d["text"])
            if title is None:
                continue
            token = d["doc_id"].strip()
            token_to_title[token] = title
            # also index by 3-token prefix
            parts = token.split()
            if len(parts) > 3:
                prefix = " ".join(parts[:3])
                if prefix not in token_to_title:
                    token_to_title[prefix] = title
    return token_to_title


def _scan_doc_tokens(path: Path) -> dict:
    """Pre-scan a file for document entries and return {token → title}.

    Used so that query entries in the same file can be resolved against
    documents that exist only in that file (e.g. icl_test queries → icl_test docs).
    """
    local_map: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if not d["text"].startswith("document: "):
                continue
            title, _ = extract_title_and_passage(d["text"])
            if title is None:
                continue
            token = d["doc_id"].strip()
            local_map[token] = title
            parts = token.split()
            if len(parts) > 3:
                local_map.setdefault(" ".join(parts[:3]), title)
    return local_map


def convert_file(src: Path, dst: Path, token_to_title: dict):
    """Convert a query or document file to title-based docids.

    - query entries (text starts with "question: "): map structured docid → title
      (checks local doc map first, then global train map)
    - document entries (text starts with "document: "): extract title from text directly
    """
    # Pre-scan for local document→token mappings (needed when queries and
    # their target documents are in the same file, e.g. icl_test).
    local_map = _scan_doc_tokens(src)
    combined = {**token_to_title, **local_map}

    missing = 0
    total = 0
    with open(src) as fin, open(dst, "w") as fout:
        for line in fin:
            d = json.loads(line)
            text = d["text"]

            if text.startswith("document: "):
                # document entry — extract title directly from text
                title, passage = extract_title_and_passage(text)
                if title is None:
                    missing += 1
                    continue
                out = {
                    "text": passage,
                    "doc_id": title,
                    "operation": "indexing",
                }
            else:
                # query entry — map structured docid → title
                token = d["doc_id"].strip()
                title = combined.get(token)
                if title is None:
                    parts = token.split()
                    title = combined.get(" ".join(parts[:3]))
                if title is None:
                    missing += 1
                    continue
                query_text = text
                if query_text.startswith("question: "):
                    query_text = query_text[len("question: "):].strip()
                out = {
                    "text": query_text,
                    "doc_id": title,
                    "operation": "query",
                }

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            total += 1
    print(f"  {src.name}: {total} written, {missing} skipped")


def load_hf_nq_queries(known_titles: set[str]) -> list[dict]:
    """Load real NQ training queries from HuggingFace and filter to known corpus titles.

    Each NQ example has a question and a Wikipedia document title. We use the
    title (lowercased) as the doc_id and only keep examples whose title exists
    in our passage corpus so the model isn't trained to recall unseen titles.
    """
    import os
    from datasets import load_dataset

    # HF XET storage backend can return 500; streaming + disable XET avoids it
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    print("Loading NQ training queries from HuggingFace (natural_questions) ...")
    ds = load_dataset("google-research-datasets/natural_questions", split="train", streaming=True)

    queries = []
    skipped = 0
    for example in ds:
        title = example["document"]["title"].strip().lower()
        if title not in known_titles:
            skipped += 1
            continue
        query_text = example["question"]["text"].strip()
        queries.append({
            "text": query_text,
            "doc_id": title,
            "operation": "query",
        })

    print(f"  {len(queries)} queries kept, {skipped} skipped (title not in corpus)")
    return queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nq_dir", default="data/NQ")
    ap.add_argument("--out_dir", default="data/nq_text_title")
    ap.add_argument("--skip_hf_queries", action="store_true",
                    help="Skip loading real NQ queries from HuggingFace")
    args = ap.parse_args()

    nq_dir = Path(args.nq_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Building token→title map from train.jsonl ...")
    token_to_title = build_token_to_title_map(nq_dir / "train.jsonl")
    print(f"  {len(token_to_title)} token entries in map")

    print("Writing train.jsonl (unique docs + real NQ queries) ...")
    seen_titles: set[str] = set()
    n_docs = 0
    indexing_entries = []
    with open(nq_dir / "train.jsonl") as fin:
        for line in fin:
            d = json.loads(line)
            title, passage = extract_title_and_passage(d["text"])
            if title is None or title in seen_titles:
                continue
            seen_titles.add(title)
            indexing_entries.append({
                "text": passage,
                "doc_id": title,
                "operation": "indexing",
            })
            n_docs += 1
    print(f"  {n_docs} unique documents")

    hf_queries = []
    if not args.skip_hf_queries:
        hf_queries = load_hf_nq_queries(seen_titles)

    with open(out_dir / "train.jsonl", "w") as fout:
        for entry in indexing_entries:
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
        for entry in hf_queries:
            fout.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"  train.jsonl: {n_docs} indexing + {len(hf_queries)} queries written")

    for split in ("test", "icl_test"):
        src = nq_dir / f"{split}.jsonl"
        dst = out_dir / f"{split}.jsonl"
        if not src.exists():
            print(f"  {split}.jsonl not found, skipping")
            continue
        print(f"Converting {split}.jsonl ...")
        convert_file(src, dst, token_to_title)

    print("Done. Output →", out_dir)


if __name__ == "__main__":
    main()
