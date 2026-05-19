"""Probe: does the model copy a doc_id because of TEXT CONTENT or just the IDENTIFIER?

Experiment
----------
We construct a *context_dependent*-style prompt where one of the documents has
doc_id = "rq kmeans".  Since this identifier is absent from the training corpus
the model has zero parametric memory of it — the only way it can emit
[COPY] rq kmeans is by reading the identifier from the context window.

Two variants of the prompt are tested:
  real_text   – the document contains a coherent passage about k-means clustering
  random_text – the same document slot uses ~20 random English words as content

If the model copies the identifier in BOTH variants it learned to copy labels
from context regardless of text semantics.
If it only copies with real_text it needs a semantic match between query and text.

Usage
-----
python -m src.probe_doc_text_vs_id \
    --model_name Abner0803/Qwen3-1.7B-icl-3shot-v4_128k-copy_tag \
    --output_dir ./probe_results

Run for both SFT and DPO models and compare.
"""

import argparse
import json
import os
import random
import re
import string

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LogitsProcessor,
    LogitsProcessorList,
)

# ---------------------------------------------------------------------------
# Target doc ID under study
# ---------------------------------------------------------------------------

TARGET_DOC_ID = "rq kmeans"

# ---------------------------------------------------------------------------
# Probe document content variants
# ---------------------------------------------------------------------------

REAL_TEXT = (
    "K-means is an unsupervised machine learning algorithm that partitions "
    "data points into K clusters. Each point is assigned to the cluster whose "
    "centroid is closest in Euclidean distance. The centroids are then "
    "recomputed as the mean of all assigned points and the process repeats "
    "until assignments no longer change. RQ (residual quantization) combined "
    "with K-means is often used for approximate nearest-neighbour search in "
    "high-dimensional spaces."
)

RANDOM_WORDS = [
    "pelican", "granite", "trumpet", "vortex", "cobbler", "saffron", "wrench",
    "balloon", "lantern", "frosting", "cascade", "tundra", "urchin", "blossom",
    "crevice", "pebble", "thunder", "lattice", "marrow", "cobalt",
]

def make_random_text(seed: int = 42) -> str:
    rng = random.Random(seed)
    words = rng.sample(RANDOM_WORDS * 3, 20)
    return " ".join(words) + "."

# ---------------------------------------------------------------------------
# Noise documents (from real MS MARCO entries)
# ---------------------------------------------------------------------------

NOISE_DOCS = [
    {
        "text": (
            "To avoid problems with nitrogen burn in the future, use only "
            "slow-release fertilizers, spread the fertilizer uniformly over "
            "the lawn, do not spill or overlap the fertilizer and apply no "
            "more than 1 1/2 pounds of nitrogen per 1000 square feet."
        ),
        "doc_id": "how to save a nitrogen burned lawn",
    },
    {
        "text": (
            "Camden County is a county located in the southeastern corner of "
            "the U.S. state of Georgia. According to the 2010 Census, the "
            "population was 50,513. Its county seat is Woodbine, and the "
            "largest city is St. Marys."
        ),
        "doc_id": "camden county, georgia",
    },
]

# ---------------------------------------------------------------------------
# ICL examples (taken verbatim from real training prompts, no [COPY] so that
# the examples themselves don't bias the copy-token probability)
# ---------------------------------------------------------------------------

ICL_EXAMPLES = [
    ("when was introduced the bankruptcy law",
     "bankruptcy abuse prevention and consumer protection act"),
    ("how much does it normally cost for a focus group",
     "faq's about focus groups"),
    ("do hoa restrict flag poles",
     "can your hoa prohibit you from flying your favorite flag?"),
]

# ---------------------------------------------------------------------------
# Target query variants
# ---------------------------------------------------------------------------

QUERIES = {
    "semantic_match": "how does rq kmeans work for approximate nearest neighbour search",
    "keyword_overlap": "k-means clustering algorithm centroid assignment",
    "unrelated":       "what is the capital city of france",
}

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(doc_text: str, query: str, pos_position: int = 1) -> str:
    """Build a v4-style ICL prompt.

    pos_position: which Document slot (1-indexed) holds the target doc.
    """
    docs = list(NOISE_DOCS)
    target_doc = {"text": doc_text, "doc_id": TARGET_DOC_ID}
    # Insert target at desired position
    docs.insert(pos_position - 1, target_doc)

    lines = ["## Documents"]
    for i, d in enumerate(docs, 1):
        lines.append(f"Document {i}")
        lines.append(f"Text: {d['text']}")
        lines.append(f"Identifier: {d['doc_id']}")
        lines.append("")

    lines.append("")
    lines.append("## Task")
    for j, (q, a) in enumerate(ICL_EXAMPLES, 1):
        lines.append(f"Example {j}:")
        lines.append(f"Query: {q}")
        lines.append(f"Answer: {a}")
        lines.append("")

    lines.append(f"Query: {query}")
    lines.append("Answer:")

    user_content = "\n".join(lines)
    return (
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

# ---------------------------------------------------------------------------
# Trie for constrained decoding  (includes target + noise doc IDs)
# ---------------------------------------------------------------------------

class TrieNode:
    def __init__(self):
        self.children: dict = {}


def build_local_trie(tokenizer) -> TrieNode:
    root = TrieNode()
    known_ids = (
        [TARGET_DOC_ID, f"[COPY] {TARGET_DOC_ID}"]
        + [d["doc_id"] for d in NOISE_DOCS]
        + [f"[COPY] {d['doc_id']}" for d in NOISE_DOCS]
        + [a for _, a in ICL_EXAMPLES]
    )
    for s in known_ids:
        toks = tokenizer.encode(s, add_special_tokens=False)
        node = root
        for t in toks:
            if t not in node.children:
                node.children[t] = TrieNode()
            node = node.children[t]
    return root


class TrieLogitsProcessor(LogitsProcessor):
    def __init__(self, root: TrieNode, prompt_len: int, eos_id: int,
                 num_beams: int):
        self.root = root
        self.prompt_len = prompt_len
        self.eos_id = eos_id
        self.num_beams = num_beams

    def _valid(self, generated):
        node = self.root
        for t in generated:
            if t in node.children:
                node = node.children[t]
            else:
                return [self.eos_id]
        valid = list(node.children.keys())
        return valid if valid else [self.eos_id]

    def __call__(self, input_ids, scores):
        for beam_idx in range(input_ids.shape[0]):
            gen = input_ids[beam_idx][self.prompt_len:].tolist()
            valid = self._valid(gen)
            new_scores = torch.full_like(scores[beam_idx], float("-inf"))
            idx = torch.tensor(valid, device=scores.device, dtype=torch.long)
            new_scores[idx] = scores[beam_idx][idx]
            scores[beam_idx] = new_scores
        return scores

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def clean(text: str) -> str:
    text = text.strip()
    for tok in ("</s>", "<|endoftext|>", "<|im_end|>", "<|im_start|>"):
        text = text.replace(tok, "")
    if "<think>" in text:
        m = re.search(r"</think>\s*(.*?)(?:<\|im_end\|>|</s>|$)", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
    if text.startswith("[COPY]") and not text.startswith("[COPY] "):
        text = "[COPY] " + text[6:]
    return text.strip()


@torch.no_grad()
def run_probe(model, tokenizer, trie_root, gen_config, prompt: str):
    inputs = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    prompt_len = inputs["input_ids"].shape[1]

    logits_proc = TrieLogitsProcessor(
        trie_root, prompt_len, tokenizer.eos_token_id, gen_config.num_beams
    )
    outputs = model.generate(
        **inputs,
        generation_config=gen_config,
        logits_processor=LogitsProcessorList([logits_proc]),
    )
    generated = outputs[:, prompt_len:]
    decoded = tokenizer.batch_decode(generated, skip_special_tokens=False)
    return [clean(d) for d in decoded]


@torch.no_grad()
def get_copy_logprob(model, tokenizer, prompt: str, copy_token_id: int) -> float:
    """Forward pass only: P([COPY]) at the first output position."""
    inputs = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False
    ).to(model.device)
    out = model(**inputs)
    # last token logits → first output position
    logits = out.logits[0, -1, :]
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return log_probs[copy_token_id].item()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Abner0803/Qwen3-1.7B-icl-3shot-v4_128k-copy_tag")
    p.add_argument("--num_beams", type=int, default=5)
    p.add_argument("--num_return", type=int, default=5)
    p.add_argument("--max_new_tokens", type=int, default=30)
    p.add_argument("--output_dir", default="./probe_results")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    print(f"\nLoading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, padding_side="left", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    copy_token_id = tokenizer.encode("[COPY]", add_special_tokens=False)[-1]
    print(f"[COPY] token id: {copy_token_id}")

    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
        device_map={"": device} if device.type == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()

    trie_root = build_local_trie(tokenizer)
    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_beams=args.num_beams,
        num_return_sequences=args.num_return,
    )

    random_text = make_random_text()
    doc_variants = {
        "real_text":   REAL_TEXT,
        "random_text": random_text,
    }

    results = {}

    for query_type, query in QUERIES.items():
        results[query_type] = {}
        print(f"\n{'='*60}")
        print(f"Query type : {query_type}")
        print(f"Query      : {query}")

        for doc_variant, doc_text in doc_variants.items():
            prompt = build_prompt(doc_text, query)
            preds = run_probe(model, tokenizer, trie_root, gen_config, prompt)
            copy_lp = get_copy_logprob(model, tokenizer, prompt, copy_token_id)
            copy_prob = round(float(torch.exp(torch.tensor(copy_lp)).item()), 4)

            top1 = preds[0] if preds else ""
            copied_correctly = top1 == f"[COPY] {TARGET_DOC_ID}"

            print(f"\n  doc_variant : {doc_variant}")
            print(f"  doc_text    : {doc_text[:80]}...")
            print(f"  P([COPY])   : {copy_prob:.4f}  (log {copy_lp:.3f})")
            print(f"  top-1 pred  : {top1!r}")
            print(f"  [COPY] correct : {'YES ✓' if copied_correctly else 'NO  ✗'}")
            for i, p in enumerate(preds, 1):
                print(f"    beam {i:2d}: {p!r}")

            results[query_type][doc_variant] = {
                "query": query,
                "doc_text_snippet": doc_text[:120],
                "copy_prob": copy_prob,
                "predictions": preds,
                "copied_correctly": copied_correctly,
            }

    # Summary table
    print(f"\n\n{'='*70}")
    print(f"SUMMARY  —  doc_id under study: '{TARGET_DOC_ID}'")
    print(f"{'='*70}")
    print(f"{'Query type':<25} {'Variant':<15} {'P([COPY])':>10} {'[COPY] correct?':>16}")
    print("-" * 70)
    for qt, variants in results.items():
        for dv, r in variants.items():
            print(
                f"  {qt:<23} {dv:<15} {r['copy_prob']:>10.4f} "
                f"{'YES' if r['copied_correctly'] else 'NO':>16}"
            )

    print(f"\nDiagnosis:")
    for qt, variants in results.items():
        real_ok   = variants["real_text"]["copied_correctly"]
        random_ok = variants["random_text"]["copied_correctly"]
        if real_ok and random_ok:
            verdict = "copies identifier regardless of text (identifier-dependent)"
        elif real_ok and not random_ok:
            verdict = "copies only with semantic text match (text-dependent)"
        elif not real_ok and not random_ok:
            verdict = "fails to copy in both cases"
        else:
            verdict = "copies random but not real text (unexpected)"
        print(f"  [{qt}] {verdict}")

    out_path = os.path.join(
        args.output_dir,
        f"probe_{args.model_name.split('/')[-1]}.json"
    )
    with open(out_path, "w") as f:
        json.dump({"model": args.model_name, "target_doc_id": TARGET_DOC_ID,
                   "results": results}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
