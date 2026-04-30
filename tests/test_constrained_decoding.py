"""
Tests for TrieConstrainedLogitsProcessorWithTag.

Verifies three routing behaviours without loading a real model:
  1. Step 0 (nothing generated yet): both [COPY] and global-trie root tokens are valid.
  2. After [COPY]: only context-trie tokens are valid.
  3. After a non-[COPY] first token: only global-trie tokens are valid.

Also includes a data-driven integration test that reads
data/msmarco-icl-3shot-v4/test.jsonl and checks routing for a handful of
context_dependent and all_noise samples.
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.inference_utils import TrieConstrainedLogitsProcessorWithTag, TrieNode


# ---------------------------------------------------------------------------
# Minimal word-level tokenizer (no model needed)
# ---------------------------------------------------------------------------

class FakeTokenizer:
    """
    Deterministic word-level tokenizer.

    Reserved IDs:
      0 -> [PAD]
      1 -> [EOS]
      2 -> [COPY]
    All other words receive sequential IDs starting at 3.
    """

    PAD_ID = 0
    EOS_ID = 1
    COPY_ID = 2

    def __init__(self) -> None:
        self._vocab: Dict[str, int] = {
            "[PAD]": self.PAD_ID,
            "[EOS]": self.EOS_ID,
            "[COPY]": self.COPY_ID,
        }
        self._next = 3

    def encode(self, text: str) -> List[int]:
        ids = []
        for word in text.strip().split():
            if word not in self._vocab:
                self._vocab[word] = self._next
                self._next += 1
            ids.append(self._vocab[word])
        return ids

    @property
    def vocab_size(self) -> int:
        return self._next


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_trie(doc_ids: List[str], tok: FakeTokenizer) -> TrieNode:
    root = TrieNode()
    for doc_id in doc_ids:
        node = root
        for tid in tok.encode(doc_id):
            node = node.children.setdefault(tid, TrieNode())
        node.end_of_docid = True
    return root


def _valid_tokens(scores: torch.Tensor) -> set:
    return set((scores > float("-inf")).nonzero(as_tuple=True)[0].tolist())


def _make_input_ids(prompt_len: int, generated: List[int]) -> torch.LongTensor:
    """Single-beam input_ids tensor: [prompt zeros] + generated."""
    full = [0] * prompt_len + generated
    return torch.tensor([full], dtype=torch.long)


def _make_scores(vocab_size: int) -> torch.FloatTensor:
    return torch.zeros(1, vocab_size)


def _make_processor(
    global_trie: TrieNode,
    context_ids_per_sample: List[List[List[int]]],
    tok: FakeTokenizer,
    prompt_length: int = 4,
    num_beams: int = 1,
) -> TrieConstrainedLogitsProcessorWithTag:
    return TrieConstrainedLogitsProcessorWithTag(
        trie_root=global_trie,
        prompt_length=prompt_length,
        eos_token_id=tok.EOS_ID,
        copy_token_id=tok.COPY_ID,
        num_beams=num_beams,
        context_ids=context_ids_per_sample,
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestStep0:
    """At step 0, both [COPY] and all global-trie root tokens must be valid."""

    def test_copy_token_included(self) -> None:
        tok = FakeTokenizer()
        global_trie = _build_trie(["alpha beta", "gamma delta"], tok)
        proc = _make_processor(global_trie, [[tok.encode("alpha beta")]], tok)

        scores = proc(_make_input_ids(4, []), _make_scores(tok.vocab_size))
        valid = _valid_tokens(scores[0])
        assert tok.COPY_ID in valid

    def test_global_trie_roots_included(self) -> None:
        tok = FakeTokenizer()
        global_trie = _build_trie(["alpha beta", "gamma delta"], tok)
        proc = _make_processor(global_trie, [[tok.encode("alpha beta")]], tok)

        scores = proc(_make_input_ids(4, []), _make_scores(tok.vocab_size))
        valid = _valid_tokens(scores[0])

        expected_roots = set(global_trie.children.keys()) | {tok.COPY_ID}
        assert valid == expected_roots

    def test_no_other_tokens_valid(self) -> None:
        tok = FakeTokenizer()
        global_trie = _build_trie(["foo bar"], tok)
        # encode a word that's NOT in the trie
        random_tok = tok.encode("zzz")[0]
        proc = _make_processor(global_trie, [[tok.encode("foo bar")]], tok)

        scores = proc(_make_input_ids(4, []), _make_scores(tok.vocab_size))
        valid = _valid_tokens(scores[0])

        assert random_tok not in valid


class TestCopyPath:
    """After [COPY], the processor must use the context trie exclusively."""

    def test_context_roots_are_valid(self) -> None:
        tok = FakeTokenizer()
        global_trie = _build_trie(["apple orange", "mango papaya"], tok)
        context_doc_ids = [tok.encode("mango papaya")]
        proc = _make_processor(global_trie, [context_doc_ids], tok)

        scores = proc(
            _make_input_ids(4, [tok.COPY_ID]),
            _make_scores(tok.vocab_size),
        )
        valid = _valid_tokens(scores[0])
        context_roots = {context_doc_ids[0][0]}  # first token of "mango papaya"
        assert context_roots <= valid

    def test_global_only_tokens_are_masked(self) -> None:
        tok = FakeTokenizer()
        # "apple" is in global but NOT in context
        global_trie = _build_trie(["apple orange", "mango papaya"], tok)
        apple_id = tok.encode("apple")[0]
        context_doc_ids = [tok.encode("mango papaya")]
        proc = _make_processor(global_trie, [context_doc_ids], tok)

        scores = proc(
            _make_input_ids(4, [tok.COPY_ID]),
            _make_scores(tok.vocab_size),
        )
        valid = _valid_tokens(scores[0])
        assert apple_id not in valid

    def test_context_second_token(self) -> None:
        """After [COPY] + first context token, next step still follows context trie."""
        tok = FakeTokenizer()
        global_trie = _build_trie(["mango papaya"], tok)
        mango_id, papaya_id = tok.encode("mango papaya")
        context_doc_ids = [tok.encode("mango papaya")]
        proc = _make_processor(global_trie, [context_doc_ids], tok)

        # Generated so far: [COPY] mango
        scores = proc(
            _make_input_ids(4, [tok.COPY_ID, mango_id]),
            _make_scores(tok.vocab_size),
        )
        valid = _valid_tokens(scores[0])
        # "papaya" (end of context doc) -> next valid is eos
        assert papaya_id in valid or tok.EOS_ID in valid

    def test_eos_when_context_doc_exhausted(self) -> None:
        """After fully traversing a context doc, EOS must be the only valid token."""
        tok = FakeTokenizer()
        doc_tokens = tok.encode("mango papaya")
        global_trie = _build_trie(["mango papaya"], tok)
        proc = _make_processor(global_trie, [[doc_tokens]], tok)

        # Generated: [COPY] mango papaya  (full doc consumed)
        scores = proc(
            _make_input_ids(4, [tok.COPY_ID] + doc_tokens),
            _make_scores(tok.vocab_size),
        )
        valid = _valid_tokens(scores[0])
        assert valid == {tok.EOS_ID}


class TestNonCopyPath:
    """After a non-[COPY] first token the processor must fall back to global trie."""

    def test_copy_token_not_valid_after_global_first_token(self) -> None:
        tok = FakeTokenizer()
        global_trie = _build_trie(["foo bar"], tok)
        foo_id = tok.encode("foo")[0]
        proc = _make_processor(global_trie, [[tok.encode("foo bar")]], tok)

        scores = proc(
            _make_input_ids(4, [foo_id]),
            _make_scores(tok.vocab_size),
        )
        valid = _valid_tokens(scores[0])
        assert tok.COPY_ID not in valid

    def test_global_trie_second_token_valid(self) -> None:
        tok = FakeTokenizer()
        global_trie = _build_trie(["foo bar"], tok)
        foo_id, bar_id = tok.encode("foo bar")
        proc = _make_processor(global_trie, [[tok.encode("foo bar")]], tok)

        scores = proc(
            _make_input_ids(4, [foo_id]),
            _make_scores(tok.vocab_size),
        )
        valid = _valid_tokens(scores[0])
        assert bar_id in valid

    def test_invalid_global_path_yields_eos(self) -> None:
        """A generated token that doesn't exist in the global trie -> EOS only."""
        tok = FakeTokenizer()
        global_trie = _build_trie(["foo bar"], tok)
        rogue_id = tok.encode("zzz")[0]  # not in trie
        proc = _make_processor(global_trie, [[tok.encode("foo bar")]], tok)

        scores = proc(
            _make_input_ids(4, [rogue_id]),
            _make_scores(tok.vocab_size),
        )
        valid = _valid_tokens(scores[0])
        assert valid == {tok.EOS_ID}


class TestMultiBeam:
    """With num_beams > 1, each sample's beams must use that sample's context trie."""

    def test_two_samples_two_beams_each(self) -> None:
        tok = FakeTokenizer()
        # Sample 0: context = "alpha beta"; Sample 1: context = "gamma delta"
        sample0_ctx = tok.encode("alpha beta")
        sample1_ctx = tok.encode("gamma delta")
        alpha_id = sample0_ctx[0]
        gamma_id = sample1_ctx[0]

        global_trie = _build_trie(["alpha beta", "gamma delta"], tok)
        processor = TrieConstrainedLogitsProcessorWithTag(
            trie_root=global_trie,
            prompt_length=4,
            eos_token_id=tok.EOS_ID,
            copy_token_id=tok.COPY_ID,
            num_beams=2,
            context_ids=[[sample0_ctx], [sample1_ctx]],
        )

        # batch_size = 4 (2 samples × 2 beams)
        input_ids = torch.zeros(4, 4, dtype=torch.long)
        # All beams at step 0 have nothing generated yet -> add [COPY] to each
        input_ids[0, :] = 0   # beam 0 of sample 0 -> empty
        input_ids[1, :] = 0   # beam 1 of sample 0 -> empty
        input_ids[2, :] = 0   # beam 0 of sample 1 -> empty
        input_ids[3, :] = 0   # beam 1 of sample 1 -> empty

        # Simulate step AFTER [COPY] for all beams
        generated = torch.tensor([tok.COPY_ID], dtype=torch.long)
        after_copy = torch.cat(
            [input_ids, generated.expand(4, 1)], dim=1
        )  # shape (4, 5)

        scores = torch.zeros(4, tok.vocab_size)
        out = processor(after_copy, scores.clone())

        # Beams 0 and 1 belong to sample 0 -> only alpha_id valid (context of sample 0)
        valid_s0_b0 = _valid_tokens(out[0])
        valid_s0_b1 = _valid_tokens(out[1])
        assert alpha_id in valid_s0_b0 and gamma_id not in valid_s0_b0
        assert alpha_id in valid_s0_b1 and gamma_id not in valid_s0_b1

        # Beams 2 and 3 belong to sample 1 -> only gamma_id valid
        valid_s1_b0 = _valid_tokens(out[2])
        valid_s1_b1 = _valid_tokens(out[3])
        assert gamma_id in valid_s1_b0 and alpha_id not in valid_s1_b0
        assert gamma_id in valid_s1_b1 and alpha_id not in valid_s1_b1


# ---------------------------------------------------------------------------
# Integration test: real ICL test data
# ---------------------------------------------------------------------------

DATA_PATH = Path(__file__).parent.parent / "data/msmarco-icl-3shot-v4/test.jsonl"
SAMPLES_PER_PATTERN = 10


def _extract_context_doc_ids(user_content: str) -> List[str]:
    return [m.strip() for m in re.findall(r"Identifier:\s*(.+)", user_content)]


def _load_icl_samples():
    with open(DATA_PATH) as f:
        data = [json.loads(l) for l in f]
    ctx_dep = [d for d in data if d["metadata"]["pattern"] == "context_dependent"]
    all_noise = [d for d in data if d["metadata"]["pattern"] == "all_noise"]
    return ctx_dep[:SAMPLES_PER_PATTERN], all_noise[:SAMPLES_PER_PATTERN]


class TestICLData:
    """Routing correctness verified against real ICL test samples."""

    @classmethod
    def setup_class(cls) -> None:
        ctx_dep_samples, all_noise_samples = _load_icl_samples()
        cls.ctx_dep = ctx_dep_samples
        cls.all_noise = all_noise_samples

        # Build vocabulary from all doc IDs in these samples
        cls.tok = FakeTokenizer()
        all_doc_ids: set = set()
        for item in ctx_dep_samples + all_noise_samples:
            user = item["conversations"][0]["content"]
            for d in _extract_context_doc_ids(user):
                all_doc_ids.add(d)
            answer = item["conversations"][1]["content"]
            # strip [COPY] prefix if present
            all_doc_ids.add(re.sub(r"^\[COPY\]\s*", "", answer).strip())

        # Encode all to build vocab
        for d in all_doc_ids:
            cls.tok.encode(d)

        cls.global_trie = _build_trie(list(all_doc_ids), cls.tok)

    def _make_proc_for_sample(self, item) -> TrieConstrainedLogitsProcessorWithTag:
        user = item["conversations"][0]["content"]
        ctx_ids = [self.tok.encode(d) for d in _extract_context_doc_ids(user)]
        return _make_processor(
            self.global_trie,
            [ctx_ids],
            self.tok,
            prompt_length=4,
            num_beams=1,
        )

    # -- context_dependent samples ------------------------------------------

    def test_context_dependent_step0_copy_valid(self) -> None:
        for item in self.ctx_dep:
            proc = self._make_proc_for_sample(item)
            out = proc(_make_input_ids(4, []), _make_scores(self.tok.vocab_size))
            valid = _valid_tokens(out[0])
            assert self.tok.COPY_ID in valid, (
                f"[COPY] must be valid at step 0 for context_dependent sample: "
                f"{item['metadata']}"
            )

    def test_context_dependent_after_copy_uses_context_trie(self) -> None:
        for item in self.ctx_dep:
            user = item["conversations"][0]["content"]
            ctx_doc_ids = _extract_context_doc_ids(user)
            ctx_ids = [self.tok.encode(d) for d in ctx_doc_ids]
            context_first_tokens = {ids[0] for ids in ctx_ids if ids}

            proc = self._make_proc_for_sample(item)
            out = proc(
                _make_input_ids(4, [self.tok.COPY_ID]),
                _make_scores(self.tok.vocab_size),
            )
            valid = _valid_tokens(out[0]) - {self.tok.EOS_ID}

            assert valid == context_first_tokens, (
                f"After [COPY], valid tokens {valid} != context first tokens "
                f"{context_first_tokens} for {item['metadata']}"
            )

    def test_context_dependent_global_tokens_masked_after_copy(self) -> None:
        for item in self.ctx_dep:
            user = item["conversations"][0]["content"]
            ctx_doc_ids = _extract_context_doc_ids(user)
            ctx_ids = [self.tok.encode(d) for d in ctx_doc_ids]
            context_first_tokens = {ids[0] for ids in ctx_ids if ids}
            global_only = set(self.global_trie.children.keys()) - context_first_tokens

            proc = self._make_proc_for_sample(item)
            out = proc(
                _make_input_ids(4, [self.tok.COPY_ID]),
                _make_scores(self.tok.vocab_size),
            )
            valid = _valid_tokens(out[0])

            for tok_id in global_only:
                assert tok_id not in valid, (
                    f"Token {tok_id} is global-only but was not masked after [COPY] "
                    f"for {item['metadata']}"
                )

    # -- all_noise samples --------------------------------------------------

    def test_all_noise_step0_copy_valid(self) -> None:
        for item in self.all_noise:
            proc = self._make_proc_for_sample(item)
            out = proc(_make_input_ids(4, []), _make_scores(self.tok.vocab_size))
            valid = _valid_tokens(out[0])
            assert self.tok.COPY_ID in valid, (
                f"[COPY] must be valid at step 0 even for all_noise sample: "
                f"{item['metadata']}"
            )

    def test_all_noise_non_copy_first_token_excludes_copy(self) -> None:
        for item in self.all_noise:
            answer = item["conversations"][1]["content"].strip()
            answer_tokens = self.tok.encode(answer)
            first_tok = answer_tokens[0]

            # Only test when the first answer token is actually in the global trie
            if first_tok not in self.global_trie.children:
                continue

            proc = self._make_proc_for_sample(item)
            out = proc(
                _make_input_ids(4, [first_tok]),
                _make_scores(self.tok.vocab_size),
            )
            valid = _valid_tokens(out[0])
            assert self.tok.COPY_ID not in valid, (
                f"[COPY] must NOT be valid after non-[COPY] first token "
                f"for all_noise sample {item['metadata']}"
            )

    def test_all_noise_second_token_follows_global_trie(self) -> None:
        for item in self.all_noise:
            answer = item["conversations"][1]["content"].strip()
            answer_tokens = self.tok.encode(answer)
            if len(answer_tokens) < 2:
                continue
            first_tok, second_tok = answer_tokens[0], answer_tokens[1]
            if first_tok not in self.global_trie.children:
                continue

            proc = self._make_proc_for_sample(item)
            out = proc(
                _make_input_ids(4, [first_tok]),
                _make_scores(self.tok.vocab_size),
            )
            valid = _valid_tokens(out[0])

            # The second answer token must be reachable in the global trie
            if second_tok in self.global_trie.children.get(first_tok, TrieNode()).children:
                assert second_tok in valid, (
                    f"Second token {second_tok} should be valid via global trie "
                    f"for all_noise sample {item['metadata']}"
                )


def _show_step0(label: str, tok: FakeTokenizer, proc, prompt_len: int = 4) -> None:
    out = proc(_make_input_ids(prompt_len, []), _make_scores(tok.vocab_size))
    valid = sorted(_valid_tokens(out[0]))
    inv_vocab = {v: k for k, v in tok._vocab.items()}
    readable = [inv_vocab.get(t, f"id={t}") for t in valid]
    print(f"    {label}")
    print(f"      generated : (empty)")
    print(f"      valid ids : {valid}")
    print(f"      valid toks: {readable}")


def _show_after(
    label: str,
    tok: FakeTokenizer,
    proc,
    generated: List[int],
    prompt_len: int = 4,
) -> None:
    out = proc(_make_input_ids(prompt_len, generated), _make_scores(tok.vocab_size))
    valid = sorted(_valid_tokens(out[0]))
    inv_vocab = {v: k for k, v in tok._vocab.items()}
    gen_readable = [inv_vocab.get(t, f"id={t}") for t in generated]
    readable = [inv_vocab.get(t, f"id={t}") for t in valid]
    print(f"    {label}")
    print(f"      generated : {gen_readable}  (ids={generated})")
    print(f"      valid ids : {valid}")
    print(f"      valid toks: {readable}")


if __name__ == "__main__":
    import traceback

    # -----------------------------------------------------------------------
    # Verbose output: show actual token IDs and valid sets for every test case
    # -----------------------------------------------------------------------

    print("=" * 70)
    print("TestStep0")
    print("=" * 70)

    print("\n  test_copy_token_included")
    tok = FakeTokenizer()
    global_trie = _build_trie(["alpha beta", "gamma delta"], tok)
    proc = _make_processor(global_trie, [[tok.encode("alpha beta")]], tok)
    _show_step0("[COPY] must be in valid set", tok, proc)

    print("\n  test_global_trie_roots_included")
    tok = FakeTokenizer()
    global_trie = _build_trie(["alpha beta", "gamma delta"], tok)
    proc = _make_processor(global_trie, [[tok.encode("alpha beta")]], tok)
    _show_step0("global trie roots + [COPY] == valid set", tok, proc)

    print("\n  test_no_other_tokens_valid")
    tok = FakeTokenizer()
    global_trie = _build_trie(["foo bar"], tok)
    _ = tok.encode("zzz")
    proc = _make_processor(global_trie, [[tok.encode("foo bar")]], tok)
    _show_step0("'zzz' must NOT appear in valid set", tok, proc)

    print("\n" + "=" * 70)
    print("TestCopyPath")
    print("=" * 70)

    print("\n  test_context_roots_are_valid")
    tok = FakeTokenizer()
    global_trie = _build_trie(["apple orange", "mango papaya"], tok)
    context_doc_ids = [tok.encode("mango papaya")]
    proc = _make_processor(global_trie, [context_doc_ids], tok)
    _show_after("after [COPY]: 'mango' must be valid", tok, proc, [tok.COPY_ID])

    print("\n  test_global_only_tokens_are_masked")
    tok = FakeTokenizer()
    global_trie = _build_trie(["apple orange", "mango papaya"], tok)
    context_doc_ids = [tok.encode("mango papaya")]
    proc = _make_processor(global_trie, [context_doc_ids], tok)
    _show_after("after [COPY]: 'apple' must be masked", tok, proc, [tok.COPY_ID])

    print("\n  test_context_second_token")
    tok = FakeTokenizer()
    global_trie = _build_trie(["mango papaya"], tok)
    mango_id, papaya_id = tok.encode("mango papaya")
    context_doc_ids = [tok.encode("mango papaya")]
    proc = _make_processor(global_trie, [context_doc_ids], tok)
    _show_after("after [COPY] mango: 'papaya' must be valid", tok, proc, [tok.COPY_ID, mango_id])

    print("\n  test_eos_when_context_doc_exhausted")
    tok = FakeTokenizer()
    doc_tokens = tok.encode("mango papaya")
    global_trie = _build_trie(["mango papaya"], tok)
    proc = _make_processor(global_trie, [[doc_tokens]], tok)
    _show_after("after [COPY] mango papaya: only EOS valid", tok, proc, [tok.COPY_ID] + doc_tokens)

    print("\n" + "=" * 70)
    print("TestNonCopyPath")
    print("=" * 70)

    print("\n  test_copy_token_not_valid_after_global_first_token")
    tok = FakeTokenizer()
    global_trie = _build_trie(["foo bar"], tok)
    foo_id = tok.encode("foo")[0]
    proc = _make_processor(global_trie, [[tok.encode("foo bar")]], tok)
    _show_after("after 'foo': [COPY] must be masked", tok, proc, [foo_id])

    print("\n  test_global_trie_second_token_valid")
    tok = FakeTokenizer()
    global_trie = _build_trie(["foo bar"], tok)
    foo_id, bar_id = tok.encode("foo bar")
    proc = _make_processor(global_trie, [[tok.encode("foo bar")]], tok)
    _show_after("after 'foo': 'bar' must be valid", tok, proc, [foo_id])

    print("\n  test_invalid_global_path_yields_eos")
    tok = FakeTokenizer()
    global_trie = _build_trie(["foo bar"], tok)
    rogue_id = tok.encode("zzz")[0]
    proc = _make_processor(global_trie, [[tok.encode("foo bar")]], tok)
    _show_after("after 'zzz' (off-trie): only EOS valid", tok, proc, [rogue_id])

    print("\n" + "=" * 70)
    print("TestMultiBeam  (2 samples × 2 beams)")
    print("=" * 70)

    tok = FakeTokenizer()
    sample0_ctx = tok.encode("alpha beta")
    sample1_ctx = tok.encode("gamma delta")
    alpha_id = sample0_ctx[0]
    gamma_id = sample1_ctx[0]
    global_trie = _build_trie(["alpha beta", "gamma delta"], tok)
    processor = TrieConstrainedLogitsProcessorWithTag(
        trie_root=global_trie,
        prompt_length=4,
        eos_token_id=tok.EOS_ID,
        copy_token_id=tok.COPY_ID,
        num_beams=2,
        context_ids=[[sample0_ctx], [sample1_ctx]],
    )
    input_ids = torch.zeros(4, 5, dtype=torch.long)
    input_ids[:, 4] = tok.COPY_ID
    scores = torch.zeros(4, tok.vocab_size)
    out = processor(input_ids, scores.clone())
    inv_vocab = {v: k for k, v in tok._vocab.items()}
    for beam_idx in range(4):
        sample = beam_idx // 2
        valid = sorted(_valid_tokens(out[beam_idx]))
        readable = [inv_vocab.get(t, f"id={t}") for t in valid]
        print(f"\n  beam {beam_idx} (sample {sample}), after [COPY]:")
        print(f"    valid ids : {valid}")
        print(f"    valid toks: {readable}")

    print("\n" + "=" * 70)
    print("TestICLData  (10 context_dependent + 10 all_noise from real data)")
    print("=" * 70)

    ctx_dep_samples, all_noise_samples = _load_icl_samples()
    tok2 = FakeTokenizer()
    all_doc_ids: set = set()
    for item in ctx_dep_samples + all_noise_samples:
        user = item["conversations"][0]["content"]
        for d in _extract_context_doc_ids(user):
            all_doc_ids.add(d)
        answer = item["conversations"][1]["content"]
        all_doc_ids.add(re.sub(r"^\[COPY\]\s*", "", answer).strip())
    for d in all_doc_ids:
        tok2.encode(d)
    global_trie2 = _build_trie(list(all_doc_ids), tok2)
    inv2 = {v: k for k, v in tok2._vocab.items()}

    print("\n  --- context_dependent samples ---")
    for item in ctx_dep_samples:
        user = item["conversations"][0]["content"]
        answer = item["conversations"][1]["content"]
        ctx_doc_ids = _extract_context_doc_ids(user)
        ctx_ids = [tok2.encode(d) for d in ctx_doc_ids]
        context_first_toks = {ids[0] for ids in ctx_ids if ids}
        proc = _make_processor(global_trie2, [ctx_ids], tok2, prompt_length=4)

        # step 0
        out0 = proc(_make_input_ids(4, []), _make_scores(tok2.vocab_size))
        valid0 = sorted(_valid_tokens(out0[0]))

        # after [COPY]
        out1 = proc(_make_input_ids(4, [tok2.COPY_ID]), _make_scores(tok2.vocab_size))
        valid1 = sorted(_valid_tokens(out1[0]) - {tok2.EOS_ID})

        print(f"\n  query: {item['conversations'][0]['content'].split(chr(10))[-2].strip()}")
        print(f"  answer: {answer!r}")
        print(f"  context doc IDs: {ctx_doc_ids}")
        print(f"  step-0 valid toks : {[inv2.get(t, f'id={t}') for t in valid0]}")
        print(f"  after-[COPY] valid: {[inv2.get(t, f'id={t}') for t in valid1]}")
        print(f"  context 1st tokens: {[inv2.get(t, f'id={t}') for t in sorted(context_first_toks)]}")

    print("\n  --- all_noise samples ---")
    for item in all_noise_samples:
        user = item["conversations"][0]["content"]
        answer = item["conversations"][1]["content"]
        ctx_doc_ids = _extract_context_doc_ids(user)
        ctx_ids = [tok2.encode(d) for d in ctx_doc_ids]
        answer_tokens = tok2.encode(answer.strip())
        first_tok = answer_tokens[0]
        proc = _make_processor(global_trie2, [ctx_ids], tok2, prompt_length=4)

        # step 0
        out0 = proc(_make_input_ids(4, []), _make_scores(tok2.vocab_size))
        valid0 = sorted(_valid_tokens(out0[0]))

        # after first answer token
        out1 = proc(_make_input_ids(4, [first_tok]), _make_scores(tok2.vocab_size))
        valid1 = sorted(_valid_tokens(out1[0]))

        print(f"\n  query: {item['conversations'][0]['content'].split(chr(10))[-2].strip()}")
        print(f"  answer: {answer!r}")
        print(f"  answer first tok: {inv2.get(first_tok, f'id={first_tok}')!r}  (id={first_tok})")
        print(f"  step-0 valid toks    : {[inv2.get(t, f'id={t}') for t in valid0]}")
        print(f"  after-first-tok valid: {[inv2.get(t, f'id={t}') for t in valid1]}")
        copy_present = tok2.COPY_ID in set(valid1)
        print(f"  [COPY] in valid after first tok: {copy_present}  (expected: False)")

    # -----------------------------------------------------------------------
    # Final pass/fail
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Running assertion checks...")
    print("=" * 70)

    suites = [TestStep0, TestCopyPath, TestNonCopyPath, TestMultiBeam, TestICLData]
    passed = failed = 0
    for suite_cls in suites:
        obj = suite_cls()
        if hasattr(obj, "setup_class"):
            obj.setup_class()
        for name in sorted(m for m in dir(obj) if m.startswith("test_")):
            try:
                getattr(obj, name)()
                print(f"  PASS  {suite_cls.__name__}::{name}")
                passed += 1
            except Exception:
                print(f"  FAIL  {suite_cls.__name__}::{name}")
                traceback.print_exc()
                failed += 1

    print(f"\nResults: {passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
