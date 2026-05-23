import json
from typing import Dict, List, Optional

import torch
from transformers import LogitsProcessor
# from rank_bm25 import BM25Okapi


# class EmbeddingSearch:
#     def __init__(self, train_data_path: str) -> None:
#         self.train_data_path = train_data_path
#         self.embedding_model = self._load_embedding_model()
#         self.catalog = self._get_all_product_ids()
#         self.embeddings = self._compute_embeddings()
#
#     def _load_embedding_model(self) -> Any:
#         return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
#
#     def _get_all_product_ids(self) -> List[str]:
#         catalog: List[str] = []
#
#         with open(self.train_data_path, "r") as f:
#             data = [json.loads(line) for line in f]
#
#         for item in data:
#             if item["operation"] == "indexing":
#                 catalog.append(item["doc_id"])
#
#         return catalog
#
#     def _compute_embeddings(self) -> np.ndarray:
#         embeddings = []
#
#         for title in self.catalog:
#             embedding = self.embedding_model.encode(title)
#             embeddings.append(embedding)
#
#         return np.array(embeddings)
#
#     def retrieve(self, generated_beams: List[str], top_k: int = 5) -> List[tuple]:
#         candidate_map = {}
#
#         for _, beam_text in enumerate(generated_beams):
#             query_embedding = self.embedding_model.encode(beam_text)
#             similarities = np.dot(self.embeddings, query_embedding)
#             weight = 1.0
#             top_n_indices = np.argpartition(similarities, -5)[-5:]
#
#             for idx in top_n_indices:
#                 score = similarities[idx]
#
#                 if score <= 0:
#                     continue
#
#                 if idx not in candidate_map:
#                     candidate_map[idx] = score * weight
#                 else:
#                     candidate_map[idx] = max(candidate_map[idx], score * weight)
#
#         sorted_candidates = sorted(
#             candidate_map.items(), key=lambda item: item[1], reverse=True
#         )
#
#         results = []
#
#         for idx, score in sorted_candidates[:top_k]:
#             results.append((self.catalog[idx], score))
#
#         return results
#
#
# class BM25Retriever:
#     def __init__(self, train_data_path: str) -> None:
#         self.train_data_path = train_data_path
#         self.catalog = self._get_all_product_ids()
#         self.tokenized_corpus = [self.tokenize(title) for title in self.catalog]
#         self.bm25 = BM25Okapi(self.tokenized_corpus)
#
#     def _get_all_product_ids(self) -> List[str]:
#         catalog: List[str] = []
#
#         with open(self.train_data_path, "r") as f:
#             data = [json.loads(line) for line in f]
#
#         for item in data:
#             if item["operation"] == "indexing":
#                 catalog.append(item["doc_id"])
#
#         return catalog
#
#     def tokenize(self, text: str) -> List[str]:
#         return text.lower().split()
#
#     def retrieve(self, generated_beams: List[str], top_k: int = 5) -> List[tuple]:
#         candidate_map = {}
#
#         for _, beam_text in enumerate(generated_beams):
#             tokenized_query = self.tokenize(beam_text)
#             doc_scores = self.bm25.get_scores(tokenized_query)
#             weight = 1.0
#             top_n_indices = np.argpartition(doc_scores, -5)[-5:]
#
#             for idx in top_n_indices:
#                 score = doc_scores[idx]
#
#                 if score <= 0:
#                     continue
#
#                 if idx not in candidate_map:
#                     candidate_map[idx] = score * weight
#                 else:
#                     candidate_map[idx] = max(candidate_map[idx], score * weight)
#
#         sorted_candidates = sorted(
#             candidate_map.items(), key=lambda item: item[1], reverse=True
#         )
#
#         results = []
#
#         for idx, score in sorted_candidates[:top_k]:
#             results.append((self.catalog[idx], score))
#
#         return results
#


class TrieNode:
    def __init__(self) -> None:
        self.children: Dict[int, "TrieNode"] = {}
        self.end_of_docid: bool = False


def build_semantic_docid_trie(
    data_paths: List[str],
    tokenizer,
    subsample_size: int = 0,
    seed: int = 42,
) -> TrieNode:
    """Build a trie from indexing entries in data_paths.

    subsample_size: if > 0, randomly sample this many docs from the LAST path only.
    seed: random seed for reproducible subsampling.
    """
    import random

    root = TrieNode()

    if isinstance(data_paths, str):
        data_paths = [data_paths]

    docids: List[str] = []

    for path_idx, path in enumerate(data_paths):
        path_docids: List[str] = []
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line)

                if item.get("operation") == "indexing":
                    path_docids.append(item["doc_id"])

        # subsample only the last (new) file
        if subsample_size > 0 and path_idx == len(data_paths) - 1 and len(data_paths) > 1:
            rng = random.Random(seed)
            path_docids = rng.sample(path_docids, min(subsample_size, len(path_docids)))

        docids.extend(path_docids)

    for doc_id_str in docids:
        token_ids = tokenizer.encode(doc_id_str, add_special_tokens=False)
        node = root

        for token_id in token_ids:
            if token_id not in node.children:
                node.children[token_id] = TrieNode()

            node = node.children[token_id]
        node.end_of_docid = True

    return root


class TrieConstrainedLogitsProcessor(LogitsProcessor):
    """
    HuggingFace LogitsProcessor that masks invalid tokens based on a Trie.

    Optionally accepts a second trie (extra_root); valid tokens at each step
    are the union of both tries, so the decode space = trie_root ∪ extra_root.
    prompt_length: input_ids length before generation starts.
    """

    def __init__(
        self,
        trie_root: TrieNode,
        prompt_length: int,
        eos_token_id: int,
        extra_root: Optional[TrieNode] = None,
    ):
        self.root = trie_root
        self.extra_root = extra_root
        self.prompt_length = prompt_length
        self.eos_token_id = eos_token_id

    def _navigate(self, root: TrieNode, token_seq: list):
        """Walk root along token_seq; return (node, is_valid_path)."""
        node = root
        for token in token_seq:
            if token in node.children:
                node = node.children[token]
            else:
                return node, False
        return node, True

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Args:
            input_ids: (batch_size, sequence_length) - The full sequence generated so far.
            scores: (batch_size, vocab_size) - The raw logits for the NEXT token.
        """
        batch_size = input_ids.shape[0]
        mask_value = float("-inf")

        for i in range(batch_size):
            generated = input_ids[i][self.prompt_length :].tolist()

            node, valid = self._navigate(self.root, generated)
            valid_tokens = set(node.children.keys()) if valid else set()
            at_end = valid and node.end_of_docid

            if self.extra_root is not None:
                extra_node, extra_valid = self._navigate(self.extra_root, generated)
                if extra_valid:
                    valid_tokens |= set(extra_node.children.keys())
                    at_end = at_end or extra_node.end_of_docid

            if at_end:
                valid_tokens.add(self.eos_token_id)

            if not valid_tokens:
                new_row = torch.full_like(scores[i], mask_value)
                if self.eos_token_id is not None:
                    new_row[self.eos_token_id] = 0.0
                scores[i] = new_row
            else:
                valid_indices = torch.tensor(
                    list(valid_tokens), device=scores.device, dtype=torch.long
                )
                new_row = torch.full_like(scores[i], mask_value)
                new_row[valid_indices] = scores[i][valid_indices]
                scores[i] = new_row

        return scores
