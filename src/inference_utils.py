import json
from typing import Any, Dict, List

import torch
import numpy as np
from transformers import LogitsProcessor
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi


class EmbeddingSearch:
    def __init__(self, train_data_path: str) -> None:
        self.train_data_path = train_data_path
        self.embedding_model = self._load_embedding_model()
        self.catalog = self._get_all_product_ids()
        self.embeddings = self._compute_embeddings()

    def _load_embedding_model(self) -> Any:
        return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def _get_all_product_ids(self) -> List[str]:
        catalog: List[str] = []

        with open(self.train_data_path, "r") as f:
            data = [json.loads(line) for line in f]

        for item in data:
            if item["operation"] == "indexing":
                catalog.append(item["doc_id"])

        return catalog

    def _compute_embeddings(self) -> np.ndarray:
        embeddings = []

        for title in self.catalog:
            embedding = self.embedding_model.encode(title)
            embeddings.append(embedding)

        return np.array(embeddings)

    def retrieve(self, generated_beams: List[str], top_k: int = 5) -> List[tuple]:
        candidate_map = {}

        for _, beam_text in enumerate(generated_beams):
            query_embedding = self.embedding_model.encode(beam_text)
            similarities = np.dot(self.embeddings, query_embedding)
            weight = 1.0
            top_n_indices = np.argpartition(similarities, -5)[-5:]

            for idx in top_n_indices:
                score = similarities[idx]

                if score <= 0:
                    continue

                if idx not in candidate_map:
                    candidate_map[idx] = score * weight
                else:
                    candidate_map[idx] = max(candidate_map[idx], score * weight)

        sorted_candidates = sorted(
            candidate_map.items(), key=lambda item: item[1], reverse=True
        )

        results = []

        for idx, score in sorted_candidates[:top_k]:
            results.append((self.catalog[idx], score))

        return results


class BM25Retriever:
    def __init__(self, train_data_path: str) -> None:
        self.train_data_path = train_data_path
        self.catalog = self._get_all_product_ids()
        self.tokenized_corpus = [self.tokenize(title) for title in self.catalog]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _get_all_product_ids(self) -> List[str]:
        catalog: List[str] = []

        with open(self.train_data_path, "r") as f:
            data = [json.loads(line) for line in f]

        for item in data:
            if item["operation"] == "indexing":
                catalog.append(item["doc_id"])

        return catalog

    def tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def retrieve(self, generated_beams: List[str], top_k: int = 5) -> List[tuple]:
        candidate_map = {}

        for _, beam_text in enumerate(generated_beams):
            tokenized_query = self.tokenize(beam_text)
            doc_scores = self.bm25.get_scores(tokenized_query)
            weight = 1.0
            top_n_indices = np.argpartition(doc_scores, -5)[-5:]

            for idx in top_n_indices:
                score = doc_scores[idx]

                if score <= 0:
                    continue

                if idx not in candidate_map:
                    candidate_map[idx] = score * weight
                else:
                    candidate_map[idx] = max(candidate_map[idx], score * weight)

        sorted_candidates = sorted(
            candidate_map.items(), key=lambda item: item[1], reverse=True
        )

        results = []

        for idx, score in sorted_candidates[:top_k]:
            results.append((self.catalog[idx], score))

        return results


class TrieNode:
    def __init__(self) -> None:
        self.children: Dict[int, "TrieNode"] = {}
        self.end_of_docid: bool = False


def build_semantic_docid_trie(train_data_path: str, tokenizer) -> TrieNode:
    root = TrieNode()

    with open(train_data_path, "r") as f:
        docids: List = []
        for line in f:
            item = json.loads(line)

            if item["operation"] == "indexing":
                docid = item["doc_id"]
                docids.append(docid)

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
    prompt_length: input_ids length before generation starts
    """

    def __init__(self, trie_root: TrieNode, prompt_length: int, eos_token_id: int):
        self.root = trie_root
        self.prompt_length = prompt_length
        self.eos_token_id = eos_token_id

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """
        Args:
            input_ids: (batch_size, sequence_length) - The full sequence generated so far.
            scores: (batch_size, vocab_size) - The raw logits for the NEXT token.
        """
        batch_size = input_ids.shape[0]

        for i in range(batch_size):
            current_generated_seq = input_ids[i][self.prompt_length :].tolist()
            node = self.root
            is_valid_path = True

            for token in current_generated_seq:
                if token in node.children:
                    node = node.children[token]
                else:
                    is_valid_path = False
                    break

            mask_value = float("-inf")

            if is_valid_path:
                valid_tokens = list(node.children.keys())

                if not valid_tokens and self.eos_token_id is not None:
                    valid_tokens = [self.eos_token_id]

                if len(valid_tokens) > 0:
                    valid_indices = torch.tensor(
                        valid_tokens, device=scores.device, dtype=torch.long
                    )

                    new_row_scores = torch.full_like(scores[i], mask_value)
                    new_row_scores[valid_indices] = scores[i][valid_indices]
                    scores[i] = new_row_scores
                else:
                    new_row_scores = torch.full_like(scores[i], mask_value)
                    if self.eos_token_id is not None:
                        new_row_scores[self.eos_token_id] = 0.0
                    scores[i] = new_row_scores

            else:
                new_row_scores = torch.full_like(scores[i], mask_value)
                if self.eos_token_id is not None:
                    new_row_scores[self.eos_token_id] = 0.0
                scores[i] = new_row_scores

        return scores
