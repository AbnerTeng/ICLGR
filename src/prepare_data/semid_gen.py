import json
import os

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from ..models.pq import LearnablePQ
from .utils import set_seed


class SemDataset(Dataset):
    def __init__(self, embeddings: torch.Tensor) -> None:
        self.embeddings = embeddings

    def __len__(self) -> int:
        return self.embeddings.size(0)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.embeddings[idx]


def codes_to_semantic_docid(codes: list, separator: str = " ") -> str:
    """Convert [0, 1, 2] to '<|d0_0|> <|d1_1|> <|d2_2|>'."""
    tokens = [f"<|d{level}_{code}|>" for level, code in enumerate(codes)]

    return separator.join(tokens)


if __name__ == "__main__":
    set_seed(42)
    semid_code_path = "./data/amzn_100k/semid_codes.csv"
    all_items_path = "./data/amzn_100k/all_subset_items_first3700.jsonl"

    with open(all_items_path, "r") as f:
        all_items = [json.loads(line) for line in f]

    if not os.path.exists(semid_code_path):
        documents = [item["description"] for item in all_items]
        # documents = [item["title"] for item in all_items]
        embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda")
        embeddings = embedding_model.encode(
            documents, batch_size=256, convert_to_tensor=True, show_progress_bar=True
        )
        print(f"Embeddings shape: {embeddings.shape}")

        dataset = SemDataset(embeddings)
        train_dataloader = DataLoader(dataset, batch_size=256, shuffle=True)
        test_dataloader = DataLoader(dataset, batch_size=256, shuffle=False)
        pq_model = LearnablePQ(dim=embeddings.shape[1]).to("cuda")
        pq_model.init_codebooks(train_dataloader, device="cuda", batches_to_use=50)
        optimizer = torch.optim.AdamW(pq_model.parameters(), lr=1e-3)
        diversity_weight = 1.0
        recon_loss_fn = torch.nn.MSELoss()

        num_epochs = 30

        for epoch in tqdm(range(num_epochs), desc="Training PQ"):
            epoch_loss, epoch_recon, epoch_div = 0.0, 0.0, 0.0
            pq_model.train()

            for batch in train_dataloader:
                batch = batch.to("cuda")
                quantized, aux = pq_model(batch, temp=1.0)
                loss_main = recon_loss_fn(quantized, batch)
                loss_div = pq_model.compute_diversity_loss(aux["prob"])
                loss = loss_main + (diversity_weight * loss_div)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                epoch_recon += loss_main.item()
                epoch_div += loss_div.item()

            if epoch % 1 == 0:
                avg_loss = epoch_loss / len(train_dataloader)
                avg_recon = epoch_recon / len(train_dataloader)
                avg_div = epoch_div / len(train_dataloader)
                print(
                    f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}, Recon: {avg_recon:.4f}, Usage: {avg_div:.4f}"
                )

        pq_model.eval()

        with torch.no_grad():
            full_codes = []

            for batch in tqdm(test_dataloader, desc="Encoding test set"):
                batch = batch.to("cuda")
                codes = pq_model.encode_codes(batch)
                full_codes.append(codes.cpu().numpy())

        full_codes = np.concatenate(full_codes, axis=0)
        stats = pq_model.ambiguity_stats(torch.tensor(full_codes))
        print("Ambiguity stats on test set:", stats)
        np.savetxt(semid_code_path, full_codes, delimiter=",", fmt="%d")

        print(f"Saved SEMID codes to {semid_code_path}")

    sem_docids = np.loadtxt(semid_code_path, delimiter=",", dtype=int).tolist()
    sem_to_title_map = {}
    titles = [item["title"] for item in all_items]

    for title, code in zip(titles, sem_docids):
        code_tuple = tuple(code)

        if code_tuple not in sem_to_title_map:
            sem_to_title_map[code_tuple] = []
        sem_to_title_map[code_tuple].append(title)

    title_to_sem_map = {}

    for code_tuple, title_list in sem_to_title_map.items():
        if len(title_list) == 1:
            title_to_sem_map[title_list[0]] = [int(c) for c in code_tuple]
        else:
            for extra, title_id in enumerate(title_list):
                title_to_sem_map[title_id] = [int(c) for c in code_tuple] + [extra]

    all_pq_tokens = set()

    for codes in title_to_sem_map.values():
        for level, code in enumerate(codes):
            all_pq_tokens.add(f"<|d{level}_{code}|>")

    print("\nProduct Quantization Results:")
    print(f"Total unique tokens: {len(all_pq_tokens)}")
    print(
        f"Items requiring disambiguation: {sum(1 for c in title_to_sem_map.values() if len(c) > 3)}"
    )

    stage1_semantic_data, stage1_title_data = [], []

    for item in all_items:
        title = item.get("title", "")
        item_copy_semantic = item.copy()
        item_copy_title = item.copy()
        item_copy_semantic["doc_id"] = codes_to_semantic_docid(title_to_sem_map[title])
        item_copy_title["doc_id"] = title
        stage1_semantic_data.append(item_copy_semantic)
        stage1_title_data.append(item_copy_title)

    with (
        open("./data/amzn_100k/train_semantic_docids.jsonl", "w") as f_semantic,
        open("./data/amzn_100k/train_title_docids.jsonl", "w") as f_title,
        open("./data/amzn_100k/title_to_semantic.jsonl", "w") as f_map,
    ):
        for item_semantic, item_title in zip(stage1_semantic_data, stage1_title_data):
            f_semantic.write(json.dumps(item_semantic, ensure_ascii=False) + "\n")
            f_title.write(json.dumps(item_title, ensure_ascii=False) + "\n")
            f_map.write(
                json.dumps(
                    {
                        "title": item_title["doc_id"],
                        "semantic_docid": item_semantic["doc_id"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
