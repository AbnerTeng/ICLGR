import json
import os
from argparse import ArgumentParser, Namespace

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

from .models.pq import LearnablePQ
from .utils import seed_all


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


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--semid_code_path", type=str, help="csv")
    parser.add_argument("--all_items_path", type=str, help="jsonl")

    return parser.parse_args()


if __name__ == "__main__":
    seed_all(42)
    args = get_args()

    with open(args.all_items_path, "r") as f:
        all_items = [json.loads(line) for line in f]

    if not os.path.exists(args.semid_code_path):
        if "amazon" in args.all_items_path:
            documents = [item["description"] for item in all_items]
        elif "NQ" in args.all_items_path:
            documents = [item["text"].split(":")[1] for item in all_items]
        else:
            raise ValueError("Unknown dataset type in all_items_path")

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
        np.savetxt(args.semid_code_path, full_codes, delimiter=",", fmt="%d")

        print(f"Saved SEMID codes to {args.semid_code_path}")

    sem_docids = np.loadtxt(args.semid_code_path, delimiter=",", dtype=int).tolist()
    sem_code_count = {}
    for i, code in enumerate(sem_docids):
        code_tuple = tuple(code)
        if code_tuple not in sem_code_count:
            sem_code_count[code_tuple] = []
        sem_code_count[code_tuple].append(i)

    final_codes = []
    for code in sem_docids:
        code_tuple = tuple(code)
        indices = sem_code_count[code_tuple]

        if len(indices) == 1:
            final_codes.append([int(c) for c in code_tuple])
        else:
            final_codes.append(None)

    for code_tuple, indices in sem_code_count.items():
        if len(indices) > 1:
            for extra, idx in enumerate(indices):
                final_codes[idx] = [int(c) for c in code_tuple] + [extra]

    all_pq_tokens = set()
    for codes in final_codes:
        for level, code in enumerate(codes):
            all_pq_tokens.add(f"<|d{level}_{code}|>")

    print("\nProduct Quantization Results:")
    print(f"Total unique tokens: {len(all_pq_tokens)}")
    print(
        f"Items requiring disambiguation: {sum(1 for c in final_codes if len(c) > len(sem_docids[0]))}"
    )

    semantic_data = []
    for item, codes in zip(all_items, final_codes):
        item_copy = item.copy()
        item_copy["doc_id"] = codes_to_semantic_docid(codes)
        semantic_data.append(item_copy)

    output_path = "./data/NQ_gen_semid/train_semantic_docids.jsonl"
    with open(output_path, "w") as f:
        for item in semantic_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved semantic docids data to {output_path}")
