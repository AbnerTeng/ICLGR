import json

import hydra
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration


@hydra.main(
    version_base=None, config_path="../configs", config_name="pseudo_query_conf"
)
def main(cfg: DictConfig) -> None:
    p, gpq = cfg.paths, cfg.gen_pseudo_queries

    print(f"Loading {gpq.model}...")
    tokenizer = AutoTokenizer.from_pretrained(gpq.model)
    model = T5ForConditionalGeneration.from_pretrained(
        gpq.model, torch_dtype=torch.float16
    ).to(gpq.device)
    model.eval()

    print("Loading MSMARCO docs...")

    with open(p.msmarco_train) as f:
        raw = [json.loads(line) for line in f]
    docs = [d for d in raw if d.get("operation") == "indexing"]

    print(f"  {len(docs)} docs")

    with open(p.pseudo_queries, "w") as out_f:
        for i in tqdm(
            range(0, len(docs), gpq.batch_size), desc="Generating pseudo-queries"
        ):
            batch = docs[i : i + gpq.batch_size]
            texts = [d["text"] for d in batch]
            ids = [d["doc_id"] for d in batch]

            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=gpq.max_doc_len,
                return_tensors="pt",
            ).to(gpq.device)

            with torch.no_grad():
                outputs = model.generate(
                    **enc,
                    max_length=gpq.max_query_len,
                    do_sample=True,
                    top_p=0.95,
                    num_return_sequences=gpq.n_queries,
                )

            # outputs: (batch * n_queries, seq_len)
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for j, docid in enumerate(ids):
                pqs = decoded[j * gpq.n_queries : (j + 1) * gpq.n_queries]
                out_f.write(
                    json.dumps(
                        {"doc_idx": i + j, "doc_id": docid, "pseudo_queries": pqs}
                    )
                    + "\n"
                )

    print(f"Saved {len(docs)} entries → {p.pseudo_queries}")


if __name__ == "__main__":
    main()
