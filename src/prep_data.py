import json
from pathlib import Path
from typing import List

from omegaconf import DictConfig
import hydra


@hydra.main(
    version_base=None, config_path="../configs", config_name="pseudo_query_conf"
)
def main(cfg: DictConfig) -> None:
    p = cfg.paths

    with open(p.msmarco_train) as f:
        org_train = [json.loads(line) for line in f]

    print(f"  Original training samples: {len(org_train)}")

    n_pq: int = 0
    pseudo_samples: List = []

    with open(p.pseudo_queries) as f:
        for line in f:
            entry = json.loads(line)
            docid = entry["doc_id"]

            for pq in entry["pseudo_queries"]:
                if pq.strip():
                    pseudo_samples.append(
                        {
                            "Unnamed: 0": len(org_train) + n_pq,
                            "item": docid,
                            "text": pq.strip(),
                            "source": "msmarco",
                            "doc_id": docid,
                            "operation": "query",
                        }
                    )
                    n_pq += 1

    print(f"  Pseudo-queries: {n_pq}")

    merged = org_train + pseudo_samples
    print(f"  Total merged samples: {len(merged)}")

    output_path = Path(p.msmarco_train).parent / "train_with_pseudo.jsonl"
    with open(output_path, "w") as f:
        for sample in merged:
            f.write(json.dumps(sample) + "\n")

    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
