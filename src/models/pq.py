from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans


class LearnablePQ(nn.Module):
    """
    Learnable Product Quantizer (no k-means).
    - dim: original embedding dim (e.g. 768)
    - M: number of subspaces (e.g. 8)
    - Ks: number of codes per subspace (e.g. 256)

    It learns:
    - a linear rotation W: [dim, dim] (optional "OPQ-like")
    - codebooks: [M, Ks, subdim]
    """

    def __init__(
        self,
        dim: int,
        M: int = 3,
        Ks: int = 256,
        normalize: bool = True,
        use_rotation: bool = True,
    ) -> None:
        super().__init__()
        assert dim % M == 0, "dim must be divisible by M."
        self.dim = dim
        self.M = M
        self.Ks = Ks
        self.subdim = dim // M
        self.normalize = normalize
        self.use_rotation = use_rotation

        if use_rotation:
            self.rot = nn.Linear(dim, dim, bias=False)
            nn.init.orthogonal_(self.rot.weight)
        else:
            self.rot = nn.Identity()

        self.codebooks = nn.Parameter(torch.empty(M, Ks, self.subdim))
        nn.init.normal_(self.codebooks, std=0.02)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, dim]
        return: [B, dim] (normalized + rotated)
        """
        z = x

        if self.normalize:
            z = F.normalize(z, dim=-1)
        z = self.rot(z)

        return z

    def _split_subspaces(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: [B, dim] -> [B, M, subdim]
        """
        B, D = z.shape
        assert D == self.dim
        return z.view(B, self.M, self.subdim)

    def forward(
        self, x: torch.Tensor, temp: float = 1.0, hard: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with differentiable quantization.

        x: [B, dim] embeddings.
        temp: Gumbel-Softmax temperature.
        hard: if True, straight-through hard assignment.

        Returns:
            quantized: [B, dim] quantized embeddings
            aux: dict containing:
                - "codes": [B, M] (argmax indices)
                - "prob":  [B, M, Ks] soft assignments
                - "usage": [M, Ks] avg probs per code
        """
        B, D = x.shape
        assert D == self.dim

        z = self._preprocess(x)  # [B, dim]
        z_split = self._split_subspaces(z)  # [B, M, subdim]
        codebooks_norm = F.normalize(self.codebooks, dim=-1)
        z_exp = z_split.unsqueeze(2)  # [B, M, 1, subdim]
        c_exp = codebooks_norm.unsqueeze(0)  # [1, M, Ks, subdim]
        dist2 = torch.sum((z_exp - c_exp) ** 2, dim=-1)  # [B, M, Ks]
        logits = -dist2 / temp
        prob = F.gumbel_softmax(logits, tau=temp, hard=hard, dim=-1)  # [B, M, Ks]
        cb = codebooks_norm.unsqueeze(0)  # [1, M, Ks, subdim]
        quant_sub = torch.sum(prob.unsqueeze(-1) * cb, dim=2)  # [B, M, subdim]
        quantized = quant_sub.reshape(B, self.dim)  # [B, dim]
        codes = prob.argmax(dim=-1)  # [B, M]
        usage = prob.mean(dim=0)  # [M, Ks]

        aux = {
            "codes": codes,
            "prob": prob,
            "usage": usage,
        }
        return quantized, aux

    @torch.no_grad()
    def encode_codes(self, x: torch.Tensor) -> torch.Tensor:
        """
        Non-differentiable encoding: nearest centroid per subspace.
        x: [B, dim]
        return codes: [B, M] (ints in [0, Ks-1])
        """
        _, D = x.shape
        assert D == self.dim

        z = self._preprocess(x)  # [B, dim]
        z_split = self._split_subspaces(z)  # [B, M, subdim]

        z_exp = z_split.unsqueeze(2)  # [B, M, 1, subdim]
        c_exp = self.codebooks.unsqueeze(0)  # [1, M, Ks, subdim]

        dist2 = torch.sum((z_exp - c_exp) ** 2, dim=-1)  # [B, M, Ks]
        codes = dist2.argmin(dim=-1)  # [B, M]
        return codes

    @torch.no_grad()
    def codes_to_docid_strings(
        self,
        codes: torch.Tensor,
        prefix: str = "<|d",
        sep: str = "_",
        suffix: str = "|>",
    ) -> List[str]:
        """
        Convert [B, M] codes to string docids like:
        '<|d0_123|><|d1_045|>...'
        """
        B, M = codes.shape
        codes_np = codes.cpu().numpy()
        res: List[str] = []
        for i in range(B):
            parts = []
            for m in range(M):
                parts.append(f"{prefix}{m}{sep}{int(codes_np[i, m]):03d}{suffix}")
            res.append("".join(parts))
        return res

    @torch.no_grad()
    def ambiguity_stats(self, codes: torch.Tensor) -> Dict[str, Any]:
        """
        Compute some ambiguity stats for encoded codes.
        codes: [N, M]
        """
        import collections

        N, M = codes.shape
        tuples = [tuple(row.tolist()) for row in codes.cpu()]
        counter = collections.Counter(tuples)

        num_unique = len(counter)
        max_coll = max(counter.values()) if counter else 1
        avg_docs_per_docid = N / max(num_unique, 1)

        per_sub = []
        for m in range(M):
            counts = collections.Counter(codes[:, m].tolist())
            used = len(counts)
            per_sub.append(
                {
                    "subspace": m,
                    "used_codes": used,
                    "fraction_used": used / self.Ks,
                    "max_cluster_size": max(counts.values()),
                    "min_cluster_size": min(counts.values()),
                }
            )

        return {
            "num_docs": N,
            "num_unique_docids": num_unique,
            "avg_docs_per_docid": avg_docs_per_docid,
            "max_docs_sharing_one_docid": max_coll,
            "per_subspace_usage": per_sub,
        }

    def compute_diversity_loss(self, probs: torch.Tensor) -> torch.Tensor:
        """
        Computes the entropy of the *average* code usage in the batch.
        We want to MAXIMIZE this entropy (make usage uniform).
        Loss = -Entropy.

        probs: [B, M, Ks] (soft assignments from forward pass)
        """
        avg_probs = probs.mean(dim=0)
        entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-10), dim=-1)  # [M]
        loss_div = -entropy.mean()

        return loss_div

    @torch.no_grad()
    def init_codebooks(self, data_loader, device="cpu", batches_to_use=10):
        """
        Run K-Means on a subset of data to initialize codebooks.
        Crucial for avoiding 'dead codes' at start.
        """
        print("Initializing codebooks with K-Means...")
        all_data = []
        for i, batch in enumerate(data_loader):
            if i >= batches_to_use:
                break
            # Assume batch is the raw embedding or tuple where 0 is embedding
            vals = batch[0] if isinstance(batch, (list, tuple)) else batch
            all_data.append(vals.to(device))

        x = torch.cat(all_data, dim=0)  # [N, dim]

        # Preprocess (Normalize / Rotate if needed)
        # Note: Rotation matrix is random at this point, but that's fine.
        z = self._preprocess(x)
        z_split = self._split_subspaces(z)  # [N, M, subdim]

        # Run K-Means for each subspace
        for m in range(self.M):
            sub_data = z_split[:, m, :].cpu().numpy()
            kmeans = KMeans(n_clusters=self.Ks, n_init=10, random_state=42)
            kmeans.fit(sub_data)

            # Assign centroids to codebooks
            new_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)
            self.codebooks.data[m] = new_centers.to(self.codebooks.device)

        print("K-Means initialization complete.")
