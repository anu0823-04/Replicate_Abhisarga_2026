"""
EEGMoE Architecture — Adapted for Landslide Prediction
Original paper: EEGMoE: A Domain-Decoupled Mixture-of-Experts Model
                for Self-Supervised EEG Representation Learning
Adaptation: Landslide susceptibility prediction using satellite + hydro data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ─────────────────────────────────────────────
# 1. EXPERT (FFN with GELU) — directly from paper
# ─────────────────────────────────────────────
class Expert(nn.Module):
    """Single expert: 2-layer MLP with GELU (paper Section III-B)"""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────
# 2. SPECIFIC MoE — Top-K routing (Eq. 1,2,3 in paper)
# ─────────────────────────────────────────────
class SpecificMoE(nn.Module):
    """
    Specific expert group with Top-K routing.
    Learns domain-specific representations.
    Each token selects the top-K most relevant experts.
    """
    def __init__(self, input_dim, hidden_dim, num_experts=6, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim) for _ in range(num_experts)
        ])
        # Router: W_e in paper Eq.(1)
        self.router = nn.Linear(input_dim, num_experts, bias=False)

    def forward(self, x):
        # x: (batch, seq_len, dim)
        B, S, D = x.shape
        x_flat = x.view(B * S, D)

        # Routing scores g_x = W_e * x  (Eq. 1)
        logits = self.router(x_flat)  # (B*S, num_experts)

        # Softmax probabilities p_i(x)  (Eq. 2)
        probs = F.softmax(logits, dim=-1)

        # Top-K selection (Eq. 3)
        topk_vals, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        # Normalize selected probs
        topk_vals = topk_vals / topk_vals.sum(dim=-1, keepdim=True)

        # Weighted sum of top-k expert outputs
        output = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_indices = topk_idx[:, k]  # (B*S,)
            weights = topk_vals[:, k].unsqueeze(-1)  # (B*S, 1)
            for e_idx in range(self.num_experts):
                mask = (expert_indices == e_idx)
                if mask.any():
                    output[mask] += weights[mask] * self.experts[e_idx](x_flat[mask])

        return output.view(B, S, D), probs  # return probs for load balancing

    def load_balance_loss(self, probs):
        """Auxiliary load balancing loss (Eq. 7,8,9 in paper)"""
        # h_i: fraction of tokens assigned to expert i
        B_S = probs.shape[0]
        topk_vals, topk_idx = torch.topk(probs, self.top_k, dim=-1)
        
        # h_i = fraction of tokens routed to expert i (Eq. 8)
        h = torch.zeros(self.num_experts, device=probs.device)
        for i in range(self.num_experts):
            h[i] = (topk_idx == i).float().sum() / B_S

        # D_i = mean routing probability to expert i (Eq. 9)
        D = probs.mean(dim=0)

        # L_aux = E * sum(h_i * D_i) (Eq. 7)
        return self.num_experts * (h * D).sum()


# ─────────────────────────────────────────────
# 3. SHARED MoE — Soft routing (Eq. 4 in paper)
# ─────────────────────────────────────────────
class SharedMoE(nn.Module):
    """
    Shared expert group with soft routing.
    ALL experts are used for EVERY token.
    Learns domain-shared representations.
    """
    def __init__(self, input_dim, hidden_dim, num_shared_experts=2):
        super().__init__()
        self.experts = nn.ModuleList([
            Expert(input_dim, hidden_dim) for _ in range(num_shared_experts)
        ])
        self.router = nn.Linear(input_dim, num_shared_experts, bias=False)

    def forward(self, x):
        B, S, D = x.shape
        x_flat = x.view(B * S, D)

        # Soft routing: all experts contribute (Eq. 4)
        probs = F.softmax(self.router(x_flat), dim=-1)  # (B*S, F)

        output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            output += probs[:, i].unsqueeze(-1) * expert(x_flat)

        return output.view(B, S, D)


# ─────────────────────────────────────────────
# 4. SSMoE BLOCK — Specific + Shared (Eq. 5)
# ─────────────────────────────────────────────
class SSMoEBlock(nn.Module):
    """
    SSMoE = SpecificMoE + SharedMoE (Eq. 5 in paper)
    SSMoE(x) = SpecMoE(x) + ShareMoE(x)
    """
    def __init__(self, input_dim, hidden_dim,
                 num_specific=6, top_k=2, num_shared=2):
        super().__init__()
        self.specific = SpecificMoE(input_dim, hidden_dim, num_specific, top_k)
        self.shared = SharedMoE(input_dim, hidden_dim, num_shared)
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        spec_out, probs = self.specific(x)
        share_out = self.shared(x)
        # Additive fusion (Table XII shows this is best)
        out = spec_out + share_out
        return self.norm(x + out), probs  # residual + norm


# ─────────────────────────────────────────────
# 5. DOMAIN-DECOUPLED ENCODER (Transformer + SSMoE)
# ─────────────────────────────────────────────
class DomainDecoupledEncoderBlock(nn.Module):
    """One Transformer block with SSMoE replacing FFN"""
    def __init__(self, d_model, nhead, hidden_dim,
                 num_specific=6, top_k=2, num_shared=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ssmoe = SSMoEBlock(d_model, hidden_dim, num_specific, top_k, num_shared)

    def forward(self, x):
        # Multi-head self-attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        # SSMoE block replaces FFN
        x, probs = self.ssmoe(x)
        return x, probs


# ─────────────────────────────────────────────
# 6. SPATIAL & FREQUENCY ENCODER (MLP — paper Section III-A)
# ─────────────────────────────────────────────
class SpatialFrequencyEncoder(nn.Module):
    """
    2-layer MLP to encode 4D input into embeddings.
    Adapted: EEG frequency bands → satellite feature bands
    Input shape: (B, T, num_bands, H, W)
    Output shape: (B, T, embed_dim)
    """
    def __init__(self, input_channels, h, w, hidden_dim=64, embed_dim=128):
        super().__init__()
        flat_dim = input_channels * h * w
        self.encoder = nn.Sequential(
            nn.Flatten(start_dim=2),       # (B, T, C*H*W)
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        return self.encoder(x)  # (B, T, embed_dim)


# ─────────────────────────────────────────────
# 7. FULL EEGMoE MODEL (adapted for landslide)
# ─────────────────────────────────────────────
class LandslideEEGMoE(nn.Module):
    """
    Full EEGMoE adapted for landslide prediction.
    
    Input:  4D tensor (Batch, Time, Bands, H, W)
            - Bands: SAR + Sentinel-2 + Rainfall + Soil Moisture
            - H, W: spatial grid dimensions
    Output: Landslide probability per pixel or per patch
    """
    def __init__(self, 
                 input_channels=8,   # number of feature bands
                 h=9, w=9,           # spatial patch size
                 embed_dim=128,
                 hidden_dim=512,
                 num_encoder_layers=4,
                 nhead=4,
                 num_specific_experts=6,
                 top_k=2,
                 num_shared_experts=2,
                 num_classes=2,      # 0=no landslide, 1=landslide
                 mask_ratio=0.4):    # from paper Table XIII
        super().__init__()

        self.mask_ratio = mask_ratio
        self.embed_dim = embed_dim

        # Stage 1: Spatial & Frequency Encoder
        self.sf_encoder = SpatialFrequencyEncoder(
            input_channels, h, w, hidden_dim=64, embed_dim=embed_dim
        )

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, 100, embed_dim) * 0.02)

        # Stage 2: Domain-Decoupled Encoder (M blocks)
        self.encoder_blocks = nn.ModuleList([
            DomainDecoupledEncoderBlock(
                embed_dim, nhead, hidden_dim,
                num_specific_experts, top_k, num_shared_experts
            )
            for _ in range(num_encoder_layers)
        ])

        # Reconstruction head (for self-supervised pretraining)
        self.decoder = nn.Linear(embed_dim, embed_dim)

        # Classification head (for fine-tuning — just a linear layer per paper)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def random_mask(self, x):
        """Randomly mask embeddings for self-supervised pretraining"""
        B, S, D = x.shape
        num_mask = int(S * self.mask_ratio)
        mask_idx = torch.randperm(S)[:num_mask]
        mask = torch.zeros(B, S, dtype=torch.bool, device=x.device)
        mask[:, mask_idx] = True
        x_masked = x.clone()
        x_masked[mask] = 0
        return x_masked, mask

    def forward(self, x, mode='finetune'):
        """
        x: (B, T, C, H, W)
        mode: 'pretrain' or 'finetune'
        """
        B, T, C, H, W = x.shape

        # Encode spatial/frequency features
        z = self.sf_encoder(x)  # (B, T, embed_dim)

        # Add positional embedding
        z = z + self.pos_embed[:, :T, :]

        # Self-supervised masking (pretraining only)
        if mode == 'pretrain':
            z_masked, mask = self.random_mask(z)
        else:
            z_masked = z

        # Domain-Decoupled Encoder
        all_probs = []
        for block in self.encoder_blocks:
            z_masked, probs = block(z_masked)
            all_probs.append(probs)

        if mode == 'pretrain':
            # Reconstruct masked embeddings
            z_reconstructed = self.decoder(z_masked)
            return z_reconstructed, z, mask, all_probs  # for L1 loss

        else:  # finetune
            # Global average pool over time dimension
            pooled = z_masked.mean(dim=1)  # (B, embed_dim)
            logits = self.classifier(pooled)  # (B, num_classes)
            return logits, all_probs


# ─────────────────────────────────────────────
# 8. LOSS FUNCTIONS (from paper)
# ─────────────────────────────────────────────
def pretrain_loss(z_reconstructed, z_original, mask, all_probs, alpha=1e-4):
    """
    Total pretraining loss = L1 reconstruction + alpha * load balance (Eq. 10)
    """
    # L1 reconstruction loss on masked tokens only (Eq. 6)
    l1 = F.l1_loss(z_reconstructed[mask], z_original[mask])

    # Load balancing auxiliary loss (Eq. 7)
    l_aux = sum(
        block_probs.mean() for block_probs in all_probs
    )

    return l1 + alpha * l_aux


def finetune_loss(logits, labels):
    """Cross-entropy loss for classification (Eq. 11)"""
    return F.cross_entropy(logits, labels)


# ─────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────
if __name__ == '__main__':
    model = LandslideEEGMoE(
        input_channels=8,  # SAR(2) + S2(4) + rainfall(1) + soil(1)
        h=9, w=9,
        embed_dim=128,
        hidden_dim=512,
        num_classes=2
    )

    # Dummy input: (batch=4, time=5, channels=8, height=9, width=9)
    x = torch.randn(4, 5, 8, 9, 9)

    # Test finetune forward
    logits, probs = model(x, mode='finetune')
    print(f"✅ Model output shape: {logits.shape}")  # should be (4, 2)
    print(f"✅ Total parameters: {sum(p.numel() for p in model.parameters()):,}")
