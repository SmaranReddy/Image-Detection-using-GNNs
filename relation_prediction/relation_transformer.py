"""
Cross-attention relation transformer with predicate queries.

Architecture
------------
Each feature group is a separate token with its own projection and modality embedding:

    [subj_label | obj_label | geo | subj_clip | obj_clip | union | pose]

Transformer encoder enables cross-modal interaction reasoning:

    subject ↔ object ↔ union ↔ pose ↔ geometry ↔ labels

Learnable predicate queries cross-attend to encoded interaction tokens
via a transformer decoder, enabling predicate-specific reasoning.

    riding query ← attends strongly to pose + union
    wearing query ← attends to person/object alignment
    looking at ← attends to head direction + geometry

Output: logits (B, num_predicates) — same API as RelationMLP.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vg_dataset import GEO_DIM, POSE_FEATURE_DIM, UNION_FEATURE_DIM


class _AttentionEncoderLayer(nn.TransformerEncoderLayer):
    """Encoder layer that forces attention weight computation.

    PyTorch 2.6 hardcodes ``need_weights=False`` in the base class
    ``_sa_block``, making attention capture impossible. This subclass
    overrides the helper to request per-head weights.
    """

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        x = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)


class _AttentionDecoderLayer(nn.TransformerDecoderLayer):
    """Decoder layer that forces attention weight computation.

    Overrides both ``_sa_block`` and ``_mha_block`` which PyTorch 2.6
    hardcodes with ``need_weights=False``.
    """

    def _sa_block(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        x = self.self_attn(
            x, x, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)

    def _mha_block(
        self,
        x: torch.Tensor,
        mem: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        key_padding_mask: Optional[torch.Tensor],
        is_causal: bool = False,
    ) -> torch.Tensor:
        x = self.multihead_attn(
            x, mem, mem,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
            is_causal=is_causal,
        )[0]
        return self.dropout2(x)


class RelationTransformer(nn.Module):
    """
    Cross-attention relation transformer.

    Treats each feature group as a token, applies transformer encoder
    for interaction reasoning, then uses predicate queries with a
    transformer decoder to produce predicate-specific logits.

    Args:
        num_labels:     Vocabulary size for subject / object labels.
        num_predicates: Number of predicate classes.
        d_model:        Common hidden dimension for all tokens.
        nhead:          Number of attention heads.
        num_encoder_layers: Number of transformer encoder layers.
        num_decoder_layers: Number of transformer decoder layers.
        dim_feedforward:   FFN hidden dimension.
        dropout:        Dropout probability.
        embed_dim:      Label embedding dimension.
        clip_dim:       CLIP visual embedding dimension (0 = no visual).
        pose_dim:       Pose feature dimension (0 = no pose).
        union_dim:      Union-region CLIP dimension (0 = no union).
    """

    def __init__(
        self,
        num_labels: int,
        num_predicates: int,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 1,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        embed_dim: int = 64,
        clip_dim: int = 0,
        pose_dim: int = 0,
        union_dim: int = 0,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.clip_dim = clip_dim
        self.pose_dim = pose_dim
        self.union_dim = union_dim
        self.embed_dim = embed_dim
        self.num_predicates = num_predicates

        # Label embedding (same interface as RelationMLP)
        self.label_emb = nn.Embedding(num_labels, embed_dim, padding_idx=0)

        # ── Modality-specific projections → d_model ──
        self.subj_label_proj = nn.Linear(embed_dim, d_model)
        self.obj_label_proj = nn.Linear(embed_dim, d_model)
        self.geo_proj = nn.Linear(GEO_DIM, d_model)

        if clip_dim > 0:
            self.subj_clip_proj = nn.Linear(clip_dim, d_model)
            self.obj_clip_proj = nn.Linear(clip_dim, d_model)
        if union_dim > 0:
            self.union_proj = nn.Linear(union_dim, d_model)
        if pose_dim > 0:
            self.pose_proj = nn.Linear(pose_dim, d_model)

        # ── Learned modality embeddings ──
        # Indices: 0=subj_label, 1=obj_label, 2=geo,
        #          3=subj_clip, 4=obj_clip, 5=union, 6=pose
        self.modality_emb = nn.Parameter(torch.randn(7, d_model))

        # ── Transformer encoder ──
        # Tokens attend to each other for interaction reasoning
        encoder_layer = _AttentionEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # ── Predicate queries ──
        # Each predicate learns what interaction patterns matter
        self.pred_queries = nn.Parameter(torch.randn(num_predicates, d_model))

        # ── Transformer decoder ──
        # Predicate queries cross-attend to encoded interaction tokens
        decoder_layer = _AttentionDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # ── Output head ──
        self.ln = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, 1)

        # ── Attention capture (for debugging) ──
        self._capture_attn = False
        self._encoder_attn_weights: List[torch.Tensor] = []
        self._decoder_self_attn_weights: List[torch.Tensor] = []
        self._decoder_cross_attn_weights: List[torch.Tensor] = []

        self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=0.1)
        # Init output head with smaller variance
        nn.init.normal_(self.output.weight, std=0.01)
        nn.init.zeros_(self.output.bias)

    # ------------------------------------------------------------------
    # Attention capture (Steps 9, 11)
    # ------------------------------------------------------------------

    def set_attention_capture(self, enabled: bool) -> None:
        """Enable or disable attention weight capture for debugging."""
        self._capture_attn = enabled
        if enabled:
            self._register_attn_hooks()
        else:
            self._remove_attn_hooks()

    def _clear_attn_weights(self) -> None:
        self._encoder_attn_weights.clear()
        self._decoder_self_attn_weights.clear()
        self._decoder_cross_attn_weights.clear()

    def _register_attn_hooks(self) -> None:
        if getattr(self, '_attn_hooks', None) is not None:
            self._remove_attn_hooks()

        self._attn_hooks = []
        for layer in self.encoder.layers:
            hook = self._make_attn_hook(self._encoder_attn_weights)
            self._attn_hooks.append(layer.self_attn.register_forward_hook(hook))
        for layer in self.decoder.layers:
            hook1 = self._make_attn_hook(self._decoder_self_attn_weights)
            self._attn_hooks.append(layer.self_attn.register_forward_hook(hook1))
            hook2 = self._make_attn_hook(self._decoder_cross_attn_weights)
            self._attn_hooks.append(layer.multihead_attn.register_forward_hook(hook2))

    def _remove_attn_hooks(self) -> None:
        hooks = getattr(self, '_attn_hooks', None)
        if hooks is not None:
            for hook in hooks:
                hook.remove()
            self._attn_hooks = None

    @staticmethod
    def _make_attn_hook(store: List):
        def hook(module, inp, out):
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                attn_weights = out[1]
                if attn_weights is not None and isinstance(attn_weights, torch.Tensor):
                    store.append(attn_weights.detach().cpu())
                elif attn_weights is None:
                    print(f"[attention] weights missing from {type(module).__name__}")
        return hook

    def get_attention_summary(
        self,
        pred_vocab_labels: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Aggregate attention weights per modality per predicate.

        Returns dict mapping predicate names to modality attention distributions.
        Only available after a forward pass with attention capture enabled.

        Expected weight shape after reduction:
            ``[num_predicates, num_modality_tokens]``

        Example output:
            {
                "riding": {"union_clip": 0.42, "pose": 0.29, ...},
                "sitting on": {"union_clip": 0.31, "pose": 0.25, ...},
            }
        """
        if not self._decoder_cross_attn_weights:
            return {}

        cross_attn = self._decoder_cross_attn_weights[-1]

        if cross_attn.dim() == 4:
            cross_attn = cross_attn.mean(dim=1)
        if cross_attn.dim() == 3:
            cross_attn = cross_attn.mean(dim=0)

        # cross_attn shape: [num_predicates, num_modalities]

        modality_names = {
            0: "subj_label", 1: "obj_label", 2: "geo",
            3: "subj_clip", 4: "obj_clip", 5: "union_clip", 6: "pose",
        }

        num_predicates = cross_attn.shape[0]
        num_modalities = cross_attn.shape[1]

        result = {}
        for p in range(num_predicates):
            attn = cross_attn[p]
            if attn.sum() > 0:
                attn = attn / attn.sum()
            pred_label = str(p)
            if pred_vocab_labels is not None and p < len(pred_vocab_labels):
                pred_label = pred_vocab_labels[p]
            per_mod = {}
            for m in range(num_modalities):
                mod_name = modality_names.get(m, f"token_{m}")
                per_mod[mod_name] = round(attn[m].item(), 4)
            sorted_mod = dict(sorted(per_mod.items(), key=lambda x: -x[1]))
            result[pred_label] = sorted_mod

        return result

    def get_encoder_attention(self) -> Optional[torch.Tensor]:
        """Return encoder self-attention weights from the last layer."""
        if not self._encoder_attn_weights:
            return None
        return self._encoder_attn_weights[-1]

    # ------------------------------------------------------------------
    # Token construction
    # ------------------------------------------------------------------

    def _build_tokens(
        self,
        se: torch.Tensor,
        oe: torch.Tensor,
        geo: torch.Tensor,
        subj_feat: Optional[torch.Tensor] = None,
        obj_feat: Optional[torch.Tensor] = None,
        union_feat: Optional[torch.Tensor] = None,
        pose_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[int]]:
        tokens: List[torch.Tensor] = []
        mod_ids: List[int] = []

        # Token 0: subject label embedding
        tokens.append(self.subj_label_proj(se).unsqueeze(1))
        mod_ids.append(0)

        # Token 1: object label embedding
        tokens.append(self.obj_label_proj(oe).unsqueeze(1))
        mod_ids.append(1)

        # Token 2: geometry
        tokens.append(self.geo_proj(geo).unsqueeze(1))
        mod_ids.append(2)

        # Token 3: subject CLIP
        if hasattr(self, 'subj_clip_proj') and subj_feat is not None:
            tokens.append(self.subj_clip_proj(subj_feat).unsqueeze(1))
            mod_ids.append(3)

        # Token 4: object CLIP
        if hasattr(self, 'obj_clip_proj') and obj_feat is not None:
            tokens.append(self.obj_clip_proj(obj_feat).unsqueeze(1))
            mod_ids.append(4)

        # Token 5: union CLIP
        if hasattr(self, 'union_proj') and union_feat is not None:
            tokens.append(self.union_proj(union_feat).unsqueeze(1))
            mod_ids.append(5)

        # Token 6: pose
        if hasattr(self, 'pose_proj') and pose_feat is not None:
            tokens.append(self.pose_proj(pose_feat).unsqueeze(1))
            mod_ids.append(6)

        x = torch.cat(tokens, dim=1)

        mod_emb = self.modality_emb[mod_ids].unsqueeze(0)
        x = x + mod_emb

        return x, mod_ids

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        subj_idx: torch.Tensor,
        obj_idx: torch.Tensor,
        geo: torch.Tensor,
        subj_feat: Optional[torch.Tensor] = None,
        obj_feat: Optional[torch.Tensor] = None,
        union_feat: Optional[torch.Tensor] = None,
        pose_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self._capture_attn:
            self._clear_attn_weights()

        B = subj_idx.shape[0]

        se = self.label_emb(subj_idx)
        oe = self.label_emb(obj_idx)

        x, _ = self._build_tokens(se, oe, geo, subj_feat, obj_feat, union_feat, pose_feat)

        memory = self.encoder(x)

        queries = self.pred_queries.unsqueeze(0).expand(B, -1, -1)
        queries = self.decoder(queries, memory)

        queries = self.ln(queries)
        logits = self.output(queries).squeeze(-1)

        return logits

    # ------------------------------------------------------------------
    # Feature contribution analysis (drop-in for _get_feature_group_norms)
    # ------------------------------------------------------------------

    def get_feature_contributions(
        self,
        subj_idx: torch.Tensor,
        obj_idx: torch.Tensor,
        geo: torch.Tensor,
        subj_feat: Optional[torch.Tensor] = None,
        obj_feat: Optional[torch.Tensor] = None,
        union_feat: Optional[torch.Tensor] = None,
        pose_feat: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Estimate feature modality importance via cross-attention.

        Runs forward pass with attention capture enabled and aggregates
        cross-attention weights per modality group across all predicates.
        """
        was_capturing = self._capture_attn
        self.set_attention_capture(True)

        with torch.no_grad():
            _ = self.forward(
                subj_idx, obj_idx, geo,
                subj_feat=subj_feat, obj_feat=obj_feat,
                union_feat=union_feat, pose_feat=pose_feat,
            )

        if not was_capturing:
            self.set_attention_capture(False)

        if not self._decoder_cross_attn_weights:
            return {}

        cross_attn = self._decoder_cross_attn_weights[-1]
        if cross_attn.dim() == 4:
            cross_attn = cross_attn.mean(dim=1)
        if cross_attn.dim() == 3:
            cross_attn = cross_attn.mean(dim=0)

        num_preds = cross_attn.shape[0]
        if cross_attn.shape[1] == 7:
            return {
                "subj_label": cross_attn[:, 0].sum().item(),
                "obj_label": cross_attn[:, 1].sum().item(),
                "geo": cross_attn[:, 2].sum().item(),
                "subj_clip": cross_attn[:, 3].sum().item() if cross_attn.shape[1] > 3 else 0.0,
                "obj_clip": cross_attn[:, 4].sum().item() if cross_attn.shape[1] > 4 else 0.0,
                "union_clip": cross_attn[:, 5].sum().item() if cross_attn.shape[1] > 5 else 0.0,
                "pose": cross_attn[:, 6].sum().item() if cross_attn.shape[1] > 6 else 0.0,
            }

        return {}
