from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


REGION_MAP_62_10 = (
    0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1,
    1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6,
    4, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 5, 5, 5, 8,
    8, 8, 7, 7, 9, 9, 9, 8, 8, 7, 9, 9, 9, 8,
)


# THU-EP/FACED order:
# Fp1 Fp2 Fz F3 F4 F7 F8 FC1 FC2 FC5 FC6 Cz C3 C4 T7 T8
# A1 A2 CP1 CP2 CP5 CP6 Pz P3 P4 P7 P8 PO3 PO4 Oz O1 O2
REGION_MAP_32_10 = (
    0, 0, 0, 1, 2, 1, 2, 1, 2, 1, 2, 3, 3, 3, 4, 5,
    4, 5, 6, 7, 6, 7, 8, 6, 7, 6, 7, 9, 9, 8, 9, 9,
)


def _default_region_ids(num_channels: int) -> tuple[int, ...]:
    if num_channels == 62:
        return REGION_MAP_62_10
    if num_channels == 32:
        return REGION_MAP_32_10
    return tuple(range(num_channels))


class ChebyshevGraphConv(nn.Module):
    """ChebyNet graph convolution used by the MSGM spatial encoders."""

    def __init__(self, order: int, in_features: int, out_features: int):
        super().__init__()
        if order < 1:
            raise ValueError("Chebyshev order must be >= 1")
        self.order = order
        self.filter_weight = nn.Parameter(torch.empty(order, 1, in_features, out_features))
        self.filter_bias = nn.Parameter(torch.empty(1, 1, out_features))
        nn.init.normal_(self.filter_weight, mean=0.0, std=0.1)
        nn.init.normal_(self.filter_bias, mean=0.0, std=0.1)

    @staticmethod
    def normalized_laplacian(adjacency: torch.Tensor) -> torch.Tensor:
        degree = adjacency.sum(dim=1)
        degree_norm = torch.rsqrt(degree + 1.0e-5)
        degree_matrix = torch.diag(degree_norm)
        return -degree_matrix @ adjacency @ degree_matrix

    def forward(self, data: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x, adjacency = data
        laplacian = self.normalized_laplacian(adjacency)

        cheb_terms = [x]
        if self.order > 1:
            cheb_terms.append(torch.matmul(laplacian, x))
        for _ in range(2, self.order):
            cheb_terms.append(2 * torch.matmul(laplacian, cheb_terms[-1]) - cheb_terms[-2])

        stacked = torch.stack(cheb_terms, dim=1).permute(1, 0, 2, 3)
        out = torch.matmul(stacked, self.filter_weight).sum(dim=0)
        out = F.relu(out - self.filter_bias)
        return out, adjacency


class GraphEncoder(nn.Module):
    """A shallow or deep ChebyNet encoder followed by graph-to-token projection."""

    def __init__(
        self,
        num_layers: int,
        num_nodes: int,
        in_features: int,
        hidden_features: int,
        chebyshev_order: int,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        for layer_idx in range(num_layers):
            layer_in = in_features if layer_idx == 0 else hidden_features
            layers.append(ChebyshevGraphConv(chebyshev_order, layer_in, hidden_features))
        self.encoder = nn.Sequential(*layers)
        self.tokenizer = nn.Linear(num_nodes * hidden_features, hidden_features)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.encoder((x, adjacency))
        return self.tokenizer(encoded.reshape(encoded.size(0), -1))


@dataclass
class MambaConfig:
    d_model: int = 32
    n_layer: int = 1
    expand: int = 2
    dt_rank: Union[int, str] = "auto"
    d_conv: int = 4
    d_state: int = 16
    conv_bias: bool = True
    bias: bool = True

    def __post_init__(self) -> None:
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == "auto":
            self.dt_rank = math.ceil(self.d_model / 16)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1.0e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps) * self.weight


class MambaBlock(nn.Module):
    """Single selective state-space block used inside MSST-Mamba."""

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.config = config
        self.in_proj = nn.Linear(config.d_model, config.d_inner * 2, bias=config.bias)
        self.conv1d = nn.Conv1d(
            in_channels=config.d_inner,
            out_channels=config.d_inner,
            kernel_size=config.d_conv,
            groups=config.d_inner,
            padding=config.d_conv - 1,
            bias=config.conv_bias,
        )
        self.x_proj = nn.Linear(config.d_inner, config.dt_rank + config.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(config.dt_rank, config.d_inner, bias=True)

        a = torch.arange(1, config.d_state + 1).repeat(config.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a.float()))
        self.D = nn.Parameter(torch.ones(config.d_inner))
        self.out_proj = nn.Linear(config.d_inner, config.d_model, bias=config.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        x_proj, gate = self.in_proj(x).split(self.config.d_inner, dim=-1)
        x_proj = x_proj.transpose(1, 2)
        x_proj = self.conv1d(x_proj)[:, :, :length].transpose(1, 2)
        x_proj = F.silu(x_proj)
        y = self.ssm(x_proj)
        y = y * F.silu(gate)
        return self.out_proj(y.reshape(batch, length, self.config.d_inner))

    def ssm(self, x: torch.Tensor) -> torch.Tensor:
        d_state = self.A_log.shape[1]
        a = -torch.exp(self.A_log.float())
        d = self.D.float()
        x_dbl = self.x_proj(x)
        delta, b, c = x_dbl.split([self.config.dt_rank, d_state, d_state], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        return self.selective_scan(x, delta, a, b, c, d)

    @staticmethod
    def selective_scan(
        u: torch.Tensor,
        delta: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        c: torch.Tensor,
        d: torch.Tensor,
    ) -> torch.Tensor:
        batch, length, d_inner = u.shape
        d_state = a.shape[1]
        delta_a = torch.exp(torch.einsum("bld,dn->bldn", delta, a))
        delta_b_u = torch.einsum("bld,bln,bld->bldn", delta, b, u)

        state = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        outputs = []
        for step in range(length):
            state = delta_a[:, step] * state + delta_b_u[:, step]
            outputs.append(torch.einsum("bdn,bn->bd", state, c[:, step]))
        y = torch.stack(outputs, dim=1)
        return y + u * d


class MSSTBlock(nn.Module):
    def __init__(self, config: MambaConfig):
        super().__init__()
        self.norm = RMSNorm(config.d_model)
        self.mixer = MambaBlock(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixer(self.norm(x)) + x


class MSSTMamba(nn.Module):
    """Multi-scale spatiotemporal state-space module."""

    def __init__(self, config: MambaConfig):
        super().__init__()
        self.layers = nn.ModuleList([MSSTBlock(config) for _ in range(config.n_layer)])
        self.norm = RMSNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class MSGM(nn.Module):
    """Multi-Scale Spatiotemporal Graph Mamba for EEG emotion recognition."""

    def __init__(
        self,
        num_channels: int = 62,
        num_features: int = 7,
        num_classes: int = 2,
        hidden_dim: int = 32,
        graph_layers: Sequence[int] = (1, 2),
        chebyshev_order: int = 4,
        scale_lengths: Sequence[int] = (16, 23, 36, 76),
        region_ids: Optional[Sequence[int]] = None,
        mamba_layers: int = 1,
        mamba_state_dim: int = 16,
        mamba_conv_kernel: int = 4,
        mamba_expand: int = 2,
        dropout: float = 0.25,
    ):
        super().__init__()
        if len(graph_layers) != 2:
            raise ValueError("graph_layers must contain [shallow_layers, deep_layers]")
        if not scale_lengths:
            raise ValueError("scale_lengths must contain at least one temporal scale")

        self.num_channels = num_channels
        self.num_features = num_features
        self.hidden_dim = hidden_dim
        self.scale_lengths = tuple(int(x) for x in scale_lengths)

        regions = tuple(region_ids) if region_ids is not None else _default_region_ids(num_channels)
        if len(regions) != num_channels:
            raise ValueError("region_ids length must match num_channels")
        self.register_buffer("region_ids", torch.tensor(regions, dtype=torch.long), persistent=False)

        self.global_shallow_encoder = GraphEncoder(
            graph_layers[0], num_channels, num_features, hidden_dim, chebyshev_order
        )
        self.global_deep_encoder = GraphEncoder(
            graph_layers[1], num_channels, num_features, hidden_dim, chebyshev_order
        )
        self.local_shallow_encoder = GraphEncoder(
            graph_layers[0], num_channels, num_features, hidden_dim, chebyshev_order
        )
        self.local_deep_encoder = GraphEncoder(
            graph_layers[1], num_channels, num_features, hidden_dim, chebyshev_order
        )

        self.scale_projection = nn.ParameterDict()
        self.scale_bias = nn.ParameterDict()
        for length in self.scale_lengths:
            key = self._scale_key(length)
            self.scale_projection[key] = nn.Parameter(torch.empty(length * num_features, length))
            self.scale_bias[key] = nn.Parameter(torch.zeros(num_channels, length))
            nn.init.xavier_uniform_(self.scale_projection[key])

        self.base_embedding = nn.Linear(num_channels * num_features, hidden_dim, bias=False)
        mamba_config = MambaConfig(
            d_model=hidden_dim,
            n_layer=mamba_layers,
            d_state=mamba_state_dim,
            d_conv=mamba_conv_kernel,
            expand=mamba_expand,
        )
        self.msst_mamba = MSSTMamba(mamba_config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    @staticmethod
    def _scale_key(length: int) -> str:
        return f"s{int(length)}"

    def build_spatial_priors(
        self,
        x: torch.Tensor,
        pcc_threshold: Optional[torch.Tensor] = None,
        md_threshold: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct duplicated global and local adjacency tensors for one scale."""
        _, seq_len, channels, features = x.shape
        if channels != self.num_channels or features != self.num_features:
            raise ValueError(
                f"Expected channels/features {(self.num_channels, self.num_features)}, "
                f"got {(channels, features)}"
            )

        key = self._scale_key(seq_len)
        if key not in self.scale_projection:
            raise ValueError(
                f"Unsupported sequence length {seq_len}. "
                f"Configured scale_lengths={self.scale_lengths}"
            )

        averaged = x.mean(dim=0)
        flattened = averaged.reshape(channels, seq_len * features)
        transformed = flattened @ self.scale_projection[key] + self.scale_bias[key]

        centered = transformed - transformed.mean(dim=1, keepdim=True)
        std = transformed.std(dim=1, keepdim=True).clamp_min(1.0e-6)
        normalized = centered / std
        pcc_matrix = normalized @ normalized.T / seq_len
        md_matrix = torch.cdist(transformed, transformed, p=1)
        euclidean = torch.cdist(transformed, transformed, p=2)

        if pcc_threshold is None:
            pcc_threshold = torch.quantile(pcc_matrix.flatten(), 0.75)
        if md_threshold is None:
            md_threshold = torch.quantile(md_matrix.flatten(), 0.25)
        if sigma is None:
            sigma = (euclidean.mean() + euclidean.std()) / 2
        sigma = sigma.clamp_min(1.0e-6)

        gaussian = torch.exp(-(euclidean ** 2) / (2 * sigma ** 2))
        global_graph = gaussian * ((pcc_matrix >= pcc_threshold) & (md_matrix <= md_threshold)).float()

        region_ids = self.region_ids.to(x.device)
        local_mask = (region_ids[:, None] == region_ids[None, :]).float()
        local_graph = global_graph * local_mask
        return torch.stack([global_graph, global_graph], dim=0), torch.stack([local_graph, local_graph], dim=0)

    def fuse_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Apply graph fusion and MSST-Mamba to one temporal scale."""
        batch, seq_len, channels, features = x.shape
        global_graph, local_graph = self.build_spatial_priors(x)
        flat = x.reshape(batch * seq_len, channels, features)
        base = self.base_embedding(flat.reshape(batch * seq_len, -1))

        global_token = torch.stack(
            (
                base,
                self.global_shallow_encoder(flat, global_graph[0]),
                self.global_deep_encoder(flat, global_graph[1]),
            ),
            dim=1,
        ).mean(dim=1)
        local_token = torch.stack(
            (
                base,
                self.local_shallow_encoder(flat, local_graph[0]),
                self.local_deep_encoder(flat, local_graph[1]),
            ),
            dim=1,
        ).mean(dim=1)

        global_token = global_token.reshape(batch, seq_len, self.hidden_dim)
        local_token = local_token.reshape(batch, seq_len, self.hidden_dim)

        global_embedding = F.normalize(self.msst_mamba(global_token).mean(dim=1), p=2, dim=1)
        local_embedding = F.normalize(self.msst_mamba(local_token).mean(dim=1), p=2, dim=1)
        return torch.stack((global_embedding, local_embedding), dim=1).mean(dim=1)

    def forward(self, *scale_tensors: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
        if len(scale_tensors) == 1 and isinstance(scale_tensors[0], (list, tuple)):
            scale_tensors = tuple(scale_tensors[0])
        if len(scale_tensors) == 0:
            raise ValueError("MSGM.forward expects at least one scale tensor")

        scale_embeddings = [self.fuse_scale(scale) for scale in scale_tensors]
        fused = torch.stack(scale_embeddings, dim=1).mean(dim=1)
        return self.classifier(self.dropout(fused))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
