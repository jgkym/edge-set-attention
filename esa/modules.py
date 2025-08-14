"""
Core modules for the Edge Set Attention (ESA) model.

This module provides the building blocks for constructing graph neural networks
based on attention mechanisms over sets of edges.

- `MLP`: A standard Multi-Layer Perceptron.
- `AttentionBlock`: A versatile multi-head attention block supporting both
  self-attention and cross-attention, using a standard pre-normalization architecture.
- `PoolingByMultiHeadAttention`: A pooling layer that uses cross-attention
  to summarize a set of features into a fixed number of representations.
- `DenseEmbedding`: A module to create dense edge representations from
  graph structures.
- `EdgeSetAttention`: The main model, which composes the above blocks into a
  deep architecture for processing edge sets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.init import xavier_normal_
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch


class MLP(nn.Module):
    """A simple Multi-Layer Perceptron (MLP) with configurable hidden layers,
    activation function, and dropout.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, output_dim, bias=bias))
        layers.append(nn.GELU())
        self.network = nn.Sequential(*layers)
        self._init_weights(self.network)
        self.dropout_p = dropout

    def _init_weights(self, network):
        for module in network:
            if isinstance(module, nn.Linear):
                xavier_normal_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the MLP to the input tensor."""
        return F.dropout(self.network(x), p=self.dropout_p, training=self.training)


class SwiGLUFFN(nn.Module):
    """
    Swish-Gated Linear Unit Feed-Forward Network.

    This module implements the SwiGLU activation function within a feed-forward
    network block, as described in the paper "GLU Variants Improve Transformer"
    by Noam Shazeer. It's designed for efficiency and flexibility.

    Reference: https://arxiv.org/abs/2002.05202
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()

        # Projects from `input_dim` to `2 * hidden_dim` to compute gate and value in one go
        self.wv = nn.Linear(input_dim, 2 * hidden_dim, bias=bias)

        # Projects from `hidden_dim` back down to `output_dim`
        self.w1 = nn.Linear(hidden_dim, output_dim, bias=bias)

        self.act = nn.SiLU()  # Swish activation
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass.

        The operation is: Dropout(w3(SiLU(w1(x)) * w2(x)))
        """
        # Project and then split into two tensors for gate and value
        gate, value = self.wv(x).chunk(2, dim=-1)

        # Apply the SwiGLU activation
        fused_activation = self.act(gate) * value

        # Apply the final projection and dropout
        output = self.w1(fused_activation)
        output = F.dropout(output, p=self.dropout_p, training=self.training)

        return output


class AttentionBlock(nn.Module):
    """A versatile multi-head attention block that supports both self-attention and
    cross-attention, using a standard pre-normalization architecture.

    Args:
        dim: The dimensionality of the input and output features.
        num_heads: The number of attention heads.
        mlp_hidden_dims: A list of hidden dimensions for the feed-forward network.
        activation_fn: The activation function to use in the MLP.
        dropout: The dropout rate for both attention and MLP.
    """

    def __init__(
        self,
        dim: int,
        mlp_hidden_dim: int,
        num_heads: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")

        self.num_heads = num_heads
        self.dropout_p = dropout

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.proj_Q = nn.Linear(dim, dim, bias=False)
        self.proj_K = nn.Linear(dim, dim, bias=False)
        self.proj_V = nn.Linear(dim, dim, bias=False)
        self.proj_O = nn.Linear(dim, dim, bias=False)

        self.mlp = SwiGLUFFN(dim, mlp_hidden_dim, dim, dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        """Initializes the weights of the linear layers."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_normal_(p)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for the AttentionBlock.

        Args:
            x: The input tensor (Query). Shape: (batch_size, seq_len_q, dim).
            context: The context tensor (Key, Value). If None, performs self-attention.
                     Shape: (batch_size, seq_len_kv, dim).
            mask: The attention mask. Shape: (batch_size, num_heads, seq_len_q, seq_len_kv).

        Returns:
            The output tensor. Shape: (batch_size, seq_len_q, dim).
        """
        # --- First Sub-layer: Multi-Head Attention ---
        x_norm = self.norm1(x)
        shortcut1 = x_norm
        context_norm = self.norm1(context) if context is not None else x_norm

        Q = self.proj_Q(x_norm)
        K = self.proj_K(context_norm)
        V = self.proj_V(context_norm)

        Q = rearrange(Q, "B L (H D) -> B H L D", H=self.num_heads)
        K = rearrange(K, "B L (H D) -> B H L D", H=self.num_heads)
        V = rearrange(V, "B L (H D) -> B H L D", H=self.num_heads)

        if mask is not None and mask.dim() == 3:
            mask = rearrange(mask, "b nq nk -> b 1 nq nk").expand(
                -1, self.num_heads, -1, -1
            )

        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            attn_output = F.scaled_dot_product_attention(
                Q,
                K,
                V,
                attn_mask=mask,
                dropout_p=self.dropout_p if self.training else 0.0,
            )
        attn_output = rearrange(attn_output, "B H L D -> B L (H D)")
        attn_output = self.proj_O(attn_output)

        # Residual connection
        x = shortcut1 + F.dropout(attn_output, p=self.dropout_p, training=self.training)
        shortcut2 = x

        # --- Second Sub-layer: Feed-Forward Network ---
        x_norm2 = self.norm2(x)
        mlp_output = self.mlp(x_norm2)

        # Residual connection
        out = shortcut2 + F.dropout(mlp_output, p=self.dropout_p, training=self.training)
        return out


class PoolingByMultiHeadAttention(nn.Module):
    """A pooling layer that uses cross-attention to summarize a sequence of
    vectors into a fixed-size set of "seed" vectors.
    """

    def __init__(
        self,
        dim: int,
        mlp_hidden_dim: int,
        num_heads: int,
        num_seeds: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.S = nn.Parameter(torch.randn(1, num_seeds, dim))
        xavier_normal_(self.S)
        self.attention = AttentionBlock(dim, num_heads, mlp_hidden_dim, dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            x: The input tensor to be pooled. Shape: (batch_size, seq_len, dim).
            mask: An optional mask for the input tensor.

        Returns:
            The pooled output tensor. Shape: (batch_size, num_seeds, dim).
        """
        seeds = repeat(self.S, "1 S D -> B S D", B=x.size(0))
        return self.attention(x=seeds, context=x, mask=mask)


class DenseEmbedding(nn.Module):
    """Creates dense, fixed-size edge set representations from sparse graph data.

    It concatenates the features of source nodes, target nodes, and edges,
    then projects them through an MLP. Finally, it converts the sparse edge
    list into a dense tensor for processing by attention layers.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        mlp_hidden_dim: list[int],
        embedding_dim: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        input_dim = edge_dim + 2 * node_dim
        self.layernorm = nn.LayerNorm(input_dim)
        self.mlp = SwiGLUFFN(input_dim, mlp_hidden_dim, embedding_dim, dropout)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, data: Data, max_num_edges: int):
        """
        Args:
            data: A PyTorch Geometric `Data` or `Batch` object.
            max_num_edges: The size to pad the output tensor to. This should be
                         the maximum number of edges in any graph in the batch.
        """
        source_nodes = data.x[data.edge_index[0]]
        target_nodes = data.x[data.edge_index[1]]
        node_features = torch.cat([source_nodes, target_nodes], dim=1)

        edge_features = data.edge_attr.float()
        if edge_features.dim() == 1:
            edge_features = edge_features.unsqueeze(1)

        # Determine which graph each edge belongs to.
        edge_batch_index = data.batch.index_select(0, data.edge_index[0])

        # Concatenate all features for each edge.
        concatenated_features = torch.cat([node_features, edge_features], dim=1)

        dense_features = self.mlp(self.layernorm(concatenated_features))

        # Convert sparse edge features to a dense batch for the attention mechanism.
        final_features, _ = to_dense_batch(
            dense_features, edge_batch_index, max_num_nodes=max_num_edges
        )
        return final_features


class EdgeSetAttention(nn.Module):
    """The main Edge Set Attention (ESA) model.

    This model processes sets of edges using a configurable sequence of
    attention and pooling blocks. It uses a standard residual connection
    after each attention block, except for the final block in the sequence.
    """

    def __init__(
        self,
        dim: int,
        mlp_hidden_dim: int,
        num_heads: int,
        num_seeds: int = 1,
        dropout: float = 0.0,
        blocks: str = "MSMSP",
    ):
        super().__init__()
        self.block_types = blocks
        self.blocks = nn.ModuleList()

        for block_type in self.block_types:
            if block_type == "P":
                block = PoolingByMultiHeadAttention(
                    dim, mlp_hidden_dim, num_heads, num_seeds, dropout
                )
            elif block_type in ["M", "S"]:
                block = AttentionBlock(dim, mlp_hidden_dim, num_heads, dropout)
            else:
                raise ValueError(f"Unknown block type: {block_type}")
            self.blocks.append(block)

    def forward(
        self, x: torch.Tensor, adj_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Processes the input tensor through the sequence of blocks.

        Args:
            x: The input edge set tensor. Shape: (batch_size, num_edges, dim).
            adj_mask: The adjacency mask for the attention mechanism.
                      Shape: (batch_size, num_edges, num_edges).

        Returns:
            The processed tensor.
        """
        num_blocks = len(self.blocks)
        for i, block in enumerate(self.blocks):
            # Store the input for the residual connection
            residual = x

            # --- Block-specific processing ---
            if isinstance(block, PoolingByMultiHeadAttention):
                x = block(x)
                adj_mask = None
            elif isinstance(block, AttentionBlock):
                # Apply mask only for 'M' type blocks
                mask = adj_mask if self.block_types[i] == "M" else None
                x = block(x, mask=mask)

            # --- Residual Connection ---
            # Applied to all blocks except the final one.
            is_last_block = (i == num_blocks - 1)
            if not is_last_block and isinstance(block, AttentionBlock):
                x = x + residual

        return x