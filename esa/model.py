import torch
import torch.nn as nn
from torch_geometric.data import Data

from esa.modules import DenseEmbedding, EdgeSetAttention
from esa.utils import get_adj_mask
from utils.posenc_encoders.kernel_pos_encoder import KernelPENodeEncoder
from utils.posenc_encoders.laplace_pos_encoder import LapPENodeEncoder


class MoleculePropertyPredictor(nn.Module):
    """A model to predict molecular properties using Edge Set Attention.

    This model consists of three main parts:
    1. An embedding layer (`DenseEmbedding`) to create edge representations.
    2. An attention mechanism (`EdgeSetAttention`) to process the set of edges.
    3. A regression head to produce the final prediction.
    """

    def __init__(self, config, pe_types: list[str] | None = None):
        """
        Initializes the MoleculePropertyPredictor.

        Args:
            config: A configuration object with the following attributes:
                - embedding (object): Config for the DenseEmbedding module.
                    - node_dim (int): Dimensionality of node features.
                    - edge_dim (int): Dimensionality of edge features.
                    - hidden_dims (list[int]): Hidden dimensions for the embedding MLP.
                - esa (object): Config for the EdgeSetAttention module.
                    - num_heads (int): Number of attention heads.
                    - mlp_hidden_dims (list[int]): Hidden dimensions for the MLP inside attention blocks.
                    - num_seeds (int): Number of seed vectors for pooling.
                    - blocks (str): String defining the sequence of attention/pooling blocks.
                - activation_fn (nn.Module): The activation function to use.
                - dropout (float | None): The dropout rate.
        """
        super().__init__()

        # Use structured configs for sub-modules
        embedding_config = config.embedding
        esa_config = config.esa

        # The dimension for the ESA module is the output dim of the embedding layer
        esa_dim = embedding_config.embedding_dim

        if "RWSE" in pe_types:
            self.rwse_encoder = KernelPENodeEncoder()
            embedding_config.node_dim += 24
        if "LapPE" in pe_types:
            self.lap_encoder = LapPENodeEncoder()
            embedding_config.node_dim += 4

        self.embed = DenseEmbedding(
            node_dim=embedding_config.node_dim,
            edge_dim=embedding_config.edge_dim,
            mlp_hidden_dim=embedding_config.mlp_hidden_dim,
            embedding_dim=embedding_config.embedding_dim,
            dropout=config.dropout,
        )

        self.esa = EdgeSetAttention(
            dim=esa_dim,
            mlp_hidden_dim=esa_config.mlp_hidden_dim,
            num_heads=esa_config.num_heads,
            num_seeds=esa_config.num_seeds,
            dropout=config.dropout,
            blocks=esa_config.blocks,
        )

        # The regression head processes the output of the ESA module.
        self.use_pooling = "P" in esa_config.blocks
        if self.use_pooling:
            regression_in_features = esa_config.num_seeds * esa_dim
            self.regression_head = nn.Sequential(
                nn.Flatten(), nn.Linear(regression_in_features, 1)
            )
        else:
            # If no pooling block is used, apply global average pooling before the regressor.
            self.regression_head = nn.Linear(esa_dim, 1)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass for the model.

        Args:
            data: A PyTorch Geometric `Data` or `Batch` object.

        Returns:
            A tensor of predictions.
        """
        # Determine the maximum number of edges in the batch, which is required
        # for creating a dense tensor from the sparse graph data.
        edge_batch_index = data.batch.index_select(0, data.edge_index[0])
        edge_counts = torch.bincount(edge_batch_index)

        # Pad edge_counts to ensure it matches the batch size, even if some graphs have no edges.
        num_graphs = data.num_graphs
        if edge_counts.numel() < num_graphs:
            padding = edge_counts.new_zeros(num_graphs - edge_counts.numel())
            edge_counts = torch.cat([edge_counts, padding], dim=0)

        max_num_edges = max(1, int(edge_counts.max()) if len(edge_counts) > 0 else 0)

        data.x = data.x.float()

        if self.lap_encoder is not None:
            lap_pos_enc = self.lap_encoder(data.EigVals, data.EigVecs)
            data.x = torch.cat((data.x, lap_pos_enc), 1)
        if self.rwse_encoder is not None:
            rwse_pos_enc = self.rwse_encoder(data.pestat_RWSE)
            data.x = torch.cat((data.x, rwse_pos_enc), 1)

        # 1. Get dense edge set embeddings
        edge_set = self.embed(data, max_num_edges)
        batch_size = edge_set.size(0)

        # 2. Get adjacency mask for the attention mechanism
        adj_mask = get_adj_mask(data, batch_size, max_num_edges)
        adj_mask = adj_mask.to(edge_set.device)

        # 3. Process the edge set through the attention layers
        activations = self.esa(edge_set, adj_mask)

        # 4. Make the final prediction
        if not self.use_pooling:
            activations = activations.mean(dim=1)
        predictions = self.regression_head(activations)
        return predictions.squeeze()
