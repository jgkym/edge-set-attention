import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset
from torch_geometric.utils import unbatch_edge_index


def split_dataset(
    dataset: Dataset, test_size: float = 0.2, random_state: int = 42
) -> tuple[Subset, Subset]:
    dataset_indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(
        dataset_indices, test_size=test_size, random_state=random_state
    )
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)
    return train_dataset, test_dataset


def get_adj_mask(batch, batch_size, max_items, device="cpu"):
    edge_index, batch_mapping = batch.edge_index, batch.batch
    empty_mask_fill_value = False
    mask_dtype = torch.bool
    edge_mask_fill_value = True

    adj_mask = torch.full(
        size=(batch_size, max_items, max_items),
        fill_value=empty_mask_fill_value,
        device=device,
        dtype=mask_dtype,
        requires_grad=False,
    )

    edge_index_unbatched = unbatch_edge_index(edge_index, batch_mapping)
    edge_batch_non_cumulative = torch.cat(edge_index_unbatched, dim=1)

    edge_batch_mapping = batch_mapping.index_select(0, edge_index[0, :])

    adj_mask[
        edge_batch_mapping,
        edge_batch_non_cumulative[0, :],
        edge_batch_non_cumulative[1, :],
    ] = edge_mask_fill_value

    adj_mask = ~adj_mask

    adj_mask = adj_mask.unsqueeze(1)
    return adj_mask


def set_seed(random_seed) -> None:
    """Sets the seed for reproducibility."""
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    elif torch.mps.is_available():
        torch.mps.manual_seed(random_seed)
    print(f"Seed set to {random_seed}")
