from pathlib import Path

import numpy as np
import torch
import torch_geometric.transforms as T
from tqdm.auto import tqdm

from esa.dataset import MolecularGraphPyGDataset
from esa.transforms import (
    AddMaxEdge,
    AddMaxEdgeGlobal,
    AddMaxNode,
    AddMaxNodeGlobal,
    AddPosEnc,
    ChempropFeatures,
    FormatSingleLabel,
    LabelNanToZero,
)


def get_max_node_edge_global(dataset):
    max_node_global = 0
    max_edge_global = 0

    for data in tqdm(dataset):
        if data.max_edge > max_edge_global:
            max_edge_global = data.max_edge
        if data.max_node > max_node_global:
            max_node_global = data.max_node

    return max_edge_global, max_node_global


def load_molecule_dataset(
    data_dir,
    filename: str | Path,
    smiles_col: str,
    target_col: str | None,
    one_hot: bool,
    max_atomic_number: int,
    pe_types: list[str],
    train_ratio: float | None,
    is_test: bool = False,
):
    transforms = [
        ChempropFeatures(
            one_hot=one_hot,
            max_atomic_number=max_atomic_number,
        ),
        AddMaxEdge(),
        AddMaxNode(),
    ]
    if not is_test:
        print("Adding transforms for training")
        transforms.extend([FormatSingleLabel(), LabelNanToZero()])
    if pe_types and len(pe_types) > 0:
        t_posenc = AddPosEnc(pe_types)
        transforms.append(t_posenc)

    dataset = MolecularGraphPyGDataset(
        root=data_dir,
        file_path=data_dir / filename,
        smiles_col=smiles_col,
        target_col=target_col,
        pre_transform=T.Compose(transforms),
        pre_filter=None,
    )

    print("\nDataset items look like: ", dataset[0])

    max_edge_global, max_node_global = get_max_node_edge_global(dataset)

    print(f"Datasets has {len(dataset)} elements")
    print(f"Maximum number of nodes per graph in dataset = {max_node_global}")
    print(f"Maximum number of edges per graph in dataset = {max_edge_global}")

    dataset.max_node_global = max_node_global
    dataset.max_edge_global = max_edge_global

    print("Applying global node/edge count transforms...")
    global_transforms = T.Compose(
        [
            AddMaxNodeGlobal(max_node_global),
            AddMaxEdgeGlobal(max_edge_global),
        ]
    )
    dataset.transform = global_transforms

    node_dim, edge_dim = dataset.x.shape[1], dataset.edge_attr.shape[1]
    print(f"Node feature dimension: {node_dim}")
    print(f"Edge feature dimension: {edge_dim}")

    if is_test:
        print("Finished loading data!")
        return dataset, node_dim, edge_dim

    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    train_idx = indices[: int(train_ratio * len(dataset))]
    val_idx = indices[int(train_ratio * len(dataset)) :]

    train_dataset = dataset.index_select(torch.from_numpy(train_idx))
    val_dataset = dataset.index_select(torch.from_numpy(val_idx))

    print(f"Original dataset size: {len(dataset)}")
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(val_dataset)}")

    print("Finished loading data!")

    return train_dataset, val_dataset, node_dim, edge_dim
