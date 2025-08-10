from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.utils import from_smiles
from tqdm.auto import tqdm


class MolecularGraphPyGDataset(InMemoryDataset):
    def __init__(
        self,
        root: Path | None,
        file_path: str | Path,
        smiles_col: str = "smiles",
        target_col: str | None = "pIC50",
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        self.file_path = Path(file_path)
        self.smiles_col = smiles_col
        self.target_col = target_col
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self) -> list[str]:
        return [self.file_path.name]

    @property
    def raw_paths(self) -> list[Path]:
        return [self.file_path]

    @property
    def processed_file_names(self) -> str:
        return "data.pt"

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        smiles_list = df[self.smiles_col]

        targets = None
        if self.target_col and self.target_col in df:
            targets = df[self.target_col]

        data_list = []
        for i, smiles in enumerate(tqdm(smiles_list, desc="Processing SMILES")):
            molecular_graph = from_smiles(smiles)

            if targets is not None:
                molecular_graph.y = torch.tensor(targets[i], dtype=torch.float).view(
                    1, -1
                )

            data_list.append(molecular_graph)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
