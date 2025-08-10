import torch
import torch_geometric.transforms as T
from rdkit import Chem

from esa.chemprop_featurization import (
    atom_features,
    atom_features_int,
    bond_features,
    bond_features_int,
    factory,
    get_atom_constants,
    global_features,
)
from esa.posenc import compute_posenc_stats


def add_chemprop_features(data, one_hot, max_atomic_number):
    atom_constants = get_atom_constants(max_atomic_number)
    mol = Chem.MolFromSmiles(data.smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    # Pre-calculate molecule-wide features
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol)
    feats = factory.GetFeaturesForMol(mol)
    atom_pharmacophore_features = {}
    for i in range(len(feats)):
        atom_ids = feats[i].GetAtomIds()
        for atom_id in atom_ids:
            if atom_id not in atom_pharmacophore_features:
                atom_pharmacophore_features[atom_id] = []
            atom_pharmacophore_features[atom_id].append(feats[i].GetFamily())

    ei = torch.nonzero(torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol))).T
    if one_hot:
        atom_feat = torch.tensor(
            [
                atom_features(atom, atom_constants, atom_pharmacophore_features)
                for atom in mol.GetAtoms()
            ],
            dtype=torch.float,
        )

        bond_feat = torch.tensor(
            [
                bond_features(
                    mol.GetBondBetweenAtoms(ei[0][i].item(), ei[1][i].item()), mol
                )
                for i in range(ei.shape[1])
            ],
            dtype=torch.float,
        )
    else:
        atom_feat = torch.tensor(
            [
                atom_features_int(atom, atom_constants, atom_pharmacophore_features)
                for atom in mol.GetAtoms()
            ],
            dtype=torch.float,
        )

        bond_feat = torch.tensor(
            [
                bond_features_int(
                    mol.GetBondBetweenAtoms(ei[0][i].item(), ei[1][i].item()), mol
                )
                for i in range(ei.shape[1])
            ],
            dtype=torch.float,
        )

    global_feat = torch.tensor(global_features(mol), dtype=torch.float)
    data.x = atom_feat
    data.edge_index = ei

    data.edge_attr = bond_feat
    data.global_feat = global_feat

    return data


class ChempropFeatures(T.BaseTransform):
    def __init__(self, one_hot, max_atomic_number):
        self.one_hot = one_hot
        self.max_atomic_number = max_atomic_number

    def forward(self, data):
        data = add_chemprop_features(data, self.one_hot, self.max_atomic_number)

        return data


class AddNumNodes(T.BaseTransform):
    def forward(self, data):
        if data is not None:
            data.num_nodes = data.x.shape[0]
        return data


class AddMaxEdge(T.BaseTransform):
    def forward(self, data):
        if data is not None:
            if data.edge_index.numel() > 0:
                data.max_edge = torch.tensor(data.edge_index.shape[-1]).unsqueeze(0)
            else:
                return None

        return data


class AddMaxNode(T.BaseTransform):
    def forward(self, data):
        if data is not None:
            data.max_node = torch.tensor(data.num_nodes).unsqueeze(0)

        return data


class AddMaxEdgeGlobal(T.BaseTransform):
    def __init__(self, max_edge: int):
        self.max_edge = max_edge

    def forward(self, data):
        data.max_edge_global = self.max_edge

        return data


class AddMaxNodeGlobal(T.BaseTransform):
    def __init__(self, max_node: int):
        self.max_node = max_node

    def forward(self, data):
        data.max_node_global = self.max_node

        return data


class AddGlobalFeatures(T.BaseTransform):
    def forward(self, data):
        if data is not None:
            data.global_feat = torch.tensor([data.max_node, data.max_edge], dtype=torch.float)
        return data


class AddPosEnc(T.BaseTransform):
    def __init__(self, pe_types):
        self.pe_types = pe_types

    def forward(self, data):
        return compute_posenc_stats(data, pe_types=self.pe_types, is_undirected=True)


class FormatSingleLabel(T.BaseTransform):
    def forward(self, data):
        if data is None:
            return data

        if data.y.ndim == 0:
            data.y = data.y.unsqueeze(0)
        elif data.y.ndim == 2:
            data.y = data.y.squeeze(1)

        return data


class LabelNanToZero(T.BaseTransform):
    def forward(self, data):
        if data is None:
            return data

        data.y = torch.nan_to_num(data.y, nan=0.0)

        return data
