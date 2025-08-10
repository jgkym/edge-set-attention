from typing import Dict, Sequence
from os.path import join as pjoin

from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures, Descriptors, Lipinski

# --- Set up pharmacophore feature factory ---
fdef_name = pjoin(RDConfig.RDDataDir, "BaseFeatures.fdef")
factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

# --- Create a mapping from feature family to an integer index ---
FEATURE_FAMILIES = factory.GetFeatureFamilies()
FAMILY_TO_IDX = {family: i for i, family in enumerate(FEATURE_FAMILIES)}


def onek_encoding_unk(value: int, choices: Sequence):
    """
    Creates a one-hot encoding with an extra category for uncommon values.
    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the :code:`value` in a list of length :code:`len(choices) + 1`.
             If :code:`value` is not in :code:`choices`, then the final element in the encoding is 1.
    """
    encoding = [0] * len(choices)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding


def atom_features(
    atom: Chem.rdchem.Atom,
    features_constants: Dict[str, Sequence],
    atom_pharmacophore_features: Dict[int, Sequence[str]],
):
    features = (
        onek_encoding_unk(atom.GetAtomicNum(), features_constants["atomic_num"])
        + onek_encoding_unk(atom.GetTotalDegree(), features_constants["degree"])
        + onek_encoding_unk(atom.GetFormalCharge(), features_constants["formal_charge"])
        + onek_encoding_unk(int(atom.GetChiralTag()), features_constants["chiral_tag"])
        + onek_encoding_unk(int(atom.GetTotalNumHs()), features_constants["num_Hs"])
        + onek_encoding_unk(atom.GetExplicitValence(), features_constants["valence"])
        + onek_encoding_unk(
            int(atom.GetHybridization()), features_constants["hybridization"]
        )
        + [1 if atom.GetIsAromatic() else 0]
    )

    # Add partial charge
    try:
        features += [atom.GetDoubleProp("_GasteigerCharge")]
    except:
        features += [0]

    # Add pharmacophore features
    pharmacophore_features = [0] * len(FEATURE_FAMILIES)
    if atom.GetIdx() in atom_pharmacophore_features:
        for feature_family in atom_pharmacophore_features[atom.GetIdx()]:
            if feature_family in FAMILY_TO_IDX:
                pharmacophore_features[FAMILY_TO_IDX[feature_family]] = 1
    features += pharmacophore_features

    return features


def bond_features(bond: Chem.rdchem.Bond, mol: Chem.rdchem.Mol):
    if bond is None:
        return [1] + [0] * 19  # Adjusted for new feature dimensions

    bt = bond.GetBondType()
    features = [
        0,  # bond is not None
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ]
    features += onek_encoding_unk(int(bond.GetStereo()), list(range(6)))

    # Add new bond features
    features.append(bond.GetBondTypeAsDouble())  # Fractional bond order

    # Rotatable bond
    is_rotatable = (
        1
        if bond.GetBondType() == Chem.rdchem.BondType.SINGLE and not bond.IsInRing()
        else 0
    )
    features.append(is_rotatable)

    # Ring size
    ring_info = mol.GetRingInfo()
    min_ring_size = 0
    if bond.IsInRing():
        min_ring_size = ring_info.MinBondRingSize(bond.GetIdx())

    features += onek_encoding_unk(min_ring_size, [3, 4, 5, 6, 7, 8])

    return features


def global_features(mol: Chem.rdchem.Mol):
    """
    Computes global molecular descriptors.
    """
    return [
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        Lipinski.NumHDonors(mol),
        Lipinski.NumHAcceptors(mol),
        Descriptors.MolWt(mol),
    ]


def get_atom_constants(max_atomic_num: int):
    return {
        "atomic_num": list(range(max_atomic_num)),
        "degree": [0, 1, 2, 3, 4, 5],
        "formal_charge": [-1, -2, 1, 2, 0],
        "chiral_tag": [0, 1, 2, 3],
        "num_Hs": [0, 1, 2, 3, 4],
        "valence": [0, 1, 2, 3, 4, 5, 6],
        "hybridization": [
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
        ],
    }


def atom_features_int(
    atom: Chem.rdchem.Atom,
    features_constants: Dict[str, Sequence],
    atom_pharmacophore_features: Dict[int, Sequence[str]],
):
    features = [
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        int(atom.GetChiralTag()),
        int(atom.GetTotalNumHs()),
        int(atom.GetHybridization()),
        1 if atom.GetIsAromatic() else 0,
    ]
    # Add partial charge
    try:
        features += [atom.GetDoubleProp("_GasteigerCharge")]
    except:
        features += [0]

    # Add pharmacophore features
    pharmacophore_features = [0] * len(FEATURE_FAMILIES)
    if atom.GetIdx() in atom_pharmacophore_features:
        for feature_family in atom_pharmacophore_features[atom.GetIdx()]:
            if feature_family in FAMILY_TO_IDX:
                pharmacophore_features[FAMILY_TO_IDX[feature_family]] = 1
    features += pharmacophore_features

    return features


def bond_features_int(bond: Chem.rdchem.Bond, mol: Chem.rdchem.Mol):
    if bond is None:
        return [1] + [0] * 10  # Adjusted for new feature dimensions

    bt = bond.GetBondType()
    features = [
        0,  # bond is not None
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
        int(bond.GetStereo()),
    ]
    # Add new bond features
    features.append(bond.GetBondTypeAsDouble())

    is_rotatable = (
        1
        if bond.GetBondType() == Chem.rdchem.BondType.SINGLE and not bond.IsInRing()
        else 0
    )
    features.append(is_rotatable)

    min_ring_size = 0
    if bond.IsInRing():
        min_ring_size = mol.GetRingInfo().MinBondRingSize(bond.GetIdx())
    features.append(min_ring_size)

    return features