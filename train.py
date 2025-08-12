from torch_geometric.loader import DataLoader

from esa.config import get_config
from esa.data_loading import load_molecule_dataset
from esa.model import MoleculePropertyPredictor
from esa.trainer import Trainer
from esa.utils import set_seed


def main():
    # --- Load configuration ---
    config = get_config()

    # --- Set random seed ---
    set_seed(config.random_seed)

    # --- Load data ---
    train, val, node_dim, edge_dim = load_molecule_dataset(
        data_dir=config.data.data_dir,
        filename=config.data.filename,
        smiles_col=config.data.smiles_col,
        target_col=config.data.target_col,
        one_hot=config.data.one_hot,
        max_atomic_number=config.data.max_atomic_number,
        pe_types=config.data.pe_types,
        train_ratio=config.data.train_ratio,
    )

    # --- Create data loaders ---
    train_loader = DataLoader(
        train,
        batch_size=config.training.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val,
        batch_size=config.training.batch_size,
        shuffle=False,
        drop_last=True,
    )

    # --- Instantiate model and trainer ---
    config.model.embedding.node_dim = node_dim
    config.model.embedding.edge_dim = edge_dim
    model = MoleculePropertyPredictor(config.model, pe_types=config.data.pe_types)
    trainer = Trainer(
        config=config.training,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
    )

    # --- Train the model ---
    trainer.train()


if __name__ == "__main__":
    main()
