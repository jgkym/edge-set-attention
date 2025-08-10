import joblib
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate.utils import TorchDynamoPlugin
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from esa.config import get_config
from esa.data_loading import load_molecule_dataset
from esa.model import MoleculePropertyPredictor
from esa.utils import set_seed

def pIC50_to_IC50(pIC50):
    return 10 ** (9 - pIC50)

def inference():
    """
    Run inference on the test dataset, generate predictions,
    and save the results to a submission CSV file.
    """
    config = get_config()

    # --- Seed Setting ---
    set_seed(config.random_seed)

    # --- Accelerator ---
    # Initialize accelerator with performance optimizations
    # torch_dynamo_plugin = TorchDynamoPlugin(
    #         backend="inductor",
    #         mode="max-autotune",
    #         use_regional_compilation=True,
    # ) if hasattr(torch, "compile") else None

    accelerator = Accelerator(
        # mixed_precision="fp16",  # Use "bf16" on A100/H100, "fp16" on others
        # dynamo_plugin=torch_dynamo_plugin,
    )

    print("Starting inference...")
    # Let accelerate handle the device printing
    accelerator.print(f"Using device: {accelerator.device}")

    test_dataset, node_dim, edge_dim = load_molecule_dataset(
        data_dir=config.data.data_dir,
        filename="test.csv",
        smiles_col="Smiles",
        target_col=None,
        one_hot=config.data.one_hot,
        max_atomic_number=config.data.max_atomic_number,
        pe_types=config.data.pe_types,
        train_ratio=None,
        is_test=True,
    )
    config.model.embedding.node_dim = node_dim
    config.model.embedding.edge_dim = edge_dim

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
    )

    # --- Model Loading ---
    # BEST PRACTICE: Load state_dict before preparing the model
    best_model_path = config.training.output_dir / "best_model_state"
    accelerator.print(f"Loading model from {best_model_path}...")
    model = MoleculePropertyPredictor(config.model)

    # --- Prepare for acceleration ---
    # This will move the model to the correct device and compile it with TorchDynamo
    model, test_loader = accelerator.prepare(model, test_loader)
    accelerator.load_state(best_model_path)

    model.eval()
    all_predictions = []

    # --- Inference Loop ---
    with torch.no_grad():
        progress_bar = tqdm(
            test_loader, desc="Inference", disable=not accelerator.is_main_process
        )
        for inputs in progress_bar:
            # NO NEED for inputs.to(device). `accelerate` handles it.
            outputs = model(inputs)

            # Gather predictions from all processes
            gathered_predictions = accelerator.gather(outputs)
            all_predictions.append(gathered_predictions.cpu())

    # --- Process and Save Results ---
    # This block will only run on the main process to avoid duplicate work
    if accelerator.is_main_process:
        print("Processing and saving results...")
        # Concatenate all batches of predictions
        final_predictions = torch.cat(all_predictions)

        print(f"final_predictions.shape: {final_predictions.shape}")

        # Now you can process `final_predictions` and save to a CSV
        # e.g., create_submission_file(final_predictions, config.submission_path)
        submission = pd.read_csv(config.data.data_dir / "sample_submission.csv")
        
        y_scaler = joblib.load(config.data.data_dir / "y_scaler.joblib")

        inverse_scaled = y_scaler.inverse_transform(final_predictions.numpy().reshape(-1, 1))
        ic50s = [pIC50_to_IC50(p).item() for p in inverse_scaled]
        
        submission["ASK1_IC50_nM"] = ic50s
        submission.to_csv(config.training.output_dir / "output.csv", index=False)
        

    accelerator.wait_for_everyone()
    print("Inference completed.")


if __name__ == "__main__":
    inference()
