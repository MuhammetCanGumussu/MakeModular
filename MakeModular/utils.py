"""
Contains various utiliy functions for pytorch model training and saving
"""

import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
              target_dir: str,
              model_name: str):
    """
    Saves a pytorch model to a target directory.

    Args:
        model: a target pytorch model to save
        target_dir: a directory for saving the model to
        mdoel_name: a filename for the saved model. Should include
                    either ".pth" or "pt" as the file extension
                    
    """
    # create target directory
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True,
                         exist_ok=True)

    # create model save path
    assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth' "
    model_save_path = target_dir_path / model_name

    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)

# set seeds
def set_seeds(seed: int=42):
    """
    Sets random seeds for torch operations.

    Args:
        seed (int, optional): random seed to set. Defaults to 42
    """
    # set the seed for general torch operations
    torch.manual_seed(seed)
    # set the seed for cuda torch operations (ops in gpu)
    torch.cuda.manual_seed(seed)
