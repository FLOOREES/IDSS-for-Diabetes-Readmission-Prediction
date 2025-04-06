# src/utils/helpers.py
import pickle
import joblib
import torch
import os
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

def save_artifact(artifact: Any, path: str):
    """ Saves a Python object using pickle or joblib (for sklearn). """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if path.endswith(".pkl"):
             joblib.dump(artifact, path)
             logger.info(f"Artifact saved to {path} using joblib.")
        elif path.endswith(".pth"): # Assume PyTorch model state_dict or full model
             torch.save(artifact, path)
             logger.info(f"PyTorch artifact saved to {path}.")
        else: # Default to pickle
            with open(path, 'wb') as f:
                pickle.dump(artifact, f)
            logger.info(f"Artifact saved to {path} using pickle.")
    except Exception as e:
        logger.error(f"Failed to save artifact to {path}: {e}")
        raise

def load_artifact(path: str, device: Optional[str] = 'cpu') -> Any:
    """ Loads a Python object from pickle, joblib, or torch. """
    if not os.path.exists(path):
        logger.error(f"Artifact file not found: {path}")
        raise FileNotFoundError(f"Artifact file not found: {path}")
    try:
        if path.endswith(".pkl"):
             artifact = joblib.load(path)
             logger.info(f"Artifact loaded from {path} using joblib.")
             return artifact
        elif path.endswith(".pth"):
             artifact = torch.load(path, map_location=device)
             logger.info(f"PyTorch artifact loaded from {path} to device '{device}'.")
             return artifact
        else:
            with open(path, 'rb') as f:
                artifact = pickle.load(f)
            logger.info(f"Artifact loaded from {path} using pickle.")
            return artifact
    except Exception as e:
        logger.error(f"Failed to load artifact from {path}: {e}")
        raise