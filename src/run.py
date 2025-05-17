# src/run.py
"""
run.py
======

Tiny CLI wrapper to execute the pipeline.

Usage
-----
$ python -m src.run                             # Use config defaults (preprocesses if output file missing)
$ python -m src.run --train-ae                  # Force AE retraining
$ python -m src.run --train-predictor           # Force Predictor retraining
$ python -m src.run --train-ae --train-predictor # Force both
"""

import argparse
import logging

from . import config
from .pipeline import Pipeline


def _cli_args() -> argparse.Namespace:
    """Parse command-line flags that may override `config.py` defaults."""
    parser = argparse.ArgumentParser(description="Diabetes Readmission Sequential Modeling Pipeline")
    parser.add_argument(
        "--train-ae",
        action="store_true",
        default=None, 
        help="(Re)train the autoencoder. Overrides TRAIN_AE in config if specified."
    )
    parser.add_argument(
        "--train-predictor",
        action="store_true",
        default=None,
        help="(Re)train the predictor. Overrides TRAIN_PREDICTOR in config if specified."
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the CLI runner.
    Sets up basic logging, parses arguments, updates config, and runs the pipeline.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)-15s : %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logger.info("CLI runner initiated.")

    args = _cli_args()

    # Update config based on CLI arguments
    # These flags in config.py should default to False if not overridden by CLI.
    if args.train_ae is not None: # If the flag was specified by the user
        logger.info(f"CLI override: Setting TRAIN_AE to {args.train_ae}")
        config.TRAIN_AE = args.train_ae
    elif not hasattr(config, 'TRAIN_AE'): # Ensure default if not in config at all
        config.TRAIN_AE = False
    
    if args.train_predictor is not None: # If the flag was specified by the user
        logger.info(f"CLI override: Setting TRAIN_PREDICTOR to {args.train_predictor}")
        config.TRAIN_PREDICTOR = args.train_predictor
    elif not hasattr(config, 'TRAIN_PREDICTOR'): # Ensure default
        config.TRAIN_PREDICTOR = False

    logger.info("Initializing and executing the pipeline...")
    pipeline_instance = Pipeline(cfg=config)
    pipeline_instance.run()
    logger.info("Pipeline execution finished. CLI runner exiting.")


if __name__ == "__main__":
    main()