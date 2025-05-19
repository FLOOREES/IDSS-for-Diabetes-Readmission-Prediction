# src/run.py
"""
run.py
======

CLI wrapper to execute the pipeline with tunable hyperparameters.

Usage
-----
$ python -m src.run                                  # Use config defaults
$ python -m src.run --train-ae                       # Force AE retraining
$ python -m src.run --train-predictor                # Force Predictor retraining
$ python -m src.run --train-ae --train-predictor     # Force both
$ python -m src.run --ae-epochs 50 --ae-batch-size 128 # Override specific AE params
$ python -m src.run --hidden-dim 256 --use-attention # Override model architecture and use attention
$ python -m src.run --help                           # Show help message
"""

import argparse
import logging
import sys
import inspect # Used to check if attribute exists in config

from . import config
from .pipeline import Pipeline

# Define the list of tuneable parameters and their types
# This list is used to automatically add arguments to argparse and override config
TUNEABLE_PARAMS = {
    'DATALOADER_NUM_WORKERS': int,
    'HIDDEN_DIM': int,
    'NUM_RNN_LAYERS': int,
    'DROPOUT': float,
    # Note: USE_GRU and USE_ATTENTION are best handled with action='store_true'
    # We'll handle them slightly separately in the argument parsing setup.
    'ATTENTION_DIM': int,
    'DIAGNOSIS_EMBEDDING_DIM': int,
    'DIAGNOSIS_TSNE_COMPONENTS': int,
    'GRADIENT_CLIP_VALUE': float,
    'AE_EPOCHS': int,
    'AE_BATCH_SIZE': int,
    'AE_LEARNING_RATE': float,
    'AE_WEIGHT_DECAY': float,
    'AE_OPTIMIZER': str,
    'AE_SCHEDULER_PATIENCE': int,
    'AE_SCHEDULER_FACTOR': float,
    'AE_EARLY_STOPPING_PATIENCE': int,
    'PREDICTOR_EPOCHS': int,
    'PREDICTOR_BATCH_SIZE': int,
    'PREDICTOR_LEARNING_RATE': float,
    'PREDICTOR_WEIGHT_DECAY': float,
    'PREDICTOR_OPTIMIZER': str,
    'PREDICTOR_SCHEDULER_PATIENCE': int,
    'PREDICTOR_SCHEDULER_FACTOR': float,
    'PREDICTOR_EARLY_STOPPING_PATIENCE': int,
}


def _cli_args() -> argparse.Namespace:
    """
    Parse command-line flags and hyperparameters that may override
    `config.py` defaults.
    """
    parser = argparse.ArgumentParser(
        description="Diabetes Readmission Sequential Modeling Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )

    # Add flags for training control
    parser.add_argument(
        "--train-ae",
        action="store_true",
        default=None, # Use None to distinguish between "not specified" and "specified as False"
        help="(Re)train the autoencoder. Overrides TRAIN_AE in config if specified."
    )
    parser.add_argument(
        "--train-predictor",
        action="store_true",
        default=None, # Use None to distinguish between "not specified" and "specified as False"
        help="(Re)train the predictor. Overrides TRAIN_PREDICTOR in config if specified."
    )

    # Add arguments for tuneable parameters based on the TUNEABLE_PARAMS dict
    for param_name, param_type in TUNEABLE_PARAMS.items():
        # Convert SCREAMING_SNAKE_CASE to kebab-case for CLI arguments
        arg_name = f"--{param_name.lower().replace('_', '-')}"

        # Get the default value from config.py for the help string
        # Use getattr with a default in case a parameter is in TUNEABLE_PARAMS
        # but somehow missing from config (though it shouldn't happen).
        config_default = getattr(config, param_name, None)

        parser.add_argument(
            arg_name,
            type=param_type,
            default=None, # Use None to indicate argument was not provided
            help=f"Override '{param_name}' from config. Default: {config_default}"
        )

    # Handle specific boolean flags with store_true/store_false actions
    # These are not easily handled by the generic TUNEABLE_PARAMS dict approach
    # if we want explicit --use-gru and --no-use-gru flags.
    # If we only want --use-gru (sets to True), we could add to TUNEABLE_PARAMS
    # with type=bool and use action='store_true', but default=None is tricky there.
    # Explicit handling is clearer for toggle flags.
    parser.add_argument(
        "--use-gru",
        action="store_true",
        default=None, # None means "use config default"
        help=f"Override 'USE_GRU' in config to True. Default: {getattr(config, 'USE_GRU', False)}"
    )
    parser.add_argument(
        "--no-use-gru",
        dest="use_gru", # Store in the same attribute as --use-gru
        action="store_false", # Sets use_gru to False
        help=f"Override 'USE_GRU' in config to False."
    )
    parser.add_argument(
        "--use-attention",
        action="store_true",
        default=None, # None means "use config default"
        help=f"Override 'USE_ATTENTION' in config to True. Default: {getattr(config, 'USE_ATTENTION', True)}"
    )
    parser.add_argument(
        "--no-use-attention",
        dest="use_attention", # Store in the same attribute as --use-attention
        action="store_false", # Sets use_attention to False
        help=f"Override 'USE_ATTENTION' in config to False."
    )


    # You can add argument groups for better organization in --help
    # model_group = parser.add_argument_group('Model Hyperparameters')
    # model_group.add_argument(...)
    # training_group = parser.add_argument_group('Training Hyperparameters')
    # training_group.add_argument(...)


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

    # --- Override config based on CLI arguments ---
    logger.info("Checking for config overrides from CLI arguments...")

    # Handle explicit boolean flags first (TRAIN_AE, TRAIN_PREDICTOR, USE_GRU, USE_ATTENTION)
    # These use default=None and store_true/store_false
    flags_to_check = {
        'train_ae': 'TRAIN_AE',
        'train_predictor': 'TRAIN_PREDICTOR',
        'use_gru': 'USE_GRU',
        'use_attention': 'USE_ATTENTION',
    }
    for arg_name, config_var_name in flags_to_check.items():
        arg_value = getattr(args, arg_name)
        # Only override if the argument was explicitly provided (value is not None)
        if arg_value is not None:
            current_config_value = getattr(config, config_var_name, None)
            # Use inspect.isabstract(config) or hasattr(config, config_var_name) for robustness
            # Although with from . import config, attributes are usually directly accessible.
            # hasattr is safer if config attributes might be added/removed dynamically.
            if hasattr(config, config_var_name):
                 logger.info(f"CLI override: Setting config.{config_var_name} from '{current_config_value}' to '{arg_value}'")
                 setattr(config, config_var_name, arg_value)
            else:
                 logger.warning(f"Attempted to override config.{config_var_name} via CLI, but '{config_var_name}' not found in config.py. Ignoring.")
        # If arg_value is None, the config default or previous value is kept.

    # Handle generic tuneable parameters
    for arg_name, arg_value in vars(args).items():
        # Skip the boolean flags we handled explicitly
        if arg_name in flags_to_check.keys():
            continue

        # Check if the argument was provided (i.e., its value is not None)
        if arg_value is not None:
            # Map argument name (snake_case) back to config variable name (SCREAMING_SNAKE_CASE)
            config_var_name = arg_name.upper() # Simple mapping, assumes _ matches -

            # Check if this attribute actually exists in the config module
            if hasattr(config, config_var_name):
                current_config_value = getattr(config, config_var_name)
                logger.info(f"CLI override: Setting config.{config_var_name} from '{current_config_value}' to '{arg_value}' (type: {type(arg_value).__name__})")
                # Dynamically set the attribute on the config module
                setattr(config, config_var_name, arg_value)
            else:
                # This should not happen if TUNEABLE_PARAMS list is correct
                logger.warning(f"CLI argument '--{arg_name}' corresponds to config variable '{config_var_name}', but '{config_var_name}' not found in config.py. Ignoring this override.")


    # At this point, the config module has been updated with any CLI overrides.
    logger.info("Config overrides applied. Final effective configuration will be used by the pipeline.")
    # You could optionally log the final state of relevant config variables here.

    logger.info("Initializing and executing the pipeline...")
    pipeline_instance = Pipeline(cfg=config) # Pass the potentially modified config module
    pipeline_instance.run()
    logger.info("Pipeline execution finished. CLI runner exiting.")


if __name__ == "__main__":
    main()