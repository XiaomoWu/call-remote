import sys
sys.dont_write_bytecode = True

import os
import dotenv
import hydra

from omegaconf import DictConfig

# Load environment variables from `.env` file if it exists
# Recursively searches for `.env` in all folders starting from WORK_DIR
# For system specific variables (like data paths) it's better to use .env file!
dotenv.load_dotenv(override=True)
os.chdir(os.getenv("WORK_DIR"))

@hydra.main(config_path="configs/", config_name="config.yaml", version_base='1.2')
def main(cfg: DictConfig):

    # -----------
    # Collect cfg
    # -----------
    # Imports should be nested inside @hydra.main to optimize tab completion
    from src.train import train

    # --------------------
    # Run train() with cfg
    # --------------------

    return train(cfg)


if __name__ == "__main__":
    main()
