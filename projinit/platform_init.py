"""
Initialize the running environment with various platforms.
"""

import os
import logging

import wandb
from dotenv import load_dotenv
from huggingface_hub import login
from accelerate import Accelerator

from projinit.config import Config


class InitializePlatforms:
    """Initialize the running environment of various platform."""

    def __init__(self):
        # Get the platform environments
        self.env_config = Config.items_to_dict(Config().env._asdict())
        self.dotenv_path = self.env_config["dotenv_path"]
        load_dotenv(self.dotenv_path)

        # Present mandatory platforms
        logging.info("Under %s, set:", self.dotenv_path)
        logging.info("  - For HuggingFace: 'HUGGINGFACE_TOKEN'")
        logging.info("  - For Wandb:       'WANDB_KEY'")

        # Check whether HUGGINGFACE_TOKEN in the environment
        assert os.environ.get("HUGGINGFACE_TOKEN") is not None
        assert os.environ.get("WANDB_KEY") is not None

        self.platforms = {
            "HuggingFace": os.environ.get("HUGGINGFACE_TOKEN"),
            "Wandb": os.environ.get("WANDB_KEY"),
        }

    def login_accounts(self):
        """
        Initialize the accounts for the project.
        """
        # Login to Hugging Face
        if Accelerator().is_local_main_process:
            login(token=os.environ.get("HUGGINGFACE_TOKEN"))
            # Login to Wandb
            wandb.login(key=os.environ.get("WANDB_KEY"))


class ProjectInfo:
    """Create the project information for wandb platforms."""

    def __init__(self):
        self.base_config = Config.params
        self.env_config = Config.items_to_dict(Config.env._asdict())
        self.data_config = Config.items_to_dict(Config.data._asdict())
        self.model_config = Config.items_to_dict(Config.model._asdict())
        self.train_config = Config.items_to_dict(Config.train._asdict())
        self.eval_config = Config.items_to_dict(Config.eval._asdict())
        self.log_config = Config.items_to_dict(Config.logging._asdict())

    def create_wandb(self, entity: str):
        """Create the project information for wandb platform."""
        # Set the same project as the local saving
        base_name = os.path.basename(self.base_config["base_path"])
        project = self.base_config["project_name"]
        if entity != base_name:
            project = f"{base_name}---{project}"
        if Accelerator().is_local_main_process:
            wandb_run = wandb.init(
                entity=entity,
                project=project,
                name=self.base_config["exe_id"],
            )
            # Added the record file to the wandb
            artifact = wandb.Artifact("experiment_info", type="info")
            artifact.add_file(self.base_config["record_path"])
            wandb_run.log_artifact(artifact)
        else:
            os.environ["WANDB_ENTITY"] = entity
            os.environ["WANDB_PROJECT"] = project
            os.environ["WANDB_NAME"] = self.base_config["exe_id"]
            wandb_run = None

        return wandb_run

    def close_wandb(self, wandb_run):
        """Close the wandb in the main process."""
        if Accelerator().is_local_main_process:
            wandb_run.finish()
