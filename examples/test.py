"""
Test the project initialization.
"""

from projinit import config
from projinit import platform_init

if __name__ == "__main__":
    # Initialize the project
    test_config = config.Config()
    # Set the platforms
    platform_init.InitializePlatforms().login_accounts()
    # Create the project information
    proj_info = platform_init.ProjectInfo()
    wandb_run = proj_info.create_wandb(entity="ICML25-plan")
