"""
Test the project initialization.
"""

from projinit import config
from projinit import platform_init

if __name__ == "__main__":
    # Initialize the project
    test_config = config.Config()
    # Print the configuration
    platform_init.InitializePlatforms().login_accounts()
    # Create the project information
    platform_init.ProjectInfo().create_wandb(entity="ICML25-plan")
