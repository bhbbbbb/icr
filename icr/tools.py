import os

def is_on_kaggle():
    """determine whether the environment is on kaggle."""
    return os.environ.get('KAGGLE_KERNEL_RUN_TYPE') is not None
