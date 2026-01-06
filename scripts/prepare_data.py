"""
Script to prepare and preprocess raw data for training.

Converts raw data into the format expected by the training pipeline.
"""

def prepare_data(raw_data_path, output_path):
    """
    Prepare data from raw format to training format.
    
    Args:
        raw_data_path: Path to raw data file
        output_path: Path to save processed data
    """
    # TODO: Load raw data
    # TODO: Clean and format data
    # TODO: Save processed data
    pass

def split_data(data_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split data into train/validation/test sets.
    
    Args:
        data_path: Path to processed data
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
    """
    # TODO: Load data
    # TODO: Split data
    # TODO: Save splits
    pass

if __name__ == "__main__":
    # TODO: Add command-line argument parsing
    # TODO: Run data preparation pipeline
    pass

