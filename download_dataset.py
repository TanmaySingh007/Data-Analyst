import requests
import os

def download_dataset():
    """
    Download the Predict Online Gaming Behavior Dataset from Kaggle
    Note: This is a placeholder. You'll need to manually download the file
    from https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset
    and place it in the data/ directory as 'online_gaming_behavior.csv'
    """
    print("Please manually download the dataset from:")
    print("https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset")
    print("Save it as 'online_gaming_behavior.csv' in the data/ directory")
    
    # Check if file exists
    file_path = "data/online_gaming_behavior.csv"
    if os.path.exists(file_path):
        print(f"Dataset found at {file_path}")
        return True
    else:
        print(f"Dataset not found at {file_path}")
        return False

if __name__ == "__main__":
    download_dataset()
