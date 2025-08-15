import pandas as pd
import numpy as np

def inspect_data():
    """
    Perform initial data inspection on the df_players DataFrame
    """
    try:
        # Try to load the DataFrame from pickle first
        df_players = pd.read_pickle("df_players.pkl")
        print("DataFrame loaded from pickle file.")
    except FileNotFoundError:
        print("Pickle file not found. Please run database_setup.py first.")
        return
    
    print("=" * 60)
    print("PHASE 1.5: INITIAL DATA INSPECTION")
    print("=" * 60)
    
    # Display first 5 rows
    print("\n1. First 5 rows of df_players:")
    print("-" * 40)
    print(df_players.head())
    
    # Display basic information
    print("\n2. DataFrame Info:")
    print("-" * 40)
    print(df_players.info())
    
    # Display basic statistics
    print("\n3. Basic Statistics for Numerical Columns:")
    print("-" * 40)
    print(df_players.describe())
    
    # Check for missing values
    print("\n4. Missing Values Analysis:")
    print("-" * 40)
    missing_values = df_players.isnull().sum()
    if missing_values.sum() == 0:
        print("No missing values found in the dataset!")
    else:
        print("Columns with missing values:")
        for col, count in missing_values.items():
            if count > 0:
                print(f"  {col}: {count} missing values")
    
    # Display dataset shape
    print(f"\n5. Dataset Shape: {df_players.shape}")
    print(f"   - Rows: {df_players.shape[0]}")
    print(f"   - Columns: {df_players.shape[1]}")
    
    # Display column names
    print(f"\n6. Column Names:")
    print("-" * 40)
    for i, col in enumerate(df_players.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print("\n" + "=" * 60)
    print("Data inspection completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    inspect_data()
