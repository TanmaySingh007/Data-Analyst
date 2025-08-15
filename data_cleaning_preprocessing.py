#!/usr/bin/env python3
"""
Phase 2: Data Cleaning and Preprocessing
This script handles missing values, categorical encoding, feature scaling, and outlier analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_data():
    """Load the preprocessed DataFrame"""
    try:
        df_players = pd.read_pickle("df_players.pkl")
        print("✅ DataFrame loaded successfully")
        print(f"Shape: {df_players.shape}")
        return df_players
    except FileNotFoundError:
        print("❌ DataFrame not found. Please run database_setup.py first.")
        return None

def handle_missing_values(df):
    """
    Prompt 2.1: Handle missing values using median for numerical and mode for categorical
    """
    print("\n" + "="*60)
    print("PROMPT 2.1: HANDLING MISSING VALUES")
    print("="*60)
    
    # Check for missing values
    missing_values = df.isnull().sum()
    total_missing = missing_values.sum()
    
    if total_missing == 0:
        print("✅ No missing values found in the dataset!")
        return df
    
    print(f"Found {total_missing} missing values across the dataset:")
    for col, count in missing_values.items():
        if count > 0:
            print(f"  {col}: {count} missing values")
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Handle numerical columns with median imputation
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df_clean[col].isnull().sum() > 0:
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            print(f"  Filled {col} missing values with median: {median_val}")
    
    # Handle categorical columns with mode imputation
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_val = df_clean[col].mode()[0]
            df_clean[col].fillna(mode_val, inplace=True)
            print(f"  Filled {col} missing values with mode: {mode_val}")
    
    # Verify no missing values remain
    remaining_missing = df_clean.isnull().sum().sum()
    if remaining_missing == 0:
        print("✅ All missing values have been successfully handled!")
    else:
        print(f"❌ Warning: {remaining_missing} missing values still remain")
    
    return df_clean

def one_hot_encode_categorical(df):
    """
    Prompt 2.2: Perform one-hot encoding for categorical features
    """
    print("\n" + "="*60)
    print("PROMPT 2.2: ONE-HOT ENCODING CATEGORICAL FEATURES")
    print("="*60)
    
    # Identify categorical columns to encode
    categorical_cols = ['Gender', 'Location', 'GameGenre', 'GameDifficulty']
    
    print("Categorical columns to encode:")
    for col in categorical_cols:
        if col in df.columns:
            unique_values = df[col].nunique()
            print(f"  {col}: {unique_values} unique values")
    
    # Perform one-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
    
    print(f"\n✅ One-hot encoding completed!")
    print(f"Original shape: {df.shape}")
    print(f"Encoded shape: {df_encoded.shape}")
    print(f"New columns added: {df_encoded.shape[1] - df.shape[1]}")
    
    # Show sample of encoded columns
    print("\nSample of new encoded columns:")
    encoded_cols = [col for col in df_encoded.columns if any(prefix in col for prefix in categorical_cols)]
    print(f"  {len(encoded_cols)} encoded columns created")
    for i, col in enumerate(encoded_cols[:10]):  # Show first 10
        print(f"    {i+1:2d}. {col}")
    if len(encoded_cols) > 10:
        print(f"    ... and {len(encoded_cols) - 10} more")
    
    # Explain why one-hot encoding is necessary
    print("\n" + "-"*50)
    print("WHY ONE-HOT ENCODING IS NECESSARY:")
    print("-"*50)
    print("1. Distance-based algorithms (K-Means, DBSCAN) require numerical input")
    print("2. Prevents incorrect ordinal relationships between categories")
    print("3. Each category becomes a binary feature (0 or 1)")
    print("4. Ensures equal weight for all categorical values")
    print("5. Maintains the categorical nature without implying order")
    
    return df_encoded

def scale_numerical_features(df):
    """
    Prompt 2.3: Apply StandardScaler to numerical features
    """
    print("\n" + "="*60)
    print("PROMPT 2.3: FEATURE SCALING NUMERICAL FEATURES")
    print("="*60)
    
    # Identify numerical columns to scale (exclude PlayerID and encoded columns)
    exclude_cols = ['PlayerID']
    encoded_prefixes = ['Gender_', 'Location_', 'GameGenre_', 'GameDifficulty_']
    
    # Find columns that start with encoded prefixes
    encoded_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in encoded_prefixes)]
    exclude_cols.extend(encoded_cols)
    
    # Get numerical columns for scaling
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    scaling_cols = [col for col in numerical_cols if col not in exclude_cols]
    
    print("Numerical columns to scale:")
    for col in scaling_cols:
        print(f"  {col}")
    
    print(f"\nColumns excluded from scaling:")
    for col in exclude_cols:
        print(f"  {col}")
    
    # Apply StandardScaler
    scaler = StandardScaler()
    df_scaled = df.copy()
    
    # Scale the selected numerical columns
    df_scaled[scaling_cols] = scaler.fit_transform(df[scaling_cols])
    
    print(f"\n✅ Feature scaling completed!")
    print(f"Scaled columns: {len(scaling_cols)}")
    
    # Show scaling statistics
    print("\nScaling statistics (before vs after):")
    for col in scaling_cols:
        before_mean = df[col].mean()
        before_std = df[col].std()
        after_mean = df_scaled[col].mean()
        after_std = df_scaled[col].std()
        print(f"  {col}:")
        print(f"    Before - Mean: {before_mean:.2f}, Std: {before_std:.2f}")
        print(f"    After  - Mean: {after_mean:.2f}, Std: {after_std:.2f}")
    
    # Explain importance of feature scaling
    print("\n" + "-"*50)
    print("IMPORTANCE OF FEATURE SCALING:")
    print("-"*50)
    print("1. Distance-based algorithms are sensitive to feature scales")
    print("2. Features with larger ranges dominate cluster formation")
    print("3. StandardScaler ensures all features have mean=0, std=1")
    print("4. Prevents bias towards features with larger numerical ranges")
    print("5. Essential for K-Means and DBSCAN clustering algorithms")
    
    return df_scaled, scaler

def analyze_outliers(df):
    """
    Prompt 2.4: Perform outlier analysis on scaled numerical features
    """
    print("\n" + "="*60)
    print("PROMPT 2.4: OUTLIER ANALYSIS")
    print("="*60)
    
    # Identify numerical columns for outlier analysis
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    exclude_cols = ['PlayerID']
    encoded_prefixes = ['Gender_', 'Location_', 'GameGenre_', 'GameDifficulty_']
    encoded_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in encoded_prefixes)]
    exclude_cols.extend(encoded_cols)
    
    analysis_cols = [col for col in numerical_cols if col not in exclude_cols]
    
    print("Columns for outlier analysis:")
    for col in analysis_cols:
        print(f"  {col}")
    
    # Create outlier analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Outlier Analysis: Box Plots and Distributions', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes_flat = axes.flatten()
    
    for i, col in enumerate(analysis_cols[:6]):  # Show first 6 columns
        if i < len(axes_flat):
            ax = axes_flat[i]
            
            # Box plot
            ax.boxplot(df[col], vert=True)
            ax.set_title(f'{col} - Box Plot', fontweight='bold')
            ax.set_ylabel('Value')
            
            # Add statistics
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
            outlier_count = len(outliers)
            
            ax.text(0.02, 0.98, f'Outliers: {outlier_count}\nIQR: {iqr:.2f}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Hide unused subplots
    for i in range(len(analysis_cols), len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('outlier_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Outlier analysis plot saved as 'outlier_analysis.png'")
    
    # Statistical outlier analysis using Z-scores
    print("\nStatistical Outlier Analysis (Z-scores):")
    print("-" * 40)
    
    for col in analysis_cols:
        z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
        outliers_z = df[z_scores > 3][col]
        outlier_percentage = (len(outliers_z) / len(df)) * 100
        
        print(f"\n{col}:")
        print(f"  Total outliers (|Z| > 3): {len(outliers_z)} ({outlier_percentage:.2f}%)")
        if len(outliers_z) > 0:
            print(f"  Min outlier value: {outliers_z.min():.2f}")
            print(f"  Max outlier value: {outliers_z.max():.2f}")
    
    # Discuss outlier handling strategies
    print("\n" + "-"*50)
    print("OUTLIER HANDLING STRATEGIES:")
    print("-"*50)
    print("1. CAPPING EXTREME VALUES:")
    print("   - Use percentile-based capping (e.g., 1st and 99th percentiles)")
    print("   - Prevents extreme values from skewing clusters")
    print("   - Maintains data distribution shape")
    
    print("\n2. DATA TRANSFORMATIONS:")
    print("   - Log transformation for right-skewed distributions")
    print("   - Square root transformation for moderate skewness")
    print("   - Box-Cox transformation for optimal normalization")
    
    print("\n3. IMPACT ON K-MEANS CLUSTERING:")
    print("   - Outliers can create single-point clusters")
    print("   - May shift cluster centroids significantly")
    print("   - Consider robust clustering methods (e.g., K-Medoids)")
    print("   - Or use outlier detection before clustering")
    
    return df

def save_cleaned_data(df, scaler):
    """Save the cleaned and preprocessed data"""
    print("\n" + "="*60)
    print("SAVING CLEANED DATA")
    print("="*60)
    
    # Save the cleaned DataFrame
    df.to_pickle("df_players_cleaned.pkl")
    print("✅ Cleaned DataFrame saved as 'df_players_cleaned.pkl'")
    
    # Save the scaler for later use
    import joblib
    joblib.dump(scaler, "feature_scaler.pkl")
    print("✅ Feature scaler saved as 'feature_scaler.pkl'")
    
    # Save a summary of the preprocessing steps
    preprocessing_summary = {
        'original_shape': (40034, 13),
        'cleaned_shape': df.shape,
        'scaled_columns': [col for col in df.columns if col not in ['PlayerID'] and not any(col.startswith(prefix) for prefix in ['Gender_', 'Location_', 'GameGenre_', 'GameDifficulty_'])],
        'encoded_columns': [col for col in df.columns if any(col.startswith(prefix) for prefix in ['Gender_', 'Location_', 'GameGenre_', 'GameDifficulty_'])],
        'preprocessing_steps': [
            'Missing value imputation (median for numerical, mode for categorical)',
            'One-hot encoding for categorical variables',
            'StandardScaler for numerical features',
            'Outlier analysis and documentation'
        ]
    }
    
    import json
    with open('preprocessing_summary.json', 'w') as f:
        json.dump(preprocessing_summary, f, indent=2)
    print("✅ Preprocessing summary saved as 'preprocessing_summary.json'")
    
    return preprocessing_summary

def main():
    """Main execution function"""
    print("🚀 PHASE 2: DATA CLEANING AND PREPROCESSING")
    print("=" * 60)
    
    # Load the data
    df_players = load_data()
    if df_players is None:
        return
    
    # Step 1: Handle missing values
    df_clean = handle_missing_values(df_players)
    
    # Step 2: One-hot encode categorical features
    df_encoded = one_hot_encode_categorical(df_clean)
    
    # Step 3: Scale numerical features
    df_scaled, scaler = scale_numerical_features(df_encoded)
    
    # Step 4: Analyze outliers
    df_final = analyze_outliers(df_scaled)
    
    # Save the cleaned data
    preprocessing_summary = save_cleaned_data(df_final, scaler)
    
    print("\n" + "🎉 PHASE 2 COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("- Cleaned data saved as 'df_players_cleaned.pkl'")
    print("- Feature scaler saved as 'feature_scaler.pkl'")
    print("- Preprocessing summary saved as 'preprocessing_summary.json'")
    print("- Outlier analysis plot saved as 'outlier_analysis.png'")
    print("- Ready for Phase 3: Exploratory Data Analysis!")

if __name__ == "__main__":
    main()
