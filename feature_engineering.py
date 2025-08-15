#!/usr/bin/env python3
"""
Phase 3: Feature Engineering
This script creates new insightful features and maps them to Bartle's player taxonomy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_cleaned_data():
    """Load the cleaned and preprocessed DataFrame"""
    try:
        df_players = pd.read_pickle("df_players_cleaned.pkl")
        print("✅ Cleaned DataFrame loaded successfully")
        print(f"Shape: {df_players.shape}")
        return df_players
    except FileNotFoundError:
        print("❌ Cleaned DataFrame not found. Please run data_cleaning_preprocessing.py first.")
        return None

def create_total_weekly_playtime(df):
    """
    Prompt 3.1: Create TotalWeeklyPlaytime feature
    """
    print("\n" + "="*60)
    print("PROMPT 3.1: CREATING TOTALWEEKLYPLAYTIME FEATURE")
    print("="*60)
    
    # Check if required columns exist
    required_cols = ['SessionsPerWeek', 'AvgSessionDurationMinutes']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return df
    
    # Create the new feature
    df['TotalWeeklyPlaytime'] = df['SessionsPerWeek'] * df['AvgSessionDurationMinutes']
    
    print("✅ TotalWeeklyPlaytime feature created successfully!")
    print(f"Feature calculation: SessionsPerWeek × AvgSessionDurationMinutes")
    
    # Display statistics of the new feature
    print(f"\nTotalWeeklyPlaytime Statistics:")
    print(f"  Mean: {df['TotalWeeklyPlaytime'].mean():.2f} minutes per week")
    print(f"  Median: {df['TotalWeeklyPlaytime'].median():.2f} minutes per week")
    print(f"  Min: {df['TotalWeeklyPlaytime'].min():.2f} minutes per week")
    print(f"  Max: {df['TotalWeeklyPlaytime'].max():.2f} minutes per week")
    print(f"  Std: {df['TotalWeeklyPlaytime'].std():.2f} minutes per week")
    
    # Convert to hours for better interpretation
    df['TotalWeeklyPlaytimeHours'] = df['TotalWeeklyPlaytime'] / 60
    print(f"\nTotalWeeklyPlaytimeHours Statistics:")
    print(f"  Mean: {df['TotalWeeklyPlaytimeHours'].mean():.2f} hours per week")
    print(f"  Median: {df['TotalWeeklyPlaytimeHours'].median():.2f} hours per week")
    
    # Explain the value of this feature
    print("\n" + "-"*50)
    print("WHY TOTALWEEKLYPLAYTIME IS VALUABLE:")
    print("-"*50)
    print("1. HOLISTIC TIME COMMITMENT:")
    print("   - Combines frequency (sessions) and duration (session length)")
    print("   - Provides total weekly gaming investment")
    print("   - Better than separate metrics for understanding engagement")
    
    print("\n2. PLAYER BEHAVIOR INSIGHTS:")
    print("   - High frequency + short sessions = casual players")
    print("   - Low frequency + long sessions = binge players")
    print("   - High frequency + long sessions = hardcore players")
    
    print("\n3. CLUSTERING VALUE:")
    print("   - Single metric representing time commitment")
    print("   - Useful for segmenting players by engagement level")
    print("   - Correlates with other behavioral patterns")
    
    return df

def create_achievement_rate(df):
    """
    Prompt 3.2: Create AchievementRate feature
    """
    print("\n" + "="*60)
    print("PROMPT 3.2: CREATING ACHIEVEMENTRATE FEATURE")
    print("="*60)
    
    # Check if required columns exist
    required_cols = ['AchievementsUnlocked', 'PlayerLevel']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        return df
    
    # Check for potential division by zero
    zero_level_players = (df['PlayerLevel'] == 0).sum()
    print(f"Players with PlayerLevel = 0: {zero_level_players}")
    
    # Create the feature with division by zero handling
    df['AchievementRate'] = df['AchievementsUnlocked'] / df['PlayerLevel']
    
    # Handle division by zero and infinite values
    # Replace inf with 0 and NaN with 0
    df['AchievementRate'] = df['AchievementRate'].replace([np.inf, -np.inf], 0)
    df['AchievementRate'] = df['AchievementRate'].fillna(0)
    
    # Ensure it's a float64 type
    df['AchievementRate'] = df['AchievementRate'].astype('float64')
    
    print("✅ AchievementRate feature created successfully!")
    print(f"Feature calculation: AchievementsUnlocked ÷ PlayerLevel")
    print(f"Division by zero handling: Replaced with 0")
    
    # Display statistics of the new feature
    print(f"\nAchievementRate Statistics:")
    print(f"  Mean: {df['AchievementRate'].mean():.4f}")
    print(f"  Median: {df['AchievementRate'].median():.4f}")
    print(f"  Min: {df['AchievementRate'].min():.4f}")
    print(f"  Max: {df['AchievementRate'].max():.4f}")
    print(f"  Std: {df['AchievementRate'].std():.4f}")
    
    # Analyze the distribution
    print(f"\nAchievementRate Distribution:")
    print(f"  Players with rate = 0: {(df['AchievementRate'] == 0).sum()}")
    print(f"  Players with rate > 0: {(df['AchievementRate'] > 0).sum()}")
    print(f"  Players with rate > 1: {(df['AchievementRate'] > 1).sum()}")
    
    # Explain the value of this feature
    print("\n" + "-"*50)
    print("WHY ACHIEVEMENTRATE IS VALUABLE:")
    print("-"*50)
    print("1. ACHIEVEMENT EFFICIENCY:")
    print("   - Measures achievements per level gained")
    print("   - Higher rate = more achievement-focused playstyle")
    print("   - Lower rate = more level-focused or casual playstyle")
    
    print("\n2. PLAYER MOTIVATION PROXY:")
    print("   - High rate = achievement hunters")
    print("   - Low rate = progression-focused players")
    print("   - Zero rate = new players or non-achievement players")
    
    print("\n3. CLUSTERING VALUE:")
    print("   - Distinguishes achievement-oriented from progression-oriented players")
    print("   - Useful for identifying different player archetypes")
    print("   - Complements other engagement metrics")
    
    return df

def map_to_bartle_taxonomy(df):
    """
    Prompt 3.3: Conceptual mapping to Bartle's Taxonomy
    """
    print("\n" + "="*60)
    print("PROMPT 3.3: MAPPING TO BARTLE'S PLAYER TAXONOMY")
    print("="*60)
    
    print("BARTLE'S PLAYER TYPES AND FEATURE MAPPINGS:")
    print("="*50)
    
    print("\n1. ACHIEVERS (Goal-Oriented Players):")
    print("   - Primary Features:")
    print("     • AchievementRate (HIGH) - achievement hunting efficiency")
    print("     • AchievementsUnlocked (HIGH) - total achievements earned")
    print("     • PlayerLevel (HIGH) - progression milestones")
    print("     • TotalWeeklyPlaytime (MODERATE-HIGH) - consistent engagement")
    print("   - Secondary Indicators:")
    print("     • GameDifficulty (MEDIUM-HARD) - challenging goals")
    print("     • SessionsPerWeek (HIGH) - regular play sessions")
    
    print("\n2. EXPLORERS (Discovery-Oriented Players):")
    print("   - Primary Features:")
    print("     • TotalWeeklyPlaytime (HIGH) - extensive time investment")
    print("     • AvgSessionDurationMinutes (HIGH) - long exploration sessions")
    print("     • SessionsPerWeek (MODERATE) - fewer but longer sessions")
    print("     • GameGenre (Strategy/Simulation) - exploration-heavy genres")
    print("   - Secondary Indicators:")
    print("     • PlayerLevel (MODERATE) - exploration over progression")
    print("     • AchievementRate (MODERATE) - discovery achievements")
    
    print("\n3. SOCIALIZERS (Interaction-Oriented Players):")
    print("   - Primary Features:")
    print("     • SessionsPerWeek (HIGH) - frequent social interactions")
    print("     • AvgSessionDurationMinutes (MODERATE) - social session length")
    print("     • TotalWeeklyPlaytime (MODERATE-HIGH) - social engagement")
    print("   - LIMITATIONS:")
    print("     • No direct social interaction metrics in dataset")
    print("     • Must infer from engagement patterns")
    print("     • Could be mixed with other player types")
    
    print("\n4. KILLERS (Competition-Oriented Players):")
    print("   - Primary Features:")
    print("     • GameDifficulty (HARD) - competitive challenges")
    print("     • PlayerLevel (HIGH) - skill progression")
    print("     • TotalWeeklyPlaytime (HIGH) - competitive practice")
    print("     • InGamePurchases (MODERATE-HIGH) - competitive advantages")
    print("   - LIMITATIONS:")
    print("     • No direct PvP or competitive metrics")
    print("     • Must infer from difficulty and progression patterns")
    print("     • Could overlap with Achievers")
    
    print("\n" + "-"*50)
    print("DATASET LIMITATIONS FOR BARTLE'S TAXONOMY:")
    print("-"*50)
    print("1. SOCIALIZERS:")
    print("   - No chat logs, friend lists, or guild information")
    print("   - No multiplayer session data")
    print("   - Must rely on engagement patterns as proxies")
    
    print("\n2. KILLERS:")
    print("   - No PvP statistics or competitive rankings")
    print("   - No direct aggression or dominance metrics")
    print("   - Must infer from difficulty and progression")
    
    print("\n3. OVERLAP ISSUES:")
    print("   - Players can exhibit multiple archetype characteristics")
    print("   - Pure archetypes are rare in practice")
    print("   - Clustering may reveal hybrid player types")
    
    print("\n4. FEATURE STRENGTHS:")
    print("   - Strong proxies for Achievers and Explorers")
    print("   - Good engagement and progression metrics")
    print("   - Clear behavioral patterns for clustering")
    
    return df

def create_derived_features(df):
    """Create additional derived features for better clustering"""
    print("\n" + "="*60)
    print("CREATING ADDITIONAL DERIVED FEATURES")
    print("="*60)
    
    # Engagement Efficiency (PlayTime per Session)
    df['EngagementEfficiency'] = df['TotalWeeklyPlaytime'] / df['SessionsPerWeek']
    df['EngagementEfficiency'] = df['EngagementEfficiency'].replace([np.inf, -np.inf], 0)
    df['EngagementEfficiency'] = df['EngagementEfficiency'].fillna(0)
    df['EngagementEfficiency'] = df['EngagementEfficiency'].astype('float64')
    
    # Progression Rate (Level per Achievement)
    df['ProgressionRate'] = df['PlayerLevel'] / (df['AchievementsUnlocked'] + 1)  # +1 to avoid division by zero
    df['ProgressionRate'] = df['ProgressionRate'].replace([np.inf, -np.inf], 0)
    df['ProgressionRate'] = df['ProgressionRate'].fillna(0)
    df['ProgressionRate'] = df['ProgressionRate'].astype('float64')
    
    # Purchase Intensity (Purchases per Session)
    df['PurchaseIntensity'] = df['InGamePurchases'] / (df['SessionsPerWeek'] + 1)
    df['PurchaseIntensity'] = df['PurchaseIntensity'].replace([np.inf, -np.inf], 0)
    df['PurchaseIntensity'] = df['PurchaseIntensity'].fillna(0)
    df['PurchaseIntensity'] = df['PurchaseIntensity'].astype('float64')
    
    print("✅ Additional derived features created:")
    print("  • EngagementEfficiency: PlayTime per Session")
    print("  • ProgressionRate: Level per Achievement")
    print("  • PurchaseIntensity: Purchases per Session")
    
    return df

def analyze_feature_correlations(df):
    """Analyze correlations between new and existing features"""
    print("\n" + "="*60)
    print("FEATURE CORRELATION ANALYSIS")
    print("="*60)
    
    # Select numerical features for correlation analysis
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    correlation_matrix = df[numerical_cols].corr()
    
    # Create correlation heatmap
    plt.figure(figsize=(16, 12))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix (Including New Features)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('feature_correlations.png', dpi=300, bbox_inches='tight')
    print("✅ Correlation heatmap saved as 'feature_correlations.png'")
    
    # Show top correlations with new features
    new_features = ['TotalWeeklyPlaytime', 'TotalWeeklyPlaytimeHours', 'AchievementRate', 
                   'EngagementEfficiency', 'ProgressionRate', 'PurchaseIntensity']
    
    print(f"\nTop correlations with new features:")
    for feature in new_features:
        if feature in correlation_matrix.columns:
            correlations = correlation_matrix[feature].abs().sort_values(ascending=False)
            print(f"\n{feature}:")
            for col, corr in correlations[1:6].items():  # Top 5 excluding self
                print(f"  • {col}: {corr:.3f}")
    
    return df

def save_engineered_data(df):
    """Save the DataFrame with new engineered features"""
    print("\n" + "="*60)
    print("SAVING ENGINEERED DATA")
    print("="*60)
    
    # Clean up data types for clustering
    print("Cleaning up data types for clustering...")
    
    # Convert new features to proper numerical types
    new_features = ['TotalWeeklyPlaytime', 'TotalWeeklyPlaytimeHours', 'AchievementRate', 
                   'EngagementEfficiency', 'ProgressionRate', 'PurchaseIntensity']
    
    for feature in new_features:
        if feature in df.columns:
            df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0)
    
    # Ensure all numerical columns are float64
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        df[col] = df[col].astype('float64')
    
    print("✅ Data types cleaned up for clustering")
    
    # Save the enhanced DataFrame
    df.to_pickle("df_players_engineered.pkl")
    print("✅ Enhanced DataFrame saved as 'df_players_engineered.pkl'")
    
    # Create feature engineering summary
    feature_summary = {
        'original_features': 23,
        'new_features_added': 6,
        'total_features': df.shape[1],
        'new_features': [
            'TotalWeeklyPlaytime',
            'TotalWeeklyPlaytimeHours', 
            'AchievementRate',
            'EngagementEfficiency',
            'ProgressionRate',
            'PurchaseIntensity'
        ],
        'feature_descriptions': {
            'TotalWeeklyPlaytime': 'Total weekly gaming time in minutes',
            'TotalWeeklyPlaytimeHours': 'Total weekly gaming time in hours',
            'AchievementRate': 'Achievements unlocked per player level',
            'EngagementEfficiency': 'Play time per session',
            'ProgressionRate': 'Level progression per achievement',
            'PurchaseIntensity': 'In-game purchases per session'
        }
    }
    
    import json
    with open('feature_engineering_summary.json', 'w') as f:
        json.dump(feature_summary, f, indent=2)
    print("✅ Feature engineering summary saved as 'feature_engineering_summary.json'")
    
    return feature_summary

def main():
    """Main execution function"""
    print("🚀 PHASE 3: FEATURE ENGINEERING")
    print("=" * 60)
    
    # Load the cleaned data
    df_players = load_cleaned_data()
    if df_players is None:
        return
    
    print(f"Original shape: {df_players.shape}")
    
    # Step 1: Create TotalWeeklyPlaytime
    df_players = create_total_weekly_playtime(df_players)
    
    # Step 2: Create AchievementRate
    df_players = create_achievement_rate(df_players)
    
    # Step 3: Map to Bartle's Taxonomy
    df_players = map_to_bartle_taxonomy(df_players)
    
    # Step 4: Create additional derived features
    df_players = create_derived_features(df_players)
    
    # Step 5: Analyze feature correlations
    df_players = analyze_feature_correlations(df_players)
    
    # Step 6: Save the engineered data
    feature_summary = save_engineered_data(df_players)
    
    print("\n" + "🎉 PHASE 3 COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nFeature Engineering Results:")
    print(f"  Original features: {feature_summary['original_features']}")
    print(f"  New features added: {feature_summary['new_features_added']}")
    print(f"  Total features: {feature_summary['total_features']}")
    print(f"  Final shape: {df_players.shape}")
    
    print("\nNext steps:")
    print("- Enhanced data saved as 'df_players_engineered.pkl'")
    print("- Feature engineering summary saved as 'feature_engineering_summary.json'")
    print("- Correlation analysis saved as 'feature_correlations.png'")
    print("- Ready for Phase 4: Optimal Cluster Determination!")

if __name__ == "__main__":
    main()
