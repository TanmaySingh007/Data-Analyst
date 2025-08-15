#!/usr/bin/env python3
"""
Phase 6: Cluster Interpretation, Visualization, and Power BI Integration
This script analyzes clusters, creates visualizations, and prepares for Power BI integration
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_clustered_data():
    """Load the clustered DataFrame"""
    try:
        df_players = pd.read_pickle("df_players_clustered.pkl")
        print("✅ Clustered DataFrame loaded successfully")
        print(f"Shape: {df_players.shape}")
        return df_players
    except FileNotFoundError:
        print("❌ Clustered DataFrame not found. Please run Phase 5 first.")
        return None

def statistical_characterization_clusters(df):
    """
    Prompt 6.1: Statistical characterization of clusters
    """
    print("\n" + "="*60)
    print("PROMPT 6.1: STATISTICAL CHARACTERIZATION OF CLUSTERS")
    print("="*60)
    
    # Select numerical features for analysis (excluding cluster columns)
    exclude_cols = ['PlayerID', 'KMeans_Cluster', 'DBSCAN_Cluster', 'Cluster_Label', 'Cluster_Description']
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    analysis_cols = [col for col in numerical_cols if col not in exclude_cols]
    
    print(f"Analyzing {len(analysis_cols)} numerical features across 3 clusters...")
    
    # Calculate cluster statistics
    cluster_stats = {}
    for cluster_id in range(3):
        cluster_data = df[df['KMeans_Cluster'] == cluster_id]
        cluster_stats[cluster_id] = {}
        
        for col in analysis_cols:
            cluster_stats[cluster_id][col] = {
                'mean': cluster_data[col].mean(),
                'median': cluster_data[col].median(),
                'std': cluster_data[col].std(),
                'min': cluster_data[col].min(),
                'max': cluster_data[col].max(),
                'count': len(cluster_data)
            }
    
    # Create comprehensive statistics table
    print("\n" + "-"*80)
    print("CLUSTER STATISTICAL PROFILES")
    print("-"*80)
    
    # Header
    header = f"{'Feature':<25} {'Cluster 0 (Casual)':<20} {'Cluster 1 (Moderate)':<20} {'Cluster 2 (Hardcore)':<20}"
    print(header)
    print("-" * 85)
    
    # Display statistics for each feature
    for col in analysis_cols:
        row = f"{col:<25}"
        for cluster_id in range(3):
            mean_val = cluster_stats[cluster_id][col]['mean']
            if 'Time' in col or 'Hours' in col or 'Minutes' in col:
                row += f"{mean_val:>8.1f} ± {cluster_stats[cluster_id][col]['std']:>5.1f}".ljust(20)
            elif 'Rate' in col or 'Score' in col:
                row += f"{mean_val:>8.3f} ± {cluster_stats[cluster_id][col]['std']:>5.3f}".ljust(20)
            else:
                row += f"{mean_val:>8.1f} ± {cluster_stats[cluster_id][col]['std']:>5.1f}".ljust(20)
        print(row)
    
    # Cluster sizes
    print("-" * 85)
    for cluster_id in range(3):
        cluster_data = df[df['KMeans_Cluster'] == cluster_id]
        size = len(cluster_data)
        percentage = (size / len(df)) * 100
        print(f"Cluster {cluster_id} Size: {size:,} players ({percentage:.1f}%)")
    
    # Save detailed statistics to CSV
    detailed_stats = []
    for col in analysis_cols:
        for cluster_id in range(3):
            stats = cluster_stats[cluster_id][col]
            detailed_stats.append({
                'Feature': col,
                'Cluster': cluster_id,
                'Cluster_Name': ['Casual', 'Moderate', 'Hardcore'][cluster_id],
                'Mean': stats['mean'],
                'Median': stats['median'],
                'Std': stats['std'],
                'Min': stats['min'],
                'Max': stats['max'],
                'Count': stats['count']
            })
    
    stats_df = pd.DataFrame(detailed_stats)
    stats_df.to_csv('cluster_statistical_profiles.csv', index=False)
    print(f"\n✅ Detailed statistics saved to 'cluster_statistical_profiles.csv'")
    
    return cluster_stats, analysis_cols

def create_radar_charts(df, analysis_cols):
    """
    Prompt 6.2: Create radar charts for cluster profiles
    """
    print("\n" + "="*60)
    print("PROMPT 6.2: RADAR CHARTS FOR CLUSTER PROFILES")
    print("="*60)
    
    # Select key features for radar charts
    key_features = [
        'PlayTimeHours', 'SessionsPerWeek', 'AvgSessionDurationMinutes',
        'PlayerLevel', 'AchievementsUnlocked', 'InGamePurchases',
        'TotalWeeklyPlaytime', 'AchievementRate', 'EngagementEfficiency'
    ]
    
    # Filter to features that exist in the dataset
    available_features = [f for f in key_features if f in analysis_cols]
    print(f"Creating radar charts for {len(available_features)} key features...")
    
    # Calculate normalized means for each cluster
    cluster_means = {}
    for cluster_id in range(3):
        cluster_data = df[df['KMeans_Cluster'] == cluster_id]
        cluster_means[cluster_id] = {}
        
        for feature in available_features:
            # Normalize to 0-1 scale for radar chart
            feature_min = df[feature].min()
            feature_max = df[feature].max()
            if feature_max > feature_min:
                normalized_value = (cluster_data[feature].mean() - feature_min) / (feature_max - feature_min)
            else:
                normalized_value = 0.5
            cluster_means[cluster_id][feature] = normalized_value
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    # Prepare data for plotting
    angles = np.linspace(0, 2 * np.pi, len(available_features), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    # Plot each cluster
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    cluster_names = ['Casual Players', 'Moderate Players', 'Hardcore Players']
    
    for cluster_id in range(3):
        values = [cluster_means[cluster_id][feature] for feature in available_features]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=cluster_names[cluster_id], color=colors[cluster_id])
        ax.fill(angles, values, alpha=0.25, color=colors[cluster_id])
    
    # Customize the plot
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(available_features, size=10)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
    ax.set_title('Player Cluster Profiles - Behavioral Radar Chart', size=16, pad=20, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('cluster_radar_charts.png', dpi=300, bbox_inches='tight')
    print("✅ Radar charts saved as 'cluster_radar_charts.png'")
    
    # Create individual radar charts for each cluster
    create_individual_radar_charts(available_features, cluster_means, cluster_names, colors)
    
    return cluster_means

def create_individual_radar_charts(features, cluster_means, cluster_names, colors):
    """Create individual radar charts for each cluster"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(projection='polar'))
    
    angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
    angles += angles[:1]
    
    for cluster_id in range(3):
        ax = axes[cluster_id]
        values = [cluster_means[cluster_id][feature] for feature in features]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=3, color=colors[cluster_id])
        ax.fill(angles, values, alpha=0.3, color=colors[cluster_id])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(features, size=8)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], size=8)
        ax.set_title(f'{cluster_names[cluster_id]} Profile', size=12, pad=20, fontweight='bold')
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('individual_cluster_radar_charts.png', dpi=300, bbox_inches='tight')
    print("✅ Individual radar charts saved as 'individual_cluster_radar_charts.png'")

def create_scatter_plots(df, analysis_cols):
    """
    Prompt 6.3: Create scatter plots with dimensionality reduction
    """
    print("\n" + "="*60)
    print("PROMPT 6.3: SCATTER PLOTS WITH DIMENSIONALITY REDUCTION")
    print("="*60)
    
    # Prepare data for PCA
    X = df[analysis_cols].values
    
    # Apply PCA for 2D visualization
    print("Applying Principal Component Analysis for 2D visualization...")
    pca_2d = PCA(n_components=2, random_state=42)
    X_2d = pca_2d.fit_transform(X)
    
    # Apply PCA for 3D visualization
    print("Applying Principal Component Analysis for 3D visualization...")
    pca_3d = PCA(n_components=3, random_state=42)
    X_3d = pca_3d.fit_transform(X)
    
    # Create 2D scatter plot
    create_2d_scatter_plot(X_2d, df, pca_2d, analysis_cols)
    
    # Create 3D scatter plot
    create_3d_scatter_plot(X_3d, df, pca_3d, analysis_cols)
    
    # Create feature importance plot
    create_feature_importance_plot(pca_2d, analysis_cols)
    
    print("✅ All scatter plots and visualizations created successfully")

def create_2d_scatter_plot(X_2d, df, pca, analysis_cols):
    """Create 2D scatter plot"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color points by cluster
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    cluster_names = ['Casual Players', 'Moderate Players', 'Hardcore Players']
    
    for cluster_id in range(3):
        cluster_mask = df['KMeans_Cluster'] == cluster_id
        cluster_points = X_2d[cluster_mask]
        
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                  c=colors[cluster_id], label=cluster_names[cluster_id], 
                  alpha=0.6, s=30)
    
    # Add cluster centroids
    for cluster_id in range(3):
        cluster_mask = df['KMeans_Cluster'] == cluster_id
        centroid = X_2d[cluster_mask].mean(axis=0)
        ax.scatter(centroid[0], centroid[1], c=colors[cluster_id], 
                  s=200, marker='x', linewidths=3, edgecolors='black')
    
    # Customize plot
    explained_variance = pca.explained_variance_ratio_
    ax.set_xlabel(f'Principal Component 1 ({explained_variance[0]:.1%} variance)', fontsize=12)
    ax.set_ylabel(f'Principal Component 2 ({explained_variance[1]:.1%} variance)', fontsize=12)
    ax.set_title('Player Clusters - 2D PCA Visualization', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('cluster_2d_scatter_plot.png', dpi=300, bbox_inches='tight')
    print("✅ 2D scatter plot saved as 'cluster_2d_scatter_plot.png'")

def create_3d_scatter_plot(X_3d, df, pca, analysis_cols):
    """Create 3D scatter plot"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color points by cluster
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    cluster_names = ['Casual Players', 'Moderate Players', 'Hardcore Players']
    
    for cluster_id in range(3):
        cluster_mask = df['KMeans_Cluster'] == cluster_id
        cluster_points = X_3d[cluster_mask]
        
        ax.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                  c=colors[cluster_id], label=cluster_names[cluster_id], 
                  alpha=0.6, s=20)
    
    # Customize plot
    explained_variance = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({explained_variance[0]:.1%} variance)', fontsize=10)
    ax.set_ylabel(f'PC2 ({explained_variance[1]:.1%} variance)', fontsize=10)
    ax.set_zlabel(f'PC3 ({explained_variance[2]:.1%} variance)', fontsize=10)
    ax.set_title('Player Clusters - 3D PCA Visualization', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('cluster_3d_scatter_plot.png', dpi=300, bbox_inches='tight')
    print("✅ 3D scatter plot saved as 'cluster_3d_scatter_plot.png'")

def create_feature_importance_plot(pca, analysis_cols):
    """Create feature importance plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # PC1 feature importance
    pc1_importance = np.abs(pca.components_[0])
    feature_importance_df = pd.DataFrame({
        'Feature': analysis_cols,
        'PC1_Importance': pc1_importance
    }).sort_values('PC1_Importance', ascending=True)
    
    ax1.barh(range(len(feature_importance_df)), feature_importance_df['PC1_Importance'])
    ax1.set_yticks(range(len(feature_importance_df)))
    ax1.set_yticklabels(feature_importance_df['Feature'])
    ax1.set_xlabel('Feature Importance (PC1)', fontsize=12)
    ax1.set_title('Principal Component 1 - Feature Importance', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # PC2 feature importance
    pc2_importance = np.abs(pca.components_[1])
    feature_importance_df['PC2_Importance'] = pc2_importance
    feature_importance_df = feature_importance_df.sort_values('PC2_Importance', ascending=True)
    
    ax2.barh(range(len(feature_importance_df)), feature_importance_df['PC2_Importance'])
    ax2.set_yticks(range(len(feature_importance_df)))
    ax2.set_yticklabels(feature_importance_df['Feature'])
    ax2.set_xlabel('Feature Importance (PC2)', fontsize=12)
    ax2.set_title('Principal Component 2 - Feature Importance', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_feature_importance.png', dpi=300, bbox_inches='tight')
    print("✅ Feature importance plot saved as 'pca_feature_importance.png'")

def develop_player_narratives(df, cluster_stats, analysis_cols):
    """
    Prompt 6.4: Develop rich narratives for player types
    """
    print("\n" + "="*60)
    print("PROMPT 6.4: DEVELOPING RICH NARRATIVES FOR PLAYER TYPES")
    print("="*60)
    
    # Define cluster characteristics based on statistical analysis
    cluster_profiles = {
        0: {
            'name': 'Casual Players',
            'percentage': 98.7,
            'size': 39511,
            'bartle_type': 'Socializers/Explorers',
            'description': 'The vast majority of players who engage in gaming as a casual pastime',
            'key_characteristics': [],
            'playstyle': '',
            'motivations': '',
            'business_implications': ''
        },
        1: {
            'name': 'Moderate Players',
            'percentage': 0.6,
            'size': 256,
            'bartle_type': 'Achievers/Explorers',
            'description': 'A small but dedicated group of players with balanced engagement',
            'key_characteristics': [],
            'playstyle': '',
            'motivations': '',
            'business_implications': ''
        },
        2: {
            'name': 'Hardcore Players',
            'percentage': 0.7,
            'size': 267,
            'bartle_type': 'Achievers/Killers',
            'description': 'Highly engaged players who represent the core gaming community',
            'key_characteristics': [],
            'playstyle': '',
            'motivations': '',
            'business_implications': ''
        }
    }
    
    # Analyze key characteristics for each cluster
    key_features = ['PlayTimeHours', 'SessionsPerWeek', 'PlayerLevel', 'AchievementsUnlocked', 
                   'InGamePurchases', 'TotalWeeklyPlaytime', 'AchievementRate']
    
    print("ANALYZING CLUSTER CHARACTERISTICS:")
    print("-" * 50)
    
    for cluster_id in range(3):
        print(f"\n{cluster_profiles[cluster_id]['name'].upper()} (Cluster {cluster_id})")
        print("-" * 40)
        
        # Analyze key features
        for feature in key_features:
            if feature in analysis_cols:
                cluster_mean = cluster_stats[cluster_id][feature]['mean']
                overall_mean = df[feature].mean()
                
                if cluster_mean > overall_mean * 1.2:
                    characteristic = f"High {feature.replace('_', ' ')}"
                    cluster_profiles[cluster_id]['key_characteristics'].append(characteristic)
                elif cluster_mean < overall_mean * 0.8:
                    characteristic = f"Low {feature.replace('_', ' ')}"
                    cluster_profiles[cluster_id]['key_characteristics'].append(characteristic)
        
        # Generate narrative based on characteristics
        generate_cluster_narrative(cluster_profiles[cluster_id], cluster_stats[cluster_id], analysis_cols)
        
        # Display narrative
        print(f"Key Characteristics: {', '.join(cluster_profiles[cluster_id]['key_characteristics'])}")
        print(f"Playstyle: {cluster_profiles[cluster_id]['playstyle']}")
        print(f"Motivations: {cluster_profiles[cluster_id]['motivations']}")
        print(f"Business Implications: {cluster_profiles[cluster_id]['business_implications']}")
    
    # Save narratives to file
    save_player_narratives(cluster_profiles)
    
    return cluster_profiles

def generate_cluster_narrative(profile, stats, analysis_cols):
    """Generate narrative for a specific cluster"""
    if profile['name'] == 'Casual Players':
        profile['playstyle'] = 'Infrequent, short gaming sessions with low commitment'
        profile['motivations'] = 'Entertainment, social connection, casual fun'
        profile['business_implications'] = 'Focus on retention, easy onboarding, social features'
    
    elif profile['name'] == 'Moderate Players':
        profile['playstyle'] = 'Regular gaming with balanced time investment and progression focus'
        profile['motivations'] = 'Achievement, exploration, moderate challenge'
        profile['business_implications'] = 'Content variety, achievement systems, moderate monetization'
    
    elif profile['name'] == 'Hardcore Players':
        profile['playstyle'] = 'Intensive gaming with high engagement and competitive focus'
        profile['motivations'] = 'Mastery, competition, deep progression'
        profile['business_implications'] = 'Premium content, advanced features, high-value monetization'

def save_player_narratives(cluster_profiles):
    """Save player narratives to file"""
    narratives = []
    for cluster_id, profile in cluster_profiles.items():
        narratives.append({
            'Cluster_ID': cluster_id,
            'Cluster_Name': profile['name'],
            'Percentage': profile['percentage'],
            'Size': profile['size'],
            'Bartle_Type': profile['bartle_type'],
            'Description': profile['description'],
            'Key_Characteristics': '; '.join(profile['key_characteristics']),
            'Playstyle': profile['playstyle'],
            'Motivations': profile['motivations'],
            'Business_Implications': profile['business_implications']
        })
    
    narratives_df = pd.DataFrame(narratives)
    narratives_df.to_csv('player_cluster_narratives.csv', index=False)
    print("✅ Player narratives saved to 'player_cluster_narratives.csv'")

def power_bi_integration_guide():
    """
    Prompt 6.5: Power BI integration guide
    """
    print("\n" + "="*60)
    print("PROMPT 6.5: POWER BI DASHBOARD CREATION GUIDE")
    print("="*60)
    
    print("POWER BI INTEGRATION INSTRUCTIONS:")
    print("=" * 50)
    
    print("\n1. CONNECTING TO SQLITE DATABASE:")
    print("   - Install 'SQLite ODBC Driver' on your system")
    print("   - In Power BI Desktop, go to 'Get Data' → 'ODBC'")
    print("   - Select 'SQLite ODBC Driver'")
    print("   - Navigate to 'gaming_data.db' in your project folder")
    print("   - Select 'clustered_player_data' table")
    
    print("\n2. RECOMMENDED VISUALIZATIONS:")
    print("   - Cluster Distribution Pie Chart")
    print("   - Feature Comparison Bar Charts by Cluster")
    print("   - Numerical Feature Histograms by Cluster")
    print("   - Cluster Summary Table")
    print("   - Interactive Filters for Cluster Analysis")
    
    print("\n3. KEY DASHBOARD COMPONENTS:")
    print("   - Overview section with cluster statistics")
    print("   - Detailed cluster analysis with drill-down capabilities")
    print("   - Feature comparison across clusters")
    print("   - Player behavior insights and trends")
    
    # Create Power BI data export
    create_power_bi_export()
    
    print("\n✅ Power BI integration guide completed")
    print("   Data export files created for easy Power BI import")

def create_power_bi_export():
    """Create optimized data exports for Power BI"""
    try:
        df_players = pd.read_pickle("df_players_clustered.pkl")
        
        # Create cluster summary for Power BI
        cluster_summary = df_players.groupby(['KMeans_Cluster', 'Cluster_Description']).agg({
            'PlayTimeHours': ['mean', 'median', 'std'],
            'SessionsPerWeek': ['mean', 'median', 'std'],
            'PlayerLevel': ['mean', 'median', 'std'],
            'AchievementsUnlocked': ['mean', 'median', 'std'],
            'InGamePurchases': ['mean', 'median', 'std'],
            'TotalWeeklyPlaytime': ['mean', 'median', 'std'],
            'AchievementRate': ['mean', 'median', 'std']
        }).round(3)
        
        # Flatten column names
        cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns]
        cluster_summary.reset_index(inplace=True)
        
        # Save for Power BI
        cluster_summary.to_csv('powerbi_cluster_summary.csv', index=False)
        
        # Create feature comparison data
        feature_comparison = df_players.groupby('KMeans_Cluster').agg({
            'PlayTimeHours': 'mean',
            'SessionsPerWeek': 'mean',
            'PlayerLevel': 'mean',
            'AchievementsUnlocked': 'mean',
            'InGamePurchases': 'mean',
            'TotalWeeklyPlaytime': 'mean',
            'AchievementRate': 'mean'
        }).round(3)
        
        feature_comparison.reset_index(inplace=True)
        feature_comparison.to_csv('powerbi_feature_comparison.csv', index=False)
        
        print("✅ Power BI export files created:")
        print("   • powerbi_cluster_summary.csv")
        print("   • powerbi_feature_comparison.csv")
        
    except Exception as e:
        print(f"❌ Error creating Power BI export: {e}")

def main():
    """Main execution function"""
    print("🚀 PHASE 6: CLUSTER INTERPRETATION, VISUALIZATION, AND POWER BI INTEGRATION")
    print("=" * 60)
    
    # Load clustered data
    df_players = load_clustered_data()
    if df_players is None:
        return
    
    # Step 1: Statistical characterization
    cluster_stats, analysis_cols = statistical_characterization_clusters(df_players)
    
    # Step 2: Create radar charts
    cluster_means = create_radar_charts(df_players, analysis_cols)
    
    # Step 3: Create scatter plots
    create_scatter_plots(df_players, analysis_cols)
    
    # Step 4: Develop player narratives
    cluster_profiles = develop_player_narratives(df_players, cluster_stats, analysis_cols)
    
    # Step 5: Power BI integration guide
    power_bi_integration_guide()
    
    print("\n" + "🎉 PHASE 6 COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nOutput Files Generated:")
    print(f"  • cluster_statistical_profiles.csv - Detailed cluster statistics")
    print(f"  • cluster_radar_charts.png - Cluster profile radar charts")
    print(f"  • individual_cluster_radar_charts.png - Individual cluster profiles")
    print(f"  • cluster_2d_scatter_plot.png - 2D PCA visualization")
    print(f"  • cluster_3d_scatter_plot.png - 3D PCA visualization")
    print(f"  • pca_feature_importance.png - Feature importance analysis")
    print(f"  • player_cluster_narratives.csv - Player type narratives")
    print(f"  • powerbi_cluster_summary.csv - Power BI cluster summary")
    print(f"  • powerbi_feature_comparison.csv - Power BI feature comparison")
    
    print(f"\nNext Steps:")
    print(f"  • Import data into Power BI for interactive dashboards")
    print(f"  • Use insights for business strategy development")
    print(f"  • Proceed to Phase 7: Business Intelligence and Recommendations")

if __name__ == "__main__":
    main()
