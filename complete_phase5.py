#!/usr/bin/env python3
"""
Phase 5: Complete Clustering Implementation
Simplified version to avoid hanging issues
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

def main():
    print("🚀 PHASE 5: COMPLETING CLUSTERING IMPLEMENTATION")
    print("=" * 60)
    
    # Step 1: Load data
    print("Step 1: Loading engineered data...")
    try:
        df_players = pd.read_pickle("df_players_engineered.pkl")
        print(f"✅ Data loaded: {df_players.shape}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Step 2: Prepare clustering features
    print("\nStep 2: Preparing clustering features...")
    exclude_cols = ['PlayerID']
    encoded_prefixes = ['Gender_', 'Location_', 'GameGenre_', 'GameDifficulty_']
    encoded_cols = [col for col in df_players.columns if any(col.startswith(prefix) for prefix in encoded_prefixes)]
    exclude_cols.extend(encoded_cols)
    
    numerical_cols = df_players.select_dtypes(include=[np.number]).columns
    clustering_cols = [col for col in numerical_cols if col not in exclude_cols]
    X = df_players[clustering_cols].values
    
    print(f"✅ Features prepared: {X.shape}")
    
    # Step 3: Load existing K-Means results
    print("\nStep 3: Loading existing K-Means results...")
    try:
        import joblib
        kmeans_model = joblib.load("kmeans_model.pkl")
        df_players['KMeans_Cluster'] = kmeans_model.labels_
        print("✅ K-Means results loaded")
    except Exception as e:
        print(f"❌ Error loading K-Means: {e}")
        return
    
    # Step 4: Load existing DBSCAN results
    print("\nStep 4: Loading existing DBSCAN results...")
    try:
        dbscan_model = joblib.load("dbscan_model.pkl")
        df_players['DBSCAN_Cluster'] = dbscan_model.labels_
        print("✅ DBSCAN results loaded")
    except Exception as e:
        print(f"❌ Error loading DBSCAN: {e}")
        return
    
    # Step 5: Create comparison plots
    print("\nStep 5: Creating comparison plots...")
    try:
        create_simple_comparison_plots(df_players, X)
        print("✅ Comparison plots created")
    except Exception as e:
        print(f"❌ Error creating plots: {e}")
    
    # Step 6: Store in SQL
    print("\nStep 6: Storing data in SQL...")
    try:
        success = store_in_sql(df_players)
        if success:
            print("✅ Data stored in SQL")
        else:
            print("❌ Failed to store in SQL")
    except Exception as e:
        print(f"❌ Error storing in SQL: {e}")
    
    # Step 7: Save final results
    print("\nStep 7: Saving final results...")
    try:
        df_players.to_pickle("df_players_clustered.pkl")
        create_summary(df_players, X)
        print("✅ Final results saved")
    except Exception as e:
        print(f"❌ Error saving results: {e}")
    
    print("\n🎉 PHASE 5 COMPLETED!")
    print("=" * 60)

def create_simple_comparison_plots(df, X):
    """Create simplified comparison plots"""
    # PCA reduction
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    
    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # K-Means plot
    kmeans_labels = df['KMeans_Cluster'].values
    scatter1 = axes[0].scatter(X_2d[:, 0], X_2d[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6)
    axes[0].set_title('K-Means Clustering (k=3)', fontweight='bold')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[0].grid(True, alpha=0.3)
    
    # DBSCAN plot
    dbscan_labels = df['DBSCAN_Cluster'].values
    scatter2 = axes[1].scatter(X_2d[:, 0], X_2d[:, 1], c=dbscan_labels, cmap='viridis', alpha=0.6)
    axes[1].set_title('DBSCAN Clustering', fontweight='bold')
    axes[1].set_xlabel('Principal Component 1')
    axes[1].set_ylabel('Principal Component 2')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('clustering_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def store_in_sql(df):
    """Store clustered data in SQL"""
    try:
        # Prepare data
        df_for_sql = df.copy()
        df_for_sql['Cluster_Label'] = df_for_sql['KMeans_Cluster'].astype(str)
        
        cluster_descriptions = {0: 'Casual Players', 1: 'Moderate Players', 2: 'Hardcore Players'}
        df_for_sql['Cluster_Description'] = df_for_sql['KMeans_Cluster'].map(cluster_descriptions)
        
        # Store in SQL
        engine = create_engine('sqlite:///gaming_data.db')
        table_name = 'clustered_player_data'
        df_for_sql.to_sql(table_name, engine, if_exists='replace', index=False)
        
        # Verify
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            row_count = result.fetchone()[0]
        
        print(f"  Rows stored: {row_count}")
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        return False

def create_summary(df, X):
    """Create clustering summary"""
    kmeans_labels = df['KMeans_Cluster'].values
    
    summary = {
        'preferred_algorithm': 'K-Means',
        'optimal_k': 3,
        'cluster_descriptions': {
            0: 'Casual Players',
            1: 'Moderate Players', 
            2: 'Hardcore Players'
        },
        'sql_table': 'clustered_player_data',
        'total_players': len(df),
        'features_used': X.shape[1],
        'clustering_metrics': {
            'silhouette_score': float(silhouette_score(X, kmeans_labels)),
            'calinski_harabasz_score': float(calinski_harabasz_score(X, kmeans_labels))
        }
    }
    
    import json
    with open('clustering_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
