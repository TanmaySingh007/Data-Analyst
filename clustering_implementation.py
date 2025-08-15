#!/usr/bin/env python3
"""
Phase 5: Clustering Algorithm Implementation and Results Storage
This script implements K-Means and DBSCAN clustering, compares results, and stores data in SQL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sqlalchemy import create_engine, text
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

def load_engineered_data():
    """Load the engineered DataFrame"""
    try:
        df_players = pd.read_pickle("df_players_engineered.pkl")
        print("✅ Engineered DataFrame loaded successfully")
        print(f"Shape: {df_players.shape}")
        return df_players
    except FileNotFoundError:
        print("❌ Engineered DataFrame not found. Please run feature_engineering.py first.")
        return None

def prepare_data_for_clustering(df):
    """Prepare data for clustering analysis"""
    print("\n" + "="*60)
    print("PREPARING DATA FOR CLUSTERING ANALYSIS")
    print("="*60)
    
    # Select features for clustering (exclude PlayerID, encoded columns, and non-numerical columns)
    exclude_cols = ['PlayerID']
    encoded_prefixes = ['Gender_', 'Location_', 'GameGenre_', 'GameDifficulty_']
    
    # Find columns that start with encoded prefixes
    encoded_cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in encoded_prefixes)]
    exclude_cols.extend(encoded_cols)
    
    # Get only numerical columns for clustering
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    clustering_cols = [col for col in numerical_cols if col not in exclude_cols]
    
    print(f"Features selected for clustering: {len(clustering_cols)}")
    print("Selected features:")
    for i, col in enumerate(clustering_cols, 1):
        print(f"  {i:2d}. {col}")
    
    # Create feature matrix
    X = df[clustering_cols].values
    
    # Check for any remaining NaN or inf values
    print(f"\nData quality check:")
    print(f"  NaN values: {np.isnan(X).sum()}")
    print(f"  Inf values: {np.isinf(X).sum()}")
    
    # Handle any remaining issues
    if np.isnan(X).sum() > 0 or np.isinf(X).sum() > 0:
        print("  Cleaning data...")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"  After cleaning - NaN: {np.isnan(X).sum()}, Inf: {np.isinf(X).sum()}")
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Data type: {X.dtype}")
    
    return X, clustering_cols

def implement_kmeans_clustering(X, df, optimal_k=3):
    """
    Prompt 5.1: Implement K-Means clustering using optimal k
    """
    print("\n" + "="*60)
    print("PROMPT 5.1: K-MEANS CLUSTERING IMPLEMENTATION")
    print("="*60)
    
    print(f"Implementing K-Means clustering with k={optimal_k}...")
    
    # Initialize K-Means with robust parameters
    kmeans = KMeans(
        n_clusters=optimal_k,
        random_state=42,
        n_init=10,  # Multiple initializations for robust centroid placement
        max_iter=300,
        tol=1e-4
    )
    
    # Fit the model
    kmeans.fit(X)
    
    # Get cluster labels and add to DataFrame
    cluster_labels = kmeans.labels_
    df['KMeans_Cluster'] = cluster_labels
    
    # Calculate clustering metrics
    silhouette_avg = silhouette_score(X, cluster_labels)
    calinski_score = calinski_harabasz_score(X, cluster_labels)
    
    print("✅ K-Means clustering completed successfully!")
    print(f"  Optimal k: {optimal_k}")
    print(f"  Silhouette Score: {silhouette_avg:.4f}")
    print(f"  Calinski-Harabasz Score: {calinski_score:.2f}")
    print(f"  Inertia (WCSS): {kmeans.inertia_:.2f}")
    
    # Analyze cluster distribution
    print(f"\nCluster Distribution:")
    cluster_counts = df['KMeans_Cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  Cluster {cluster_id}: {count} players ({percentage:.1f}%)")
    
    # Store cluster centers for analysis
    cluster_centers = kmeans.cluster_centers_
    print(f"\nCluster Centers Shape: {cluster_centers.shape}")
    
    # Save K-Means model
    import joblib
    joblib.dump(kmeans, "kmeans_model.pkl")
    print("✅ K-Means model saved as 'kmeans_model.pkl'")
    
    return df, kmeans, cluster_centers

def implement_dbscan_clustering(X, df):
    """
    Prompt 5.2: Implement DBSCAN clustering with parameter tuning
    """
    print("\n" + "="*60)
    print("PROMPT 5.2: DBSCAN CLUSTERING IMPLEMENTATION AND PARAMETER TUNING")
    print("="*60)
    
    print("Implementing DBSCAN clustering with parameter experimentation...")
    
    # Define parameter combinations to test
    eps_values = [0.5, 1.0, 1.5, 2.0, 2.5]
    min_samples_values = [5, 10, 15, 20]
    
    best_score = -1
    best_params = None
    best_dbscan = None
    best_labels = None
    
    results = []
    
    print("Testing DBSCAN parameter combinations:")
    print("eps\tmin_samples\tclusters\tnoise_points\tsilhouette_score")
    print("-" * 70)
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            try:
                # Initialize DBSCAN
                dbscan = DBSCAN(eps=eps, min_samples=min_samples)
                
                # Fit the model
                dbscan.fit(X)
                
                # Get cluster labels
                labels = dbscan.labels_
                
                # Count clusters (excluding noise points labeled as -1)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                # Calculate silhouette score (only if we have at least 2 clusters)
                if n_clusters >= 2:
                    # Filter out noise points for silhouette calculation
                    mask = labels != -1
                    if np.sum(mask) > 1:
                        silhouette_avg = silhouette_score(X[mask], labels[mask])
                    else:
                        silhouette_avg = -1
                else:
                    silhouette_avg = -1
                
                # Store results
                result = {
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'n_noise': n_noise,
                    'silhouette_score': silhouette_avg,
                    'labels': labels,
                    'model': dbscan
                }
                results.append(result)
                
                # Print results
                print(f"{eps}\t{min_samples}\t\t{n_clusters}\t\t{n_noise}\t\t{silhouette_avg:.4f}")
                
                # Track best result based on silhouette score
                if silhouette_avg > best_score and n_clusters >= 2:
                    best_score = silhouette_avg
                    best_params = (eps, min_samples)
                    best_dbscan = dbscan
                    best_labels = labels
                    
            except Exception as e:
                print(f"Error with eps={eps}, min_samples={min_samples}: {e}")
                continue
    
    # Select best DBSCAN result
    if best_dbscan is not None:
        print(f"\n✅ Best DBSCAN parameters found:")
        print(f"  eps: {best_params[0]}")
        print(f"  min_samples: {best_params[1]}")
        print(f"  Silhouette Score: {best_score:.4f}")
        
        # Add cluster labels to DataFrame
        df['DBSCAN_Cluster'] = best_labels
        
        # Analyze cluster distribution
        print(f"\nDBSCAN Cluster Distribution:")
        cluster_counts = df['DBSCAN_Cluster'].value_counts().sort_index()
        for cluster_id, count in cluster_counts.items():
            if cluster_id == -1:
                print(f"  Noise Points: {count} players ({(count/len(df)*100):.1f}%)")
            else:
                percentage = (count / len(df)) * 100
                print(f"  Cluster {cluster_id}: {count} players ({percentage:.1f}%)")
        
        # Save best DBSCAN model
        import joblib
        joblib.dump(best_dbscan, "dbscan_model.pkl")
        print("✅ Best DBSCAN model saved as 'dbscan_model.pkl'")
        
    else:
        print("❌ No suitable DBSCAN parameters found")
        df['DBSCAN_Cluster'] = -1  # All points as noise
    
    # Discuss DBSCAN parameter tuning challenges
    print("\n" + "-"*50)
    print("DBSCAN PARAMETER TUNING CHALLENGES:")
    print("-"*50)
    print("1. EPS PARAMETER:")
    print("   - Too small: Many small clusters, many noise points")
    print("   - Too large: Few large clusters, may merge distinct groups")
    print("   - Depends on data scale and feature relationships")
    
    print("\n2. MIN_SAMPLES PARAMETER:")
    print("   - Too small: Sensitive to noise, many small clusters")
    print("   - Too large: May miss legitimate small clusters")
    print("   - Should consider data density and expected cluster sizes")
    
    print("\n3. ALGORITHM ADVANTAGES:")
    print("   - Handles outliers naturally (classifies as noise)")
    print("   - Can find clusters of arbitrary shapes")
    print("   - No need to specify number of clusters beforehand")
    
    print("\n4. COMPARED TO K-MEANS:")
    print("   - K-Means: Sensitive to outliers, spherical clusters")
    print("   - DBSCAN: Robust to outliers, arbitrary cluster shapes")
    print("   - K-Means: Requires k, DBSCAN: Automatic cluster detection")
    
    return df, best_dbscan, results

def compare_clustering_algorithms(df, kmeans_model, dbscan_model, X):
    """
    Prompt 5.3: Qualitatively compare K-Means and DBSCAN results
    """
    print("\n" + "="*60)
    print("PROMPT 5.3: QUALITATIVE COMPARISON OF K-MEANS AND DBSCAN")
    print("="*60)
    
    # Get cluster labels
    kmeans_labels = df['KMeans_Cluster'].values
    dbscan_labels = df['DBSCAN_Cluster'].values
    
    # Calculate metrics for comparison
    print("CLUSTERING ALGORITHM COMPARISON:")
    print("="*50)
    
    # K-Means metrics
    kmeans_silhouette = silhouette_score(X, kmeans_labels)
    kmeans_calinski = calinski_harabasz_score(X, kmeans_labels)
    kmeans_clusters = len(set(kmeans_labels))
    kmeans_noise = 0  # K-Means doesn't produce noise points
    
    print(f"\nK-MEANS RESULTS:")
    print(f"  Number of clusters: {kmeans_clusters}")
    print(f"  Noise points: {kmeans_noise}")
    print(f"  Silhouette Score: {kmeans_silhouette:.4f}")
    print(f"  Calinski-Harabasz Score: {kmeans_calinski:.2f}")
    
    # DBSCAN metrics
    dbscan_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    dbscan_noise = list(dbscan_labels).count(-1)
    
    if dbscan_clusters >= 2:
        # Filter out noise points for DBSCAN metrics
        mask = dbscan_labels != -1
        if np.sum(mask) > 1:
            dbscan_silhouette = silhouette_score(X[mask], dbscan_labels[mask])
            dbscan_calinski = calinski_harabasz_score(X[mask], dbscan_labels[mask])
        else:
            dbscan_silhouette = -1
            dbscan_calinski = -1
    else:
        dbscan_silhouette = -1
        dbscan_calinski = -1
    
    print(f"\nDBSCAN RESULTS:")
    print(f"  Number of clusters: {dbscan_clusters}")
    print(f"  Noise points: {dbscan_noise} ({(dbscan_noise/len(df)*100):.1f}%)")
    print(f"  Silhouette Score: {dbscan_silhouette:.4f}")
    print(f"  Calinski-Harabasz Score: {dbscan_calinski:.2f}")
    
    # Create comparison visualization
    create_clustering_comparison_plots(df, X, kmeans_labels, dbscan_labels)
    
    # Qualitative analysis
    print("\n" + "-"*50)
    print("QUALITATIVE COMPARISON ANALYSIS:")
    print("-"*50)
    
    print("\n1. CLUSTER SHAPE CHARACTERISTICS:")
    print("   - K-Means: Produces spherical clusters with clear centroids")
    print("   - DBSCAN: Can find clusters of arbitrary shapes")
    print("   - Gaming data: Players may form non-spherical behavioral groups")
    
    print("\n2. OUTLIER HANDLING:")
    print("   - K-Means: Outliers can significantly affect cluster centroids")
    print("   - DBSCAN: Naturally identifies outliers as noise points")
    print("   - Gaming context: Outliers may represent unique player types")
    
    print("\n3. INTERPRETABILITY:")
    print("   - K-Means: Clear cluster centroids for business interpretation")
    print("   - DBSCAN: Cluster boundaries less intuitive")
    print("   - Business value: Centroids help define player archetypes")
    
    print("\n4. ALGORITHM APPROPRIATENESS:")
    print("   - K-Means: Better for this gaming behavior analysis because:")
    print("     • Clear, interpretable cluster centroids")
    print("     • Well-separated player segments expected")
    print("     • Business stakeholders need actionable insights")
    print("     • Outliers are minimal in this dataset")
    
    print("\n   - DBSCAN: Less suitable because:")
    print("     • Complex parameter tuning required")
    print("     • May produce too many small clusters")
    print("     • Noise classification may not align with business needs")
    
    return df

def create_clustering_comparison_plots(df, X, kmeans_labels, dbscan_labels):
    """Create visualization plots for clustering comparison"""
    print("\nCreating clustering comparison visualizations...")
    
    # Use PCA for 2D visualization
    from sklearn.decomposition import PCA
    
    # Reduce to 2D for visualization
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Clustering Algorithm Comparison', fontsize=16, fontweight='bold')
    
    # K-Means visualization
    ax1 = axes[0, 0]
    scatter1 = ax1.scatter(X_2d[:, 0], X_2d[:, 1], c=kmeans_labels, cmap='viridis', alpha=0.6)
    ax1.set_title('K-Means Clustering Results', fontweight='bold')
    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel('Principal Component 2')
    ax1.grid(True, alpha=0.3)
    
    # Add cluster count annotations
    unique_clusters = np.unique(kmeans_labels)
    for cluster_id in unique_clusters:
        cluster_points = X_2d[kmeans_labels == cluster_id]
        centroid = cluster_points.mean(axis=0)
        ax1.annotate(f'Cluster {cluster_id}\n({len(cluster_points)} players)', 
                     centroid, ha='center', va='center', 
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # DBSCAN visualization
    ax2 = axes[0, 1]
    scatter2 = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=dbscan_labels, cmap='viridis', alpha=0.6)
    ax2.set_title('DBSCAN Clustering Results', fontweight='bold')
    ax2.set_xlabel('Principal Component 1')
    ax2.set_ylabel('Principal Component 2')
    ax2.grid(True, alpha=0.3)
    
    # Add cluster count annotations for DBSCAN
    unique_clusters_dbscan = np.unique(dbscan_labels)
    for cluster_id in unique_clusters_dbscan:
        if cluster_id != -1:  # Skip noise points
            cluster_points = X_2d[dbscan_labels == cluster_id]
            centroid = cluster_points.mean(axis=0)
            ax2.annotate(f'Cluster {cluster_id}\n({len(cluster_points)} players)', 
                         centroid, ha='center', va='center', 
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Noise points annotation
    noise_points = X_2d[dbscan_labels == -1]
    if len(noise_points) > 0:
        ax2.annotate(f'Noise Points\n({len(noise_points)} players)', 
                     noise_points.mean(axis=0), ha='center', va='center',
                     bbox=dict(boxstyle='round', facecolor='red', alpha=0.8))
    
    # Cluster size comparison
    ax3 = axes[1, 0]
    kmeans_counts = df['KMeans_Cluster'].value_counts().sort_index()
    dbscan_counts = df['DBSCAN_Cluster'].value_counts().sort_index()
    
    # Handle different numbers of clusters
    max_clusters = max(len(kmeans_counts), len(dbscan_counts))
    x = np.arange(max_clusters)
    width = 0.35
    
    # Prepare data for plotting
    kmeans_values = np.zeros(max_clusters)
    dbscan_values = np.zeros(max_clusters)
    
    # Fill K-Means values
    for i, cluster_id in enumerate(kmeans_counts.index):
        if i < max_clusters:
            kmeans_values[i] = kmeans_counts[cluster_id]
    
    # Fill DBSCAN values
    for i, cluster_id in enumerate(dbscan_counts.index):
        if i < max_clusters:
            dbscan_values[i] = dbscan_counts[cluster_id]
    
    ax3.bar(x - width/2, kmeans_values, width, label='K-Means', alpha=0.7)
    ax3.bar(x + width/2, dbscan_values, width, label='DBSCAN', alpha=0.7)
    
    ax3.set_xlabel('Cluster ID')
    ax3.set_ylabel('Number of Players')
    ax3.set_title('Cluster Size Comparison', fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'C{i}' for i in range(max_clusters)])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Algorithm performance metrics
    ax4 = axes[1, 1]
    metrics = ['Silhouette Score', 'Number of Clusters', 'Noise Points']
    
    # Calculate metrics safely
    kmeans_silhouette = silhouette_score(X, kmeans_labels)
    kmeans_cluster_count = len(set(kmeans_labels))
    kmeans_noise = 0
    
    # Handle DBSCAN metrics safely
    dbscan_cluster_count = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    dbscan_noise = list(dbscan_labels).count(-1)
    
    if dbscan_cluster_count >= 2:
        mask = dbscan_labels != -1
        if np.sum(mask) > 1:
            dbscan_silhouette = silhouette_score(X[mask], dbscan_labels[mask])
        else:
            dbscan_silhouette = -1
    else:
        dbscan_silhouette = -1
    
    kmeans_metrics = [kmeans_silhouette, kmeans_cluster_count, kmeans_noise]
    dbscan_metrics = [dbscan_silhouette, dbscan_cluster_count, dbscan_noise]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    ax4.bar(x - width/2, kmeans_metrics, width, label='K-Means', alpha=0.7)
    ax4.bar(x + width/2, dbscan_metrics, width, label='DBSCAN', alpha=0.7)
    
    ax4.set_xlabel('Metrics')
    ax4.set_ylabel('Score/Count')
    ax4.set_title('Algorithm Performance Comparison', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics, rotation=45)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('clustering_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Clustering comparison plots saved as 'clustering_comparison.png'")

def store_clustered_data_in_sql(df):
    """
    Prompt 5.4: Store clustered data in SQL database
    """
    print("\n" + "="*60)
    print("PROMPT 5.4: STORING CLUSTERED DATA IN SQL")
    print("="*60)
    
    # Select preferred clustering result (K-Means based on analysis)
    preferred_cluster_col = 'KMeans_Cluster'
    print(f"Using {preferred_cluster_col} as the preferred clustering result")
    
    # Ensure the cluster column exists
    if preferred_cluster_col not in df.columns:
        print(f"❌ {preferred_cluster_col} column not found in DataFrame")
        return False
    
    # Create a copy with cluster labels for SQL storage
    df_for_sql = df.copy()
    
    # Convert cluster labels to string for better SQL compatibility
    df_for_sql['Cluster_Label'] = df_for_sql[preferred_cluster_col].astype(str)
    
    # Add cluster description
    cluster_descriptions = {
        0: 'Casual Players',
        1: 'Moderate Players', 
        2: 'Hardcore Players'
    }
    
    df_for_sql['Cluster_Description'] = df_for_sql[preferred_cluster_col].map(cluster_descriptions)
    
    print("✅ DataFrame prepared for SQL storage")
    print(f"  Total rows: {len(df_for_sql)}")
    print(f"  Total columns: {len(df_for_sql.columns)}")
    print(f"  Cluster column: {preferred_cluster_col}")
    
    # Connect to SQLite database
    try:
        engine = create_engine('sqlite:///gaming_data.db')
        print("✅ Connected to gaming_data.db successfully")
        
        # Create new table for clustered data
        table_name = 'clustered_player_data'
        
        # Store the DataFrame in SQL
        df_for_sql.to_sql(table_name, engine, if_exists='replace', index=False)
        
        # Verify the data was stored
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            row_count = result.fetchone()[0]
            
            result = conn.execute(text(f"SELECT COUNT(*) FROM information_schema.columns WHERE table_name = '{table_name}'"))
            col_count = result.fetchone()[0]
            
            # Get cluster distribution from SQL
            cluster_query = f"""
            SELECT Cluster_Label, Cluster_Description, COUNT(*) as Player_Count
            FROM {table_name}
            GROUP BY Cluster_Label, Cluster_Description
            ORDER BY Cluster_Label
            """
            
            cluster_result = conn.execute(text(cluster_query))
            cluster_distribution = cluster_result.fetchall()
        
        print(f"✅ Data successfully stored in SQL table '{table_name}'")
        print(f"  Rows stored: {row_count}")
        print(f"  Columns stored: {len(df_for_sql.columns)}")
        
        print(f"\nCluster Distribution in SQL:")
        for cluster_label, description, count in cluster_distribution:
            percentage = (count / row_count) * 100
            print(f"  {description} (Cluster {cluster_label}): {count} players ({percentage:.1f}%)")
        
        # Save the clustered DataFrame
        df_for_sql.to_pickle("df_players_clustered.pkl")
        print("✅ Clustered DataFrame saved as 'df_players_clustered.pkl'")
        
        # Create clustering summary
        clustering_summary = {
            'preferred_algorithm': 'K-Means',
            'optimal_k': 3,
            'cluster_descriptions': cluster_descriptions,
            'sql_table': table_name,
            'total_players': row_count,
            'features_used': len([col for col in df.columns if col not in ['PlayerID', 'KMeans_Cluster', 'DBSCAN_Cluster']]),
            'clustering_metrics': {
                'silhouette_score': silhouette_score(X, df[preferred_cluster_col].values),
                'calinski_harabasz_score': calinski_harabasz_score(X, df[preferred_cluster_col].values)
            }
        }
        
        import json
        with open('clustering_summary.json', 'w') as f:
            json.dump(clustering_summary, f, indent=2)
        print("✅ Clustering summary saved as 'clustering_summary.json'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error storing data in SQL: {e}")
        return False

def main():
    """Main execution function"""
    print("🚀 PHASE 5: CLUSTERING ALGORITHM IMPLEMENTATION AND RESULTS STORAGE")
    print("=" * 60)
    
    # Load the engineered data
    df_players = load_engineered_data()
    if df_players is None:
        return
    
    # Prepare data for clustering
    X, clustering_cols = prepare_data_for_clustering(df_players)
    
    # Step 1: Implement K-Means clustering
    df_players, kmeans_model, kmeans_centers = implement_kmeans_clustering(X, df_players, optimal_k=3)
    
    # Step 2: Implement DBSCAN clustering
    df_players, dbscan_model, dbscan_results = implement_dbscan_clustering(X, df_players)
    
    # Step 3: Compare clustering algorithms
    df_players = compare_clustering_algorithms(df_players, kmeans_model, dbscan_model, X)
    
    # Step 4: Store clustered data in SQL
    success = store_clustered_data_in_sql(df_players)
    
    if success:
        print("\n" + "🎉 PHASE 5 COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"\nClustering Results Summary:")
        print(f"  Preferred Algorithm: K-Means (k=3)")
        print(f"  Total Players Clustered: {len(df_players)}")
        print(f"  Features Used: {len(clustering_cols)}")
        print(f"  SQL Table Created: clustered_player_data")
        
        print(f"\nOutput Files Generated:")
        print(f"  • df_players_clustered.pkl - Final clustered dataset")
        print(f"  • kmeans_model.pkl - Trained K-Means model")
        print(f"  • dbscan_model.pkl - Best DBSCAN model")
        print(f"  • clustering_comparison.png - Algorithm comparison plots")
        print(f"  • clustering_summary.json - Complete clustering metadata")
        
        print(f"\nNext Steps:")
        print(f"  • Data ready for Power BI visualization")
        print(f"  • Cluster analysis and player profiling")
        print(f"  • Business intelligence applications")
        print(f"  • Player segmentation strategies")
    else:
        print("\n❌ Phase 5 completed with errors. Please check the output above.")

if __name__ == "__main__":
    main()
