#!/usr/bin/env python3
"""
Phase 4: Determining Optimal Number of Clusters (for K-Means)
This script uses Elbow Method and Silhouette Score to find optimal k
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
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

def elbow_method_analysis(X, max_k=10):
    """
    Prompt 4.1: Apply Elbow Method to determine optimal k
    """
    print("\n" + "="*60)
    print("PROMPT 4.1: ELBOW METHOD FOR K-MEANS")
    print("="*60)
    
    # Calculate WCSS for different k values
    wcss = []
    k_range = range(1, max_k + 1)
    
    print("Calculating Within-Cluster Sum of Squares (WCSS)...")
    for k in k_range:
        if k == 1:
            # For k=1, WCSS is just the total variance
            wcss.append(np.var(X, axis=0).sum())
        else:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        
        print(f"  k={k}: WCSS = {wcss[-1]:.2f}")
    
    # Create elbow plot
    plt.figure(figsize=(12, 8))
    
    # Main plot
    plt.subplot(2, 2, 1)
    plt.plot(k_range, wcss, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12, fontweight='bold')
    plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add annotations for key points
    for i, (k, w) in enumerate(zip(k_range, wcss)):
        plt.annotate(f'k={k}', (k, w), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    # Calculate and plot rate of change
    plt.subplot(2, 2, 2)
    wcss_diff = np.diff(wcss)
    plt.plot(k_range[1:], wcss_diff, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    plt.ylabel('Rate of Change in WCSS', fontsize=12, fontweight='bold')
    plt.title('Rate of Change in WCSS', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    for i, (k, diff) in enumerate(zip(k_range[1:], wcss_diff)):
        plt.annotate(f'{diff:.1f}', (k, diff), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    # Calculate and plot percentage improvement
    plt.subplot(2, 2, 3)
    wcss_improvement = [(wcss[i-1] - wcss[i]) / wcss[i-1] * 100 for i in range(1, len(wcss))]
    plt.plot(k_range[1:], wcss_improvement, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    plt.ylabel('Percentage Improvement (%)', fontsize=12, fontweight='bold')
    plt.title('Percentage Improvement in WCSS', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    for i, (k, imp) in enumerate(zip(k_range[1:], wcss_improvement)):
        plt.annotate(f'{imp:.1f}%', (k, imp), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    # Cumulative variance explained
    plt.subplot(2, 2, 4)
    total_variance = wcss[0]
    variance_explained = [(total_variance - wcss[i]) / total_variance * 100 for i in range(len(wcss))]
    plt.plot(k_range, variance_explained, 'mo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    plt.ylabel('Cumulative Variance Explained (%)', fontsize=12, fontweight='bold')
    plt.title('Cumulative Variance Explained', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    for i, (k, var) in enumerate(zip(k_range, variance_explained)):
        plt.annotate(f'{var:.1f}%', (k, var), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('elbow_method_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Elbow method analysis plots saved as 'elbow_method_analysis.png'")
    
    # Analyze elbow point
    print(f"\nElbow Method Analysis:")
    print(f"  Total variance (k=1): {wcss[0]:.2f}")
    print(f"  WCSS at k=2: {wcss[1]:.2f}")
    print(f"  WCSS at k=3: {wcss[2]:.2f}")
    print(f"  WCSS at k=4: {wcss[3]:.2f}")
    print(f"  WCSS at k=5: {wcss[4]:.2f}")
    
    # Calculate improvement rates
    print(f"\nImprovement rates:")
    for i in range(1, min(6, len(wcss))):
        improvement = (wcss[i-1] - wcss[i]) / wcss[i-1] * 100
        print(f"  k={i}→{i+1}: {improvement:.2f}% improvement")
    
    # Explain what the elbow point signifies
    print("\n" + "-"*50)
    print("WHAT THE ELBOW POINT SIGNIFIES:")
    print("-"*50)
    print("1. OPTIMAL COMPLEXITY:")
    print("   - Point where adding more clusters provides diminishing returns")
    print("   - Balance between model complexity and performance")
    print("   - Trade-off between overfitting and underfitting")
    
    print("\n2. INTERPRETATION:")
    print("   - Sharp drop in WCSS = significant cluster separation")
    print("   - Gradual decline = clusters becoming less distinct")
    print("   - Elbow = optimal number of meaningful clusters")
    
    print("\n3. BUSINESS VALUE:")
    print("   - Too few clusters = oversimplified player segments")
    print("   - Too many clusters = hard to interpret and act upon")
    print("   - Optimal k = actionable player segments")
    
    return wcss, k_range

def silhouette_score_analysis(X, max_k=10):
    """
    Prompt 4.2: Calculate Silhouette Score for different k values
    """
    print("\n" + "="*60)
    print("PROMPT 4.2: SILHOUETTE SCORE ANALYSIS FOR K-MEANS")
    print("="*60)
    
    # Calculate silhouette scores for different k values
    silhouette_scores = []
    k_range = range(2, max_k + 1)  # Silhouette score requires at least 2 clusters
    
    print("Calculating Silhouette Scores...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(X)
        score = silhouette_score(X, cluster_labels)
        silhouette_scores.append(score)
        print(f"  k={k}: Silhouette Score = {score:.4f}")
    
    # Create silhouette analysis plot
    plt.figure(figsize=(15, 10))
    
    # Main silhouette score plot
    plt.subplot(2, 2, 1)
    plt.plot(k_range, silhouette_scores, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    plt.ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    plt.title('Silhouette Score vs Number of Clusters', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    for i, (k, score) in enumerate(zip(k_range, silhouette_scores)):
        plt.annotate(f'{score:.3f}', (k, score), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    # Find optimal k based on silhouette score
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    plt.axvline(x=optimal_k_silhouette, color='red', linestyle='--', 
                label=f'Optimal k = {optimal_k_silhouette}')
    plt.legend()
    
    # Silhouette score distribution for optimal k
    plt.subplot(2, 2, 2)
    kmeans_optimal = KMeans(n_clusters=optimal_k_silhouette, random_state=42, n_init=10)
    cluster_labels_optimal = kmeans_optimal.fit_predict(X)
    
    from sklearn.metrics import silhouette_samples
    silhouette_vals = silhouette_samples(X, cluster_labels_optimal)
    
    plt.hist(silhouette_vals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Silhouette Score', fontsize=12, fontweight='bold')
    plt.ylabel('Frequency', fontsize=12, fontweight='bold')
    plt.title(f'Silhouette Score Distribution (k={optimal_k_silhouette})', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add mean line
    mean_silhouette = np.mean(silhouette_vals)
    plt.axvline(x=mean_silhouette, color='red', linestyle='--', 
                label=f'Mean: {mean_silhouette:.3f}')
    plt.legend()
    
    # Silhouette score comparison
    plt.subplot(2, 2, 3)
    plt.bar(k_range, silhouette_scores, color='lightcoral', alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    plt.ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    plt.title('Silhouette Score Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for i, (k, score) in enumerate(zip(k_range, silhouette_scores)):
        plt.text(k, score + 0.01, f'{score:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Highlight optimal k
    plt.bar(optimal_k_silhouette, silhouette_scores[np.argmax(silhouette_scores)], 
            color='red', alpha=0.8, edgecolor='black')
    
    # Score interpretation
    plt.subplot(2, 2, 4)
    score_interpretations = []
    for score in silhouette_scores:
        if score >= 0.7:
            interpretation = "Strong"
        elif score >= 0.5:
            interpretation = "Reasonable"
        elif score >= 0.25:
            interpretation = "Weak"
        else:
            interpretation = "Poor"
        score_interpretations.append(interpretation)
    
    colors = ['green' if s >= 0.5 else 'orange' if s >= 0.25 else 'red' for s in silhouette_scores]
    plt.bar(k_range, silhouette_scores, color=colors, alpha=0.7, edgecolor='black')
    plt.xlabel('Number of Clusters (k)', fontsize=12, fontweight='bold')
    plt.ylabel('Silhouette Score', fontsize=12, fontweight='bold')
    plt.title('Silhouette Score Quality Assessment', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add quality labels
    for i, (k, score, interpretation) in enumerate(zip(k_range, silhouette_scores, score_interpretations)):
        plt.text(k, score + 0.01, interpretation, ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('silhouette_score_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Silhouette score analysis plots saved as 'silhouette_score_analysis.png'")
    
    # Explain what a high silhouette score indicates
    print("\n" + "-"*50)
    print("WHAT A HIGH SILHOUETTE SCORE INDICATES:")
    print("-"*50)
    print("1. CLUSTER QUALITY:")
    print("   - High score = clusters are well-separated and cohesive")
    print("   - Low score = clusters overlap or are poorly defined")
    print("   - Score range: -1 (poor) to +1 (excellent)")
    
    print("\n2. INTERPRETATION:")
    print("   - Score ≥ 0.7: Strong clustering structure")
    print("   - Score ≥ 0.5: Reasonable clustering structure")
    print("   - Score ≥ 0.25: Weak clustering structure")
    print("   - Score < 0.25: Poor clustering structure")
    
    print("\n3. CLUSTERING INSIGHTS:")
    print("   - High score = clear player segments identified")
    print("   - Low score = players don't naturally group well")
    print("   - Optimal k maximizes both separation and cohesion")
    
    return silhouette_scores, k_range, optimal_k_silhouette

def propose_optimal_k(wcss, silhouette_scores, k_range, optimal_k_silhouette):
    """
    Prompt 4.3: Propose optimal k based on both methods
    """
    print("\n" + "="*60)
    print("PROMPT 4.3: PROPOSING OPTIMAL K")
    print("="*60)
    
    # Analyze elbow method
    print("ELBOW METHOD ANALYSIS:")
    print("-" * 30)
    
    # Calculate improvement rates
    improvements = []
    for i in range(1, len(wcss)):
        improvement = (wcss[i-1] - wcss[i]) / wcss[i-1] * 100
        improvements.append(improvement)
        print(f"  k={i}→{i+1}: {improvement:.2f}% improvement")
    
    # Find elbow point (where improvement rate drops significantly)
    if len(improvements) >= 2:
        improvement_drops = [improvements[i-1] - improvements[i] for i in range(1, len(improvements))]
        elbow_k = improvements.index(max(improvements)) + 2  # +2 because we start from k=2
        print(f"  Elbow point identified at k={elbow_k}")
    else:
        elbow_k = 2
        print(f"  Limited data for elbow analysis, defaulting to k={elbow_k}")
    
    # Analyze silhouette scores
    print(f"\nSILHOUETTE SCORE ANALYSIS:")
    print("-" * 30)
    print(f"  Optimal k based on silhouette score: {optimal_k_silhouette}")
    print(f"  Best silhouette score: {max(silhouette_scores):.4f}")
    
    # Find all k values with good silhouette scores
    good_scores = [k for k, score in zip(k_range, silhouette_scores) if score >= 0.5]
    print(f"  K values with good scores (≥0.5): {good_scores}")
    
    # Propose optimal k
    print(f"\nOPTIMAL K RECOMMENDATION:")
    print("-" * 30)
    
    # Consider both methods
    if elbow_k == optimal_k_silhouette:
        recommended_k = optimal_k_silhouette
        confidence = "HIGH"
        reason = "Both methods agree on optimal k"
    elif abs(elbow_k - optimal_k_silhouette) <= 1:
        recommended_k = optimal_k_silhouette  # Prefer silhouette score
        confidence = "MEDIUM"
        reason = "Methods suggest similar k, preferring silhouette score"
    else:
        # Choose based on business context
        if max(silhouette_scores) >= 0.6:
            recommended_k = optimal_k_silhouette
            confidence = "MEDIUM"
            reason = "Silhouette score suggests good clustering quality"
        else:
            recommended_k = elbow_k
            confidence = "MEDIUM"
            reason = "Elbow method suggests simpler model"
    
    print(f"  Recommended optimal k: {recommended_k}")
    print(f"  Confidence: {confidence}")
    print(f"  Reasoning: {reason}")
    
    # Provide justification
    print(f"\nJUSTIFICATION FOR OPTIMAL K = {recommended_k}:")
    print("-" * 50)
    
    if recommended_k == optimal_k_silhouette:
        print("1. SILHOUETTE SCORE OPTIMIZATION:")
        print(f"   - k={recommended_k} achieves the highest silhouette score: {max(silhouette_scores):.4f}")
        print("   - Indicates well-separated and cohesive clusters")
        print("   - Players within clusters are similar, between clusters are different")
        
        if recommended_k in good_scores:
            print("   - Score ≥ 0.5 indicates reasonable clustering structure")
    
    if recommended_k == elbow_k:
        print("2. ELBOW METHOD VALIDATION:")
        print(f"   - k={recommended_k} represents the optimal complexity point")
        print("   - Beyond this point, adding clusters provides diminishing returns")
        print("   - Balances model complexity with performance improvement")
    
    print("3. BUSINESS INTERPRETABILITY:")
    print(f"   - {recommended_k} clusters provide manageable number of player segments")
    print("   - Each cluster can represent distinct player archetype")
    print("   - Actionable insights for game design and marketing")
    
    print("4. CLUSTERING STABILITY:")
    print("   - Moderate k values tend to be more stable across different runs")
    print("   - Reduces risk of overfitting to noise in the data")
    print("   - More robust for business applications")
    
    return recommended_k, confidence, reason

def save_analysis_results(wcss, silhouette_scores, k_range, optimal_k, confidence, reason):
    """Save the analysis results"""
    print("\n" + "="*60)
    print("SAVING ANALYSIS RESULTS")
    print("="*60)
    
    # Create analysis summary
    analysis_summary = {
        'optimal_k': optimal_k,
        'confidence': confidence,
        'reasoning': reason,
        'elbow_method': {
            'wcss_values': wcss,
            'improvement_rates': [(wcss[i-1] - wcss[i]) / wcss[i-1] * 100 for i in range(1, len(wcss))],
            'suggested_k': wcss.index(min(wcss[1:])) + 1 if len(wcss) > 1 else 1
        },
        'silhouette_analysis': {
            'scores': silhouette_scores,
            'k_values': list(k_range),
            'optimal_k_silhouette': k_range[np.argmax(silhouette_scores)] if len(silhouette_scores) > 0 else 2,
            'best_score': max(silhouette_scores) if len(silhouette_scores) > 0 else 0
        },
        'recommendation': {
            'optimal_k': optimal_k,
            'confidence': confidence,
            'reasoning': reason,
            'next_steps': [
                f"Apply K-Means clustering with k={optimal_k}",
                "Analyze cluster characteristics and player profiles",
                "Validate clusters with business domain knowledge",
                "Use clusters for player segmentation and targeting"
            ]
        }
    }
    
    import json
    with open('optimal_clusters_analysis.json', 'w') as f:
        json.dump(analysis_summary, f, indent=2)
    print("✅ Analysis results saved as 'optimal_clusters_analysis.json'")
    
    return analysis_summary

def main():
    """Main execution function"""
    print("🚀 PHASE 4: DETERMINING OPTIMAL NUMBER OF CLUSTERS")
    print("=" * 60)
    
    # Load the engineered data
    df_players = load_engineered_data()
    if df_players is None:
        return
    
    # Prepare data for clustering
    X, clustering_cols = prepare_data_for_clustering(df_players)
    
    # Step 1: Elbow Method Analysis
    wcss, k_range = elbow_method_analysis(X, max_k=10)
    
    # Step 2: Silhouette Score Analysis
    silhouette_scores, silhouette_k_range, optimal_k_silhouette = silhouette_score_analysis(X, max_k=10)
    
    # Step 3: Propose Optimal K
    optimal_k, confidence, reason = propose_optimal_k(wcss, silhouette_scores, k_range, optimal_k_silhouette)
    
    # Step 4: Save Analysis Results
    analysis_summary = save_analysis_results(wcss, silhouette_scores, k_range, optimal_k, confidence, reason)
    
    print("\n" + "🎉 PHASE 4 COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nFinal Recommendation:")
    print(f"  Optimal number of clusters (k): {optimal_k}")
    print(f"  Confidence level: {confidence}")
    print(f"  Reasoning: {reason}")
    
    print(f"\nAnalysis Results:")
    print(f"  Elbow method plots saved as 'elbow_method_analysis.png'")
    print(f"  Silhouette analysis plots saved as 'silhouette_score_analysis.png'")
    print(f"  Complete analysis saved as 'optimal_clusters_analysis.json'")
    
    print(f"\nNext steps:")
    print(f"  • Apply K-Means clustering with k={optimal_k}")
    print(f"  • Analyze cluster characteristics")
    print(f"  • Create player profiles for each cluster")
    print(f"  • Use insights for business applications")

if __name__ == "__main__":
    main()
