#!/usr/bin/env python3
"""
Phase 7: Strategic Implications Discussion
This script translates analytical findings into actionable business strategies and considers ethical aspects
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

def load_cluster_insights():
    """Load cluster insights and narratives"""
    try:
        # Load player narratives
        narratives_df = pd.read_csv('player_cluster_narratives.csv')
        
        # Load statistical profiles
        stats_df = pd.read_csv('cluster_statistical_profiles.csv')
        
        # Load clustered data for additional analysis
        df_players = pd.read_pickle("df_players_clustered.pkl")
        
        print("✅ Cluster insights loaded successfully")
        return narratives_df, stats_df, df_players
    except FileNotFoundError as e:
        print(f"❌ Required files not found: {e}")
        print("Please run Phase 6 first to generate cluster insights.")
        return None, None, None

def analyze_game_design_implications(narratives_df, stats_df, df_players):
    """
    Prompt 7.1: Informing Game Design Decisions
    """
    print("\n" + "="*80)
    print("PROMPT 7.1: INFORMING GAME DESIGN DECISIONS")
    print("="*80)
    
    print("ANALYZING HOW PLAYER ARCHETYPES INFORM STRATEGIC GAME DESIGN...")
    print("-" * 70)
    
    # Create comprehensive game design recommendations
    design_recommendations = {
        'Casual Players (Socializers/Explorers)': {
            'target_percentage': 98.7,
            'key_characteristics': ['High PlayTimeHours', 'Low SessionsPerWeek', 'Low PlayerLevel', 'High AchievementsUnlocked'],
            'design_focus': 'Retention and Social Engagement',
            'recommended_features': [
                'Quick Play Modes: Short, satisfying sessions under 15 minutes',
                'Social Features: Friend systems, guilds, and chat functionality',
                'Achievement Systems: Easy-to-unlock achievements with social sharing',
                'Tutorial Systems: Comprehensive onboarding with progress tracking',
                'Daily Rewards: Consistent engagement through daily login bonuses',
                'Hidden Areas: Exploration rewards for curious players',
                'Casual Competitive: Low-stakes competitive modes with rewards'
            ],
            'content_strategy': 'Focus on accessibility and social connection',
            'difficulty_curve': 'Gentle progression with multiple success paths'
        },
        'Moderate Players (Achievers/Explorers)': {
            'target_percentage': 0.6,
            'key_characteristics': ['High PlayTimeHours', 'High SessionsPerWeek', 'Low PlayerLevel', 'High InGamePurchases'],
            'design_focus': 'Content Variety and Achievement Systems',
            'recommended_features': [
                'Achievement Systems: Tiered achievements with clear progression paths',
                'Content Variety: Multiple game modes and storylines',
                'Progression Systems: Visible level progression with meaningful rewards',
                'Exploration Content: Hidden areas and secret achievements',
                'Moderate Challenge: Balanced difficulty that rewards skill',
                'Monetization Features: Cosmetic items and convenience features',
                'Social Competition: Leaderboards and achievement comparisons'
            ],
            'content_strategy': 'Balance challenge with accessibility',
            'difficulty_curve': 'Moderate difficulty with clear skill progression'
        },
        'Hardcore Players (Achievers/Killers)': {
            'target_percentage': 0.7,
            'key_characteristics': ['Low PlayTimeHours', 'Low SessionsPerWeek', 'High PlayerLevel', 'High InGamePurchases'],
            'design_focus': 'Skill Mastery and Competitive Excellence',
            'recommended_features': [
                'Advanced Mechanics: Complex systems requiring mastery',
                'Competitive Modes: Ranked play, tournaments, and esports integration',
                'Skill-Based Progression: Challenging content that tests expertise',
                'Premium Content: High-value items and exclusive features',
                'Leaderboards: Global and regional competitive rankings',
                'Meta Analysis: Deep strategic elements and theorycrafting',
                'Exclusive Rewards: Unique items for top performers'
            ],
            'content_strategy': 'Focus on skill development and competition',
            'difficulty_curve': 'Steep difficulty with high skill ceilings'
        }
    }
    
    # Display detailed design recommendations
    for archetype, recommendations in design_recommendations.items():
        print(f"\n🎮 {archetype.upper()}")
        print("=" * 60)
        print(f"Target Population: {recommendations['target_percentage']}% of player base")
        print(f"Design Focus: {recommendations['design_focus']}")
        print(f"Key Characteristics: {', '.join(recommendations['key_characteristics'])}")
        print(f"Content Strategy: {recommendations['content_strategy']}")
        print(f"Difficulty Curve: {recommendations['difficulty_curve']}")
        
        print("\n📋 RECOMMENDED FEATURES:")
        for i, feature in enumerate(recommendations['recommended_features'], 1):
            print(f"  {i}. {feature}")
    
    # Create feature priority matrix
    create_feature_priority_matrix(design_recommendations)
    
    # Save design recommendations
    save_game_design_recommendations(design_recommendations)
    
    return design_recommendations

def create_feature_priority_matrix(design_recommendations):
    """Create a feature priority matrix visualization"""
    print("\n📊 CREATING FEATURE PRIORITY MATRIX...")
    
    # Define features and their impact on each archetype
    features = [
        'Quick Play Modes', 'Social Features', 'Achievement Systems', 'Tutorial Systems',
        'Daily Rewards', 'Hidden Areas', 'Competitive Modes', 'Content Variety',
        'Progression Systems', 'Monetization Features', 'Advanced Mechanics',
        'Premium Content', 'Leaderboards', 'Meta Analysis', 'Exclusive Rewards'
    ]
    
    # Create impact matrix (1=Low, 2=Medium, 3=High)
    impact_matrix = {
        'Casual Players': [3, 3, 2, 3, 3, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1],
        'Moderate Players': [2, 2, 3, 2, 2, 3, 2, 3, 3, 2, 2, 2, 2, 2, 2],
        'Hardcore Players': [1, 1, 2, 1, 1, 2, 3, 2, 2, 3, 3, 3, 3, 3, 3]
    }
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Prepare data for heatmap
    impact_data = np.array([impact_matrix['Casual Players'], 
                           impact_matrix['Moderate Players'], 
                           impact_matrix['Hardcore Players']])
    
    # Create heatmap
    sns.heatmap(impact_data, 
                annot=True, 
                fmt='d',
                xticklabels=features,
                yticklabels=['Casual', 'Moderate', 'Hardcore'],
                cmap='RdYlGn_r',
                cbar_kws={'label': 'Feature Impact Level (1=Low, 2=Medium, 3=High)'},
                ax=ax)
    
    ax.set_title('Feature Priority Matrix: Impact on Player Archetypes', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Game Features', fontsize=12)
    ax.set_ylabel('Player Archetypes', fontsize=12)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig('feature_priority_matrix.png', dpi=300, bbox_inches='tight')
    print("✅ Feature priority matrix saved as 'feature_priority_matrix.png'")

def analyze_marketing_monetization_strategies(narratives_df, stats_df, df_players):
    """
    Prompt 7.2: Tailoring Marketing and Monetization Strategies
    """
    print("\n" + "="*80)
    print("PROMPT 7.2: TAILORING MARKETING AND MONETIZATION STRATEGIES")
    print("="*80)
    
    print("ANALYZING MARKETING AND MONETIZATION STRATEGIES FOR PLAYER SEGMENTS...")
    print("-" * 70)
    
    # Create comprehensive marketing and monetization strategies
    marketing_strategies = {
        'Casual Players (98.7%)': {
            'marketing_approach': 'Retention-Focused Marketing',
            'key_messages': [
                'Easy to learn, fun to play',
                'Connect with friends and family',
                'Quick entertainment during breaks',
                'No pressure, just fun'
            ],
            'channels': [
                'Social media platforms (Facebook, Instagram)',
                'Mobile app stores',
                'Word-of-mouth campaigns',
                'Family-friendly content creators'
            ],
            'monetization_strategy': 'Low-Cost Retention Monetization',
            'monetization_features': [
                'Ad-supported free play with premium ad-free option',
                'Cosmetic items (skins, emotes)',
                'Convenience features (extra lives, hints)',
                'Season passes with achievable goals',
                'Social gifting systems'
            ],
            'pricing_model': 'Freemium with microtransactions',
            'target_arpu': '$2-5 per month',
            'retention_focus': 'Daily engagement and social features'
        },
        'Moderate Players (0.6%)': {
            'marketing_approach': 'Engagement and Progression Marketing',
            'key_messages': [
                'Master new challenges and skills',
                'Discover hidden content and secrets',
                'Compete with friends on leaderboards',
                'Build your gaming legacy'
            ],
            'channels': [
                'Gaming communities (Reddit, Discord)',
                'Gaming influencers and streamers',
                'Gaming websites and forums',
                'Email marketing campaigns'
            ],
            'monetization_strategy': 'Content and Achievement Monetization',
            'monetization_features': [
                'Expansion packs and DLC content',
                'Premium achievement systems',
                'Exclusive cosmetic collections',
                'Battle passes with progression rewards',
                'Early access to new features'
            ],
            'pricing_model': 'Premium content with subscription options',
            'target_arpu': '$15-25 per month',
            'retention_focus': 'Content variety and achievement systems'
        },
        'Hardcore Players (0.7%)': {
            'marketing_approach': 'Elite Performance Marketing',
            'key_messages': [
                'Prove your mastery and skill',
                'Compete at the highest levels',
                'Access exclusive competitive content',
                'Join the elite gaming community'
            ],
            'channels': [
                'Esports platforms and tournaments',
                'Professional gaming teams',
                'High-level gaming content creators',
                'Gaming conventions and events',
                'Specialized gaming media'
            ],
            'monetization_strategy': 'Premium Competitive Monetization',
            'monetization_features': [
                'Premium competitive modes and tournaments',
                'Exclusive skill-based rewards',
                'Professional-grade analytics and tools',
                'VIP membership programs',
                'Limited edition competitive items',
                'Direct tournament entry fees'
            ],
            'pricing_model': 'Premium subscription with competitive fees',
            'target_arpu': '$50-100+ per month',
            'retention_focus': 'Skill development and competitive excellence'
        }
    }
    
    # Display marketing strategies
    for segment, strategy in marketing_strategies.items():
        print(f"\n🎯 {segment.upper()}")
        print("=" * 60)
        print(f"Marketing Approach: {strategy['marketing_approach']}")
        print(f"Target ARPU: {strategy['target_arpu']}")
        print(f"Retention Focus: {strategy['retention_focus']}")
        
        print(f"\n📢 KEY MARKETING MESSAGES:")
        for i, message in enumerate(strategy['key_messages'], 1):
            print(f"  {i}. {message}")
        
        print(f"\n📺 MARKETING CHANNELS:")
        for i, channel in enumerate(strategy['channels'], 1):
            print(f"  {i}. {channel}")
        
        print(f"\n💰 MONETIZATION FEATURES:")
        for i, feature in enumerate(strategy['monetization_features'], 1):
            print(f"  {i}. {feature}")
    
    # Create monetization potential analysis
    create_monetization_potential_analysis(marketing_strategies, df_players)
    
    # Save marketing strategies
    save_marketing_strategies(marketing_strategies)
    
    return marketing_strategies

def create_monetization_potential_analysis(marketing_strategies, df_players):
    """Create monetization potential analysis visualization"""
    print("\n📊 CREATING MONETIZATION POTENTIAL ANALYSIS...")
    
    # Calculate potential revenue by segment
    segments = ['Casual Players', 'Moderate Players', 'Hardcore Players']
    population_percentages = [98.7, 0.6, 0.7]
    target_arpu = [3.5, 20, 75]  # Average monthly revenue per user
    
    # Calculate potential monthly revenue
    total_players = len(df_players)
    potential_revenue = []
    
    for i, percentage in enumerate(population_percentages):
        segment_players = (percentage / 100) * total_players
        segment_revenue = segment_players * target_arpu[i]
        potential_revenue.append(segment_revenue)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Pie chart of population distribution
    ax1.pie(population_percentages, labels=segments, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Player Population Distribution', fontsize=14, fontweight='bold')
    
    # Bar chart of potential revenue
    bars = ax2.bar(segments, potential_revenue, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    ax2.set_title('Potential Monthly Revenue by Segment', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Potential Monthly Revenue ($)', fontsize=12)
    ax2.set_xlabel('Player Segments', fontsize=12)
    
    # Add value labels on bars
    for bar, revenue in zip(bars, potential_revenue):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(potential_revenue)*0.01,
                f'${revenue:,.0f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('monetization_potential_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Monetization potential analysis saved as 'monetization_potential_analysis.png'")
    
    # Print revenue insights
    print(f"\n💰 MONETIZATION INSIGHTS:")
    print(f"Total Potential Monthly Revenue: ${sum(potential_revenue):,.0f}")
    for i, segment in enumerate(segments):
        print(f"{segment}: ${potential_revenue[i]:,.0f} ({population_percentages[i]}% of players)")

def discuss_ethical_considerations():
    """
    Prompt 7.3: Ethical Considerations
    """
    print("\n" + "="*80)
    print("PROMPT 7.3: ETHICAL CONSIDERATIONS")
    print("="*80)
    
    print("DISCUSSING ETHICAL CONSIDERATIONS IN PLAYER BEHAVIOR SEGMENTATION...")
    print("-" * 70)
    
    ethical_considerations = {
        'Data Privacy and Consent': {
            'concerns': [
                'Transparency about data collection and usage',
                'Explicit consent for behavioral analysis',
                'Right to opt-out of personalized features',
                'Data retention and deletion policies'
            ],
            'mitigation': [
                'Clear privacy policies and terms of service',
                'Granular consent options for different data uses',
                'Easy-to-use privacy controls and settings',
                'Regular privacy audits and compliance checks'
            ]
        },
        'Manipulation and Addiction': {
            'concerns': [
                'Using psychological triggers to increase engagement',
                'Targeting vulnerable players with monetization',
                'Creating addictive gameplay loops',
                'Exploiting player psychology for profit'
            ],
            'mitigation': [
                'Implementing responsible gaming features',
                'Setting spending limits and cool-down periods',
                'Providing tools for healthy gaming habits',
                'Regular breaks and session time warnings'
            ]
        },
        'Fairness and Inclusion': {
            'concerns': [
                'Algorithmic bias in player segmentation',
                'Exclusion of certain player demographics',
                'Unfair advantage through personalization',
                'Discrimination based on player behavior'
            ],
            'mitigation': [
                'Regular bias testing and algorithm auditing',
                'Diverse development and testing teams',
                'Transparent algorithm explanations',
                'Appeal processes for automated decisions'
            ]
        },
        'Data Security': {
            'concerns': [
                'Data breaches and unauthorized access',
                'Third-party data sharing risks',
                'Insecure data storage and transmission',
                'Employee access to sensitive player data'
            ],
            'mitigation': [
                'Encryption of all sensitive data',
                'Regular security audits and penetration testing',
                'Strict access controls and monitoring',
                'Compliance with data protection regulations'
            ]
        }
    }
    
    # Display ethical considerations
    for category, details in ethical_considerations.items():
        print(f"\n⚖️ {category.upper()}")
        print("-" * 50)
        
        print("🚨 CONCERNS:")
        for i, concern in enumerate(details['concerns'], 1):
            print(f"  {i}. {concern}")
        
        print("\n✅ MITIGATION STRATEGIES:")
        for i, strategy in enumerate(details['mitigation'], 1):
            print(f"  {i}. {strategy}")
    
    # Create ethical framework recommendations
    create_ethical_framework_recommendations()
    
    # Save ethical considerations
    save_ethical_considerations(ethical_considerations)
    
    return ethical_considerations

def create_ethical_framework_recommendations():
    """Create ethical framework recommendations"""
    print("\n🏛️ ETHICAL FRAMEWORK RECOMMENDATIONS:")
    print("-" * 50)
    
    framework = {
        'Data Governance': [
            'Establish a Data Ethics Committee',
            'Implement Privacy by Design principles',
            'Regular ethical impact assessments',
            'Transparent data usage reporting'
        ],
        'Player Protection': [
            'Age-appropriate content and features',
            'Spending limits and parental controls',
            'Mental health and wellness resources',
            'Anti-addiction measures and warnings'
        ],
        'Algorithmic Transparency': [
            'Explainable AI for player segmentation',
            'Regular algorithm bias testing',
            'Player access to their behavioral data',
            'Human oversight of automated decisions'
        ],
        'Community Standards': [
            'Clear community guidelines',
            'Fair and consistent enforcement',
            'Player feedback and appeal processes',
            'Regular policy reviews and updates'
        ]
    }
    
    for category, recommendations in framework.items():
        print(f"\n📋 {category}:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")

def save_game_design_recommendations(design_recommendations):
    """Save game design recommendations to file"""
    recommendations_data = []
    
    for archetype, recs in design_recommendations.items():
        recommendations_data.append({
            'Player_Archetype': archetype,
            'Target_Percentage': recs['target_percentage'],
            'Design_Focus': recs['design_focus'],
            'Key_Characteristics': '; '.join(recs['key_characteristics']),
            'Content_Strategy': recs['content_strategy'],
            'Difficulty_Curve': recs['difficulty_curve'],
            'Recommended_Features': '; '.join(recs['recommended_features'])
        })
    
    recommendations_df = pd.DataFrame(recommendations_data)
    recommendations_df.to_csv('game_design_recommendations.csv', index=False)
    print("✅ Game design recommendations saved to 'game_design_recommendations.csv'")

def save_marketing_strategies(marketing_strategies):
    """Save marketing strategies to file"""
    strategies_data = []
    
    for segment, strategy in marketing_strategies.items():
        strategies_data.append({
            'Player_Segment': segment,
            'Marketing_Approach': strategy['marketing_approach'],
            'Key_Messages': '; '.join(strategy['key_messages']),
            'Marketing_Channels': '; '.join(strategy['channels']),
            'Monetization_Strategy': strategy['monetization_strategy'],
            'Monetization_Features': '; '.join(strategy['monetization_features']),
            'Pricing_Model': strategy['pricing_model'],
            'Target_ARPU': strategy['target_arpu'],
            'Retention_Focus': strategy['retention_focus']
        })
    
    strategies_df = pd.DataFrame(strategies_data)
    strategies_df.to_csv('marketing_monetization_strategies.csv', index=False)
    print("✅ Marketing strategies saved to 'marketing_monetization_strategies.csv'")

def save_ethical_considerations(ethical_considerations):
    """Save ethical considerations to file"""
    ethics_data = []
    
    for category, details in ethical_considerations.items():
        ethics_data.append({
            'Ethical_Category': category,
            'Concerns': '; '.join(details['concerns']),
            'Mitigation_Strategies': '; '.join(details['mitigation'])
        })
    
    ethics_df = pd.DataFrame(ethics_data)
    ethics_df.to_csv('ethical_considerations.csv', index=False)
    print("✅ Ethical considerations saved to 'ethical_considerations.csv'")

def main():
    """Main execution function"""
    print("🚀 PHASE 7: STRATEGIC IMPLICATIONS DISCUSSION")
    print("=" * 80)
    
    # Load cluster insights
    narratives_df, stats_df, df_players = load_cluster_insights()
    if narratives_df is None:
        return
    
    print(f"📊 Analyzing strategic implications for {len(df_players):,} players across 3 clusters...")
    
    # Step 1: Game Design Implications
    design_recommendations = analyze_game_design_implications(narratives_df, stats_df, df_players)
    
    # Step 2: Marketing and Monetization Strategies
    marketing_strategies = analyze_marketing_monetization_strategies(narratives_df, stats_df, df_players)
    
    # Step 3: Ethical Considerations
    ethical_considerations = discuss_ethical_considerations()
    
    print("\n" + "🎉 PHASE 7 COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print(f"\nStrategic Analysis Completed:")
    print(f"  • Game Design Recommendations: {len(design_recommendations)} archetypes analyzed")
    print(f"  • Marketing Strategies: {len(marketing_strategies)} segments strategized")
    print(f"  • Ethical Considerations: {len(ethical_considerations)} categories addressed")
    
    print(f"\nOutput Files Generated:")
    print(f"  • game_design_recommendations.csv - Strategic game design insights")
    print(f"  • marketing_monetization_strategies.csv - Marketing and monetization strategies")
    print(f"  • ethical_considerations.csv - Ethical framework and considerations")
    print(f"  • feature_priority_matrix.png - Feature impact visualization")
    print(f"  • monetization_potential_analysis.png - Revenue potential analysis")
    
    print(f"\n🎯 KEY STRATEGIC INSIGHTS:")
    print(f"  • Casual Players (98.7%): Focus on retention and social engagement")
    print(f"  • Moderate Players (0.6%): Content variety and achievement systems")
    print(f"  • Hardcore Players (0.7%): Skill mastery and competitive excellence")
    
    print(f"\n💰 MONETIZATION POTENTIAL:")
    print(f"  • High-value segments represent significant revenue opportunities")
    print(f"  • Personalized strategies can increase player lifetime value")
    print(f"  • Ethical considerations are crucial for sustainable success")
    
    print(f"\n🚀 NEXT STEPS:")
    print(f"  • Implement recommended game design features")
    print(f"  • Execute targeted marketing campaigns")
    print(f"  • Establish ethical data governance framework")
    print(f"  • Monitor and optimize strategies based on player feedback")

if __name__ == "__main__":
    main()
