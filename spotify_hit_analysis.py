#!/usr/bin/env python3
"""
Spotify Hit Recipe Analysis
--------------------------
A comprehensive analysis of what makes songs successful across different seasons,
examining BPM, energy levels, and genre patterns.

Author: Cline
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Configure visualizations
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = [12, 6]
sns.set_style('whitegrid')

# Custom colors for seasonal visualizations
SEASON_COLORS = {
    'Summer': '#FF9933',  # Warm orange
    'Winter': '#66CCFF',  # Cool blue
    'Spring': '#99CC33',  # Fresh green
    'Fall': '#CC6633'     # Autumn brown
}

def load_and_prepare_data():
    """Load and prepare the Spotify dataset with seasonal classifications."""
    # Load data
    sheet_url = "https://docs.google.com/spreadsheets/d/1ae96nZRL_kJWb_EEv2avxMOgGClvyc77SpY-VBqVGiY/edit#gid=1052928543"
    csv_export_url = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
    df = pd.read_csv(csv_export_url)
    
    # Convert release date and extract season
    df['release_date'] = pd.to_datetime(df['track_album_release_date'])
    df['month'] = df['release_date'].dt.month
    
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    df['season'] = df['month'].apply(get_season)
    
    # Calculate artist metrics
    artist_avg_popularity = df.groupby('track_artist')['track_popularity'].mean()
    df['artist_avg_popularity'] = df['track_artist'].map(artist_avg_popularity)
    df['popularity_lift'] = df['track_popularity'] - df['artist_avg_popularity']
    
    # Define hit threshold (top 25%)
    hit_threshold = df['track_popularity'].quantile(0.75)
    df['is_hit'] = df['track_popularity'] >= hit_threshold
    
    return df

def analyze_seasonal_patterns(df):
    """Analyze how BPM and energy patterns vary by season."""
    seasonal_patterns = df[df['is_hit']].groupby('season').agg({
        'tempo': ['mean', 'std'],
        'energy': ['mean', 'std'],
        'track_popularity': ['mean', 'count']
    }).round(2)
    
    print("\n=== Seasonal Patterns in Hit Songs ===")
    print(seasonal_patterns)
    
    # Plot distributions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    for season in ['Winter', 'Spring', 'Summer', 'Fall']:
        season_hits = df[(df['season'] == season) & (df['is_hit'])]
        
        # BPM Distribution
        sns.kdeplot(data=season_hits, x='tempo', ax=ax1, 
                   color=SEASON_COLORS[season], label=season)
        
        # Energy Distribution
        sns.kdeplot(data=season_hits, x='energy', ax=ax2,
                   color=SEASON_COLORS[season], label=season)
    
    ax1.set_title('BPM Distribution by Season (Hit Songs)')
    ax2.set_title('Energy Distribution by Season (Hit Songs)')
    plt.tight_layout()
    plt.show()

def find_seasonal_masters(df):
    """Identify artists who successfully adapt to seasons."""
    # Find artists with significant seasonal lifts
    seasonal_masters = df[df['popularity_lift'] >= 10].groupby('track_artist').agg({
        'popularity_lift': ['max', 'count'],
        'track_popularity': 'max',
        'artist_avg_popularity': 'first'
    }).round(2)
    
    # Filter for consistent performers
    seasonal_masters = seasonal_masters[
        seasonal_masters[('popularity_lift', 'count')] >= 3
    ].sort_values(('popularity_lift', 'max'), ascending=False)
    
    print("\n=== Master Seasonal Adapters ===")
    for artist in seasonal_masters.head().index:
        artist_data = df[df['track_artist'] == artist]
        best_hit = artist_data.loc[artist_data['popularity_lift'].idxmax()]
        
        print(f"\n🌟 {artist}")
        print(f"Best Hit: {best_hit['track_name']}")
        print(f"Season: {best_hit['season']}")
        print(f"Popularity Lift: +{best_hit['popularity_lift']:.1f}")
        print(f"Recipe: {best_hit['tempo']:.0f} BPM, {best_hit['energy']:.2f} Energy")

def analyze_genre_seasonality(df):
    """Analyze how genres perform across seasons."""
    genre_season_success = df.groupby(['playlist_subgenre', 'season']).agg({
        'track_popularity': ['mean', 'max'],
        'tempo': 'mean',
        'energy': 'mean',
        'track_name': 'count'
    }).round(2)
    
    print("\n=== Genre Seasonal Performance ===")
    for season in ['Summer', 'Winter', 'Spring', 'Fall']:
        print(f"\n{season}'s Top Genres:")
        season_data = genre_season_success.xs(season, level=1)
        top_genres = season_data.nlargest(3, ('track_popularity', 'mean'))
        
        for genre in top_genres.index:
            stats = top_genres.loc[genre]
            print(f"\n{genre}")
            print(f"Average Popularity: {stats[('track_popularity', 'mean')]:.1f}")
            print(f"Recipe: {stats['tempo']:.0f} BPM, {stats['energy']:.2f} Energy")

def find_seasonal_recipes(df):
    """Identify winning BPM-energy combinations by season."""
    print("\n=== Seasonal Hit Recipes ===")
    for season in ['Summer', 'Winter', 'Spring', 'Fall']:
        season_hits = df[(df['season'] == season) & df['is_hit']]
        
        # Calculate optimal ranges
        bpm_range = season_hits['tempo'].agg(['mean', 'std']).round(1)
        energy_range = season_hits['energy'].agg(['mean', 'std']).round(2)
        
        print(f"\n{season} Recipe:")
        print(f"BPM: {bpm_range['mean']} ± {bpm_range['std']}")
        print(f"Energy: {energy_range['mean']} ± {energy_range['std']}")
        
        # Find perfect examples
        perfect_hits = season_hits[
            (abs(season_hits['tempo'] - bpm_range['mean']) <= bpm_range['std']) &
            (abs(season_hits['energy'] - energy_range['mean']) <= energy_range['std'])
        ].sort_values('track_popularity', ascending=False)
        
        if len(perfect_hits) > 0:
            example = perfect_hits.iloc[0]
            print(f"Example: {example['track_name']} by {example['track_artist']}")

def main():
    """Run the complete analysis pipeline."""
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    
    print("\nAnalyzing seasonal patterns...")
    analyze_seasonal_patterns(df)
    
    print("\nFinding seasonal masters...")
    find_seasonal_masters(df)
    
    print("\nAnalyzing genre seasonality...")
    analyze_genre_seasonality(df)
    
    print("\nIdentifying seasonal recipes...")
    find_seasonal_recipes(df)

if __name__ == "__main__":
    main()
