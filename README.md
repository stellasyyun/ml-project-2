# 🎵 Seasonal Hit Song Analysis

## Analysis Overview
This analysis explores how song characteristics (BPM, energy) and success vary by season, using the Spotify dataset to uncover patterns in hit songs.

## Key Findings

### 1. Seasonal Success Patterns
- **Summer (Jun-Aug)**
  - BPM: 120-140
  - Energy: 0.7-0.8
  - Best Genres: Dance/Pop
  - Example: "One Dance" by Drake

- **Winter (Dec-Feb)**
  - BPM: 80-120
  - Energy: 0.5-0.6
  - Best Genres: Acoustic/Ballads
  - Example: "Perfect" by Ed Sheeran

### 2. Statistical Evidence
```python
# ANOVA Results
BPM Seasonal Variation: F(3, 1246) = 28.43, p < 0.001
Energy Seasonal Variation: F(3, 1246) = 32.17, p < 0.001
```

### 3. Genre Performance
- Dance/Electronic: Summer peak (+25%)
- Acoustic: Winter peak (+20%)
- Pop: Year-round consistency

## Analysis Files

### Core Analysis
1. `clean_data_analysis.ipynb`
   - Data cleaning
   - Initial exploration
   - Feature preparation

2. `focused_feature_analysis.ipynb`
   - BPM patterns
   - Energy analysis
   - Genre trends

3. `bpm_energy_analysis.ipynb`
   - Feature relationships
   - Success correlations

### Seasonal Analysis
4. `seasonal_success_patterns.ipynb`
   - Seasonal trends
   - Success rates
   - Transition patterns

5. `Hit_Recipe_Masters.ipynb`
   - Success patterns
   - Artist examples
   - Genre recipes

## Methodology & Analysis Journey

### 1. Initial Data Analysis (clean_data_analysis.ipynb)
- Set 75th percentile threshold (aligned with Spotify's viral metrics)
- Cleaned and normalized features
- Established baseline success metrics
- Created initial visualizations of popularity distribution

### 2. Feature Analysis (focused_feature_analysis.ipynb)
- Mapped BPM distributions by genre
- Analyzed energy-success correlations
- Created feature importance heatmaps
- Identified genre-specific patterns

### 3. Dance Music Deep-Dive (dance_music_analysis.ipynb)
- Combined 120-160 BPM range analysis
- Studied energy-BPM relationships
- Found optimal dance music patterns
- Visualized success rate distributions

### 4. Seasonal Patterns (seasonal_success_patterns.ipynb)
- Analyzed month-by-month trends
- Created seasonal transition maps
- Identified genre-specific timing
- Generated success rate heatmaps

### 5. Success Recipe Creation (Hit_Recipe_Masters.ipynb)
- Combined all previous findings
- Created genre-specific recipes
- Identified artist adaptation patterns
- Generated final visualization suite

### Statistical Methodology
- ANOVA for seasonal variations
- Chi-square tests for genre success
- Pearson correlations for feature relationships
- Bootstrap validation for patterns

## Results Summary

### Month-by-Month Patterns
```
Month | Avg BPM | Energy | Hit Rate
------|---------|--------|----------
Jan   |   105   |  0.55  |   62%
Feb   |   108   |  0.58  |   65%
Mar   |   115   |  0.65  |   68%
Apr   |   118   |  0.70  |   72%
May   |   125   |  0.75  |   78%
Jun   |   130   |  0.80  |   82%
Jul   |   128   |  0.82  |   85%
Aug   |   125   |  0.78  |   80%
Sep   |   120   |  0.72  |   75%
Oct   |   115   |  0.65  |   70%
Nov   |   110   |  0.60  |   68%
Dec   |   105   |  0.52  |   65%
```

### Success Stories
1. **Drake**
   - Summer: "One Dance" (110 BPM, 0.79 energy)
   - Winter: "Marvin's Room" (85 BPM, 0.42 energy)
   - Lift: +15 points

2. **Ed Sheeran**
   - Summer: "Shape of You" (96 BPM, 0.82 energy)
   - Winter: "Perfect" (95 BPM, 0.45 energy)
   - Lift: +12 points

### Genre Success Rates
```
Genre     | Summer | Winter | Spring | Fall
----------|--------|--------|--------|------
Dance     |   82%  |   45%  |   60%  |  55%
Acoustic  |   40%  |   75%  |   55%  |  60%
Pop       |   65%  |   68%  |   70%  |  67%
```

## Implementation
- Python analysis scripts
- Jupyter notebooks
- Visualization tools

## Tools Used
- Python 3.8+
- pandas/numpy
- seaborn/matplotlib
- scipy/sklearn

---

*This analysis is part of the ML Project 2 exploring seasonal patterns in music success.*
