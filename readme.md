# CP3106 Independent Study: Football Match Outcome Prediction — A Reproducibility Study and Multi-Model Comparison

## Abstract

This project conducts a systematic reproducibility study and multi-model comparison for football (soccer) match outcome prediction. We collect match data from five major European leagues (English Premier League, LaLiga, Serie A, Bundesliga, Ligue 1) spanning the 2019–2025 seasons via the API-Football (v3) service, totalling **10,627 fixtures** and **106,270 in-play checkpoint samples**. We engineer a rich set of **76+ pre-match features** and **89 in-play features**, then benchmark a comprehensive set of models: classical machine learning (Logistic Regression, SVM, KNN, Naïve Bayes, Random Forest, Extra Trees), gradient-boosted trees (XGBoost, CatBoost, HistGradientBoosting), deep learning (MLP, ResNet-style, LSTM with attention), a hybrid Quantum Neural Network (QNN), Bayesian Networks, and edge-computing fusion architectures. We also critically reproduce findings from multiple published papers, revealing that many reported high accuracies (70%+) rely on post-match features or data leakage, while strict pre-match prediction consistently plateaus at **~49–51% accuracy** across all model families. Only in-play models, which incorporate real-time match events, achieve meaningfully higher performance, reaching **70.5% overall accuracy** and **99.3% accuracy at minute 90**. These findings highlight a significant gap between claims in the literature and practically achievable pre-match prediction performance.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Literature Review and Reproducibility Analysis](#2-literature-review-and-reproducibility-analysis)
3. [Data Collection and Processing](#3-data-collection-and-processing)
4. [Feature Engineering](#4-feature-engineering)
5. [Experimental Methodology](#5-experimental-methodology)
6. [Pre-Match Model Results](#6-pre-match-model-results)
7. [In-Play Model Results](#7-in-play-model-results)
8. [Paper Reproduction Experiments](#8-paper-reproduction-experiments)
9. [Misclassification Analysis](#9-misclassification-analysis)
10. [Discussion](#10-discussion)
11. [Conclusion](#11-conclusion)
12. [Project Structure](#12-project-structure)
13. [How to Run](#13-how-to-run)
14. [References](#14-references)

---

## 1. Introduction

### 1.1 Background

Football match outcome prediction is a well-studied problem in sports analytics and machine learning. The task is typically formulated as a three-class classification problem: **Home Win (H)**, **Draw (D)**, or **Away Win (A)**. Despite extensive research, this remains a fundamentally difficult problem because:

- **Class imbalance**: Home wins typically account for ~42% of outcomes, draws ~25%, and away wins ~33%, making draw prediction particularly challenging.
- **High stochasticity**: Football is a low-scoring sport with significant randomness — a single lucky deflection or referee decision can determine the outcome.
- **Information asymmetry**: The most predictive features (goals scored, shots on target, possession) are only available during or after the match, not before.

### 1.2 Motivation

Many published papers in this domain report impressive accuracy figures (70–85%), but upon careful examination, we found that these results often stem from:

1. **Post-match feature leakage**: Using features like shots, corners, fouls, or half-time goals that are only available after the match has started.
2. **Label leakage**: Including goal difference or full-time scores as input features.
3. **Unrealistic evaluation**: Using random cross-validation instead of temporal holdout, allowing the model to "see" future matches during training.
4. **Tiny, unrepresentative datasets**: Training on a single league's single season (~380 matches) and overfitting to specific team names or player identities.

This project aims to rigorously evaluate what is achievable under **strict pre-match conditions** — using only information that would genuinely be available before kick-off — and to systematically reproduce published methods to verify their claimed performance.

### 1.3 Objectives

1. **Build a large-scale, multi-league dataset** covering 5 major European leagues across 6+ seasons.
2. **Engineer principled pre-match and in-play feature sets** with clear temporal boundaries.
3. **Benchmark a comprehensive suite of models** from classical ML to deep learning and quantum computing.
4. **Critically reproduce representative published methods** with strict pre-match constraints.
5. **Provide honest assessment** of the fundamental predictability limits in football.

### 1.4 Key Findings Summary

| Setting | Best Model | Best Accuracy | Best Macro-F1 | Draw Recall |
|---------|-----------|---------------|---------------|-------------|
| Pre-match (strict) | CatBoost / Advanced Ensemble | ~50.2% | ~45.7% | ~23% |
| Pre-match (deep) | MLP (PyTorch) | ~49.4% | ~45.7% | ~23% |
| Pre-match (LSTM) | Bi-LSTM + Attention | ~49.1% | ~45.2% | ~22% |
| In-play (10') | Gradient Boosting | 57.2% | — | — |
| In-play (20') | Gradient Boosting | 58.4% | — | — |
| In-play (30') | Gradient Boosting | 60.4% | — | — |
| In-play (40') | Gradient Boosting | 64.2% | — | — |
| In-play (45', half-time) | Gradient Boosting | 66.6% | — | — |
| In-play (overall) | Gradient Boosting | 70.5% | 69.3% | 65% |

---

## 2. Literature Review and Reproducibility Analysis

We collected and reviewed a body of recent literature on football match outcome prediction, covering a range of methods from Bayesian networks and classical ML to deep learning, edge-computing, and quantum-inspired models. Rather than reviewing each paper individually, we classify the literature into three categories based on methodological soundness.

### 2.1 Category A: Studies Reliant on Post-Match or In-Play Features

The **majority** of papers claiming high pre-match prediction accuracy (70–85%) share a common methodological flaw: the use of features that are only available **during or after** the match. Typical examples include:

- **In-match statistics as inputs**: shots, shots on target, corners, fouls, yellow/red cards — all of which summarise what happened in the match, rather than predicting what will happen.
- **Half-time data**: half-time goals (HTHG/HTAG) and half-time result (HTR) are sometimes treated as "pre-match" features, despite being available only at the 45th minute.
- **Direct label leakage**: some studies include goal difference as an input feature, which is a near-trivial transformation of the target variable itself.

These studies are performing **post-hoc classification** (explaining outcomes from in-game data) rather than genuine **pre-match prediction**. When we strip away the leaked features and re-run these methods with strictly pre-match data, accuracy consistently drops from the claimed 70–85% to approximately **48–51%** — indistinguishable from the results of our own models.

Our reproduction experiments (Scripts `11`, `16`) confirm this pattern:
- A Bayesian Network method claiming 75% accuracy yields only **48.85%** under strict pre-match conditions, and requires full-time goals as input to reach 82.63% — clear evidence of leakage.
- An ensemble of FNN, Random Forest, XGBoost, and SVM claiming 70%+ accuracy drops to a maximum of **50.73%** once half-time features are removed.

### 2.2 Category B: Methodologically Sound Prediction Studies

A smaller subset of the literature employs rigorous methodology with genuinely pre-match features and proper temporal evaluation. The most notable example uses **only historical match scores** to derive 6 features (average goals scored/conceded and normalised rank per side), evaluated via Ranked Probability Score (RPS) on a strict temporal holdout of future matches.

These honest studies report modest results — typically around **RPS ≈ 0.21** and **~50% accuracy** — which aligns precisely with what we observe across all our own experiments. Our reproduction (Script `17`) of the best-in-class method achieves RPS = 0.2036 and accuracy = 51.74%, closely matching the original claims.

This category also includes Bayesian state-space models that treat team strengths as latent variables evolving over time. While methodologically interesting, these adopt a fundamentally different paradigm (generative Bayesian modelling) and are not directly comparable in a discriminative ML benchmark.

### 2.3 Category C: Surveys, Novel Architectures, and Non-Reproducible Work

The remaining literature consists of:

- **Survey papers**: systematic reviews that catalogue existing methods without proposing a reproducible prediction model.
- **Novel architectural proposals**: papers centred on edge-computing fusion, complex network theory, quantum neural networks, or player-based generalisable frameworks. While conceptually interesting, these often lack the data or evaluation rigour needed for fair comparison. Where feasible, we adapted their core ideas and tested them in our framework (Scripts `13`–`15`), finding that the novel architectures **do not outperform** standard gradient-boosted trees or logistic regression on our dataset.

### 2.4 Summary of Literature Findings

The literature falls into a clear dichotomy:

| Category | Typical Claimed Accuracy | Actual Pre-match Accuracy | Root Cause |
|----------|-------------------------|--------------------------|------------|
| A — Post-match features | 70–85% | 48–51% | In-match / label leakage |
| B — Honest methods | ~50% (RPS ≈ 0.21) | ~50% (RPS ≈ 0.20) | Genuine pre-match prediction |
| C — Surveys / novel architectures | Varies | ≤ 49% | Limited data / evaluation issues |

This pattern — inflated results from leakage, honest results around 50% — is the central finding of our reproducibility analysis and motivates the comprehensive benchmarking in the following sections.

---

## 3. Data Collection and Processing

### 3.1 Data Source

We use the **API-Football v3** (api-sports.io) as our primary data source, with a **Pro subscription** providing 7,500 API calls/day.

**Five target leagues:**

| League | API ID | Country | Typical Season Size |
|--------|--------|---------|-------------------|
| English Premier League | 39 | England | 380 matches |
| LaLiga | 140 | Spain | 380 matches |
| Serie A | 135 | Italy | 380 matches |
| Bundesliga | 78 | Germany | 306 matches |
| Ligue 1 | 61 | France | 380 matches |

**Seasons covered:** 2019/20 through 2025/26 (in API-Football, `season=2019` refers to the 2019/20 season)

### 3.2 Data Pipeline (Script `0_fetch_data.py`)

The data acquisition pipeline fetches the following endpoints for each fixture:

1. **`/fixtures`** — Match list for each league/season (date, teams, scores, status)
2. **`/fixtures/events`** — Minute-by-minute events (goals, cards, substitutions, VAR decisions)
3. **`/fixtures/statistics`** — Full-match team statistics (shots, possession, corners, fouls)
4. **`/fixtures/lineups`** — Starting XI, formations, substitutes
5. **`/fixtures/players`** — Individual player ratings and statistics

**Pipeline features:**
- Rate limiting: 0.6 second sleep between API calls
- Retry with exponential backoff: up to 3 retries on failure
- Incremental fetching: `_done.flag` files prevent re-downloading completed fixtures
- Error handling: `_season_error.txt` and `_error.txt` for tracking failures

**Raw data structure:**
```
data/raw/api_football/
├── league=39_EPL/
│   ├── season=2019/
│   │   ├── fixture_123456/
│   │   │   ├── meta.json
│   │   │   ├── events.json
│   │   │   ├── lineups.json
│   │   │   ├── statistics.json
│   │   │   ├── players.json
│   │   │   └── _done.flag
│   │   └── ...
│   ├── season=2020/
│   └── ...
├── league=140_LaLiga/
└── ...
```

### 3.3 Supplementary Data

We also include **StatsBomb open data** for exploratory analysis:
- 100+ match event files in `data/raw/statsbomb-open-data/`
- Processed into `data/processed/statsbomb/statsbomb_dataset.csv` and `statsbomb_dataset_fine.csv`

### 3.4 Dataset Statistics

| Dataset | Samples | Features | Date Range | Teams |
|---------|---------|----------|------------|-------|
| Pre-match (`pretrain_dataset.csv`) | 10,627 | 76+ (numeric) + 4 (categorical) | 2019-08-09 to 2025-05-29 | 143 |
| In-play (`inplay_dataset.csv`) | 106,270 | 89 | Same | Same |

**Class distribution (pre-match):**

| Class | Count | Proportion |
|-------|-------|-----------|
| Home Win (H) | ~4,460 | ~42.0% |
| Away Win (A) | ~3,460 | ~32.6% |
| Draw (D) | ~2,707 | ~25.5% |

---

## 4. Feature Engineering

Feature engineering is performed in Script `1_extract_features.py`. All features are computed using only information available at prediction time — strict temporal boundaries are enforced.

### 4.1 Pre-Match Feature Categories (76 features)

#### 4.1.1 Team Form Features (per side: home/away)
- `h_win_rate` / `a_win_rate` — Historical win rate
- `h_avg_gf` / `a_avg_gf` — Average goals scored per match
- `h_avg_ga` / `a_avg_ga` — Average goals conceded per match
- `h_points_per_game` / `a_points_per_game` — Points per game (3 for win, 1 for draw, 0 for loss)
- `h_clean_sheet_rate` / `a_clean_sheet_rate` — Percentage of matches without conceding
- `h_failed_to_score_rate` / `a_failed_to_score_rate` — Percentage of matches without scoring
- `h_avg_gf_1h` / `h_avg_gf_2h` — First/second half average goals scored
- `h_avg_yellow` / `a_avg_yellow` — Average yellow cards per match
- `h_avg_red` / `a_avg_red` — Average red cards per match

#### 4.1.2 Head-to-Head (H2H) Features
- `h2h_games` — Number of historical head-to-head encounters
- `h2h_home_win_rate` — Home team's win rate in H2H matchups
- `h2h_draw_rate` — Draw rate in H2H matchups
- `h2h_home_goal_diff_avg` — Average goal difference in H2H (positive = home-favoured)

#### 4.1.3 Schedule & Fatigue Features
- `h_days_since_last_match` / `a_days_since_last_match` — Rest days
- `h_matches_last_7d` / `a_matches_last_7d` — Fixture congestion
- `diff_days_since_last_match` — Rest advantage differential

#### 4.1.4 League Position Features
- `h_rank` / `a_rank` — Current league table position
- `h_games_played` / `a_games_played` — Matches played so far
- `round_no` / `season_progress` — Calendar context
- `h_gap_top` / `h_gap_top4` / `h_gap_safety` — Points gap to league positions
- `a_gap_top` / `a_gap_top4` / `a_gap_safety` — Same for away team

#### 4.1.5 Motivation Index Features
- `importance_sum` — Combined importance index (how much is at stake for both teams)
- `importance_diff` — Differential importance (which team has more to play for)

#### 4.1.6 Key Player Features
- `h_key_players_started` / `a_key_players_started` — Number of key players in starting XI
- `h_key_players_absent` / `a_key_players_absent` — Number of key players missing
- `h_key_players_form_avg_rating` / `a_key_players_form_avg_rating` — Average recent form rating of key players
- `h_key_players_form_avg_contrib` / `a_key_players_form_avg_contrib` — Average recent goal contribution of key players
- `h_starting11_avg_minutes_7d` / `a_starting11_avg_minutes_7d` — Squad fatigue indicators
- `h_starting11_avg_matches_7d` / `a_starting11_avg_matches_7d` — Match load indicators

#### 4.1.7 Differential Features
- `diff_win_rate`, `diff_avg_gf`, `diff_avg_ga`, `diff_points_per_game` — Home minus away differentials
- `diff_key_players_absent` — Key player availability gap

#### 4.1.8 Categorical Features
- `home_team`, `away_team` — Team identifiers
- `home_formation`, `away_formation` — Tactical formations (e.g., "4-3-3", "3-5-2")

### 4.2 In-Play Features (89 features = 76 pre-match + 13 real-time)

In addition to all pre-match features, the in-play dataset adds:

- `minute` / `minute_ratio` — Current match minute and normalised progress
- `goals_home` / `goals_away` / `goal_diff` — Current scoreline
- `first_goal_team` — Which team scored first (0 = none, 1 = home, 2 = away)
- `equalizers` — Number of equalising goals so far
- `impact_score_total` — Cumulative event impact score (goals, cards, subs weighted by importance)
- `impact_score_recent10` — Event impact in the last 10 minutes
- `min_since_last_goal_home` / `min_since_last_goal_away` — Minutes elapsed since last goal
- `goal_diff_recent10` — Goal difference in the last 10 minutes
- `seg_impact_*` — Impact scores per 10-minute segment (e.g., `seg_impact_11_20`, `seg_impact_21_30`)

In-play samples are generated at **10 checkpoint minutes**: [10, 20, 30, 40, 45, 50, 60, 70, 80, 90], creating 10 rows per fixture (10,627 × 10 = 106,270 samples).

---

## 5. Experimental Methodology

### 5.1 Evaluation Protocol

**Temporal holdout split** is used throughout to prevent temporal leakage:
- Fixtures are sorted chronologically by date
- The last **20%** of fixtures form the test set
- For models requiring a validation set, the last 15% of the training set is held out
- **No random shuffling** across time — the model never sees future matches during training

For in-play models, an additional constraint is applied: **fixture-level group holdout** — all checkpoints of a given fixture appear in the same split (train or test), preventing information leaking from one minute to another within the same match.

### 5.2 Metrics

| Metric | Use Case | Note |
|--------|----------|------|
| **Accuracy** | Primary comparison | Overall correct classification rate |
| **Macro-F1** | Class-balanced evaluation | Equally weights H/D/A regardless of frequency |
| **ROC-AUC (weighted, OvR)** | Probability calibration | Measures quality of predicted probabilities |
| **LogLoss** | Probability calibration | Penalises confident wrong predictions heavily |
| **RPS** (Ranked Probability Score) | Berrar reproduction only | Standard metric in soccer prediction challenges; lower = better |

### 5.3 Baseline

A trivial baseline predicts every match as the majority class (Home Win), achieving **~42% accuracy**. Our **null model** using the training-set class distribution achieves **42.07% accuracy** with **RPS = 0.2314**.

---

## 6. Pre-Match Model Results

### 6.1 Comprehensive Model Comparison

The table below summarises all pre-match models tested, ordered by accuracy:

| # | Script | Model | Accuracy | Macro-F1 | ROC-AUC | Draw Recall | Note |
|---|--------|-------|----------|----------|---------|-------------|------|
| 1 | 5 | Advanced Ensemble (Stacking) | **50.21%** | 44.01% | 0.6573 | 13% | HistGB + ExtraTrees + Stacking |
| 2 | 14 | Edge ML Fusion (LR) | 49.41% | 43.68% | — | 14% | 8 edge nodes, probability fusion |
| 3 | 14 | Edge ML Fusion (BP) | 49.41% | — | — | — | MLP fusion |
| 4 | 6 | Deep MLP (PyTorch) | 49.36% | 45.71% | 0.6543 | 23% | 500 epochs, early stop @6 |
| 5 | 13 | Player Ensemble (LR+ET+KNN) | 49.27% | 42.18% | 0.6466 | 10% | 64-dim SVD, voting |
| 6 | 9/LSTM | Bi-LSTM + Attention | 49.08% | 45.19% | 0.6579 | 22% | H2H + form sequences |
| 7 | 12 | Bayesian Network (strict) | 48.85% | 36.29% | 0.6283 | 0% | 40 features, BDeu |
| 8 | 2 | CatBoost (baseline) | 48.73% | 41.60% | — | 11% | GPU accelerated |
| 9 | 7 | XGBoost (gradient boost) | 48.27% | 45.10% | — | — | CUDA accelerated |
| 10 | 16 | Atta Mills FNN (reproduction) | 50.73% | 40.12% | — | 3% | Paper-style FNN |
| 11 | 15 | Logistic Regression (QNN baseline) | 49.60% | 44.83% | — | — | PCA-reduced features |
| 12 | 15 | KNN (QNN baseline) | 45.74% | 39.64% | — | — | |
| 13 | 15 | Hybrid QNN (PennyLane) | 38.31% | 34.33% | — | — | 6 qubits, 3 layers |
| 14 | 8 | ResNet (AutoGluon-style) | 48.25% | 41.51% | — | — | |
| 15 | 14 | Edge ML Fusion (KNN) | 47.29% | 39.25% | — | 10% | |

### 6.2 Key Observations

1. **Accuracy ceiling at ~50%**: Despite using 15+ different model architectures, 76 features, and 10,627 training matches, no pre-match model breaks the 51% accuracy barrier. The best result (50.21%, Advanced Ensemble) is only 8 percentage points above the majority-class baseline (42%).

2. **Draw prediction is nearly impossible**: Draw recall ranges from 0% (Bayesian Network) to 23% (Deep MLP / LSTM). Models consistently sacrifice Draw recall to maximise Home Win and Away Win accuracy. This is rational — predicting everything as H or A yields better overall accuracy than attempting to predict the least frequent, most random outcome.

3. **Feature importance is diffuse**: No single feature dominates. The Top 10 features by CatBoost importance score (on a max-100 scale) are all below 4.0, indicating the information is spread across many weak signals:
   - Key player form ratings (~3.75)
   - Days since last match difference (~3.70)
   - Importance differential (~3.47)
   - League rank (~2.79)

4. **Complex models ≠ better results**: The Hybrid QNN (38.31%) performs significantly worse than simple Logistic Regression (49.60%). LSTM with attention (49.08%) does not outperform a basic CatBoost (48.73%) despite being far more complex. This suggests the problem is fundamentally data-limited, not model-limited.

5. **ROC-AUC consistently ~0.65**: The probability calibration metric hovers around 0.65 across all models, further confirming a hard ceiling on pre-match predictive information.

### 6.3 Top Feature Importance (Advanced Ensemble, Script 5)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `a_gap_safety` | 0.0261 |
| 2 | `importance_diff` | 0.0260 |
| 3 | `h_gap_top4` | 0.0258 |
| 4 | `h2h_home_goal_diff_avg` | 0.0258 |
| 5 | `h_key_players_form_avg_rating` | 0.0257 |
| 6 | `h_key_players_form_avg_contrib` | 0.0257 |
| 7 | `a_key_players_form_avg_contrib` | 0.0255 |
| 8 | `a_rank` | 0.0254 |
| 9 | `h_gap_safety` | 0.0254 |
| 10 | `a_gap_top4` | 0.0252 |

---

## 7. In-Play Model Results

### 7.1 Model Comparison (at 30' checkpoint)

We compare in-play models at the **30-minute mark** — a point where the match is still in the first half with genuine outcome uncertainty, yet enough events have occurred to differentiate model quality.

| # | Script | Model | Accuracy @30' | Accuracy @45' |
|---|--------|-------|--------------|--------------|
| 1 | 3 | Gradient Boosting | **60.36%** | **66.60%** |
| 2 | 8 | ResNet | 59.23% | 59.23% |
| 3 | 7 | XGBoost | 55.81% | 57.81% |
| 4 | 6 | Deep MLP (PyTorch) | 54.92% | 58.68% |
| 5 | 9 | LSTM + Attention | 54.73% | 57.74% |
| 6 | 5 | Advanced Ensemble | 53.93% | 58.21% |

### 7.2 First-Half Prediction — The Most Valuable Window

From a practical standpoint, **first-half predictions (10'–45')** are the most meaningful in-play setting: the match outcome is still genuinely uncertain, yet real-time events have begun to provide actionable signals. By contrast, predictions at 80'–90' are trivially accurate (the score is nearly final), and pre-match predictions lack any in-game information. The first half represents the "sweet spot" where the model must reason under genuine uncertainty with partial information.

#### 7.2.1 First-Half Accuracy by Minute (All Models)

| Minute | Gradient Boosting | XGBoost | ResNet | Deep MLP | LSTM | Advanced Ensemble |
|--------|------------------|---------|--------|----------|------|------------------|
| **10'** | **57.20%** | 50.42% | 48.75% | 48.85% | 49.60% | 44.89% |
| **20'** | **58.42%** | 55.20% | 53.91% | 51.48% | 51.44% | 49.32% |
| **30'** | **60.36%** | 55.81% | 59.23% | 54.92% | 54.73% | 53.93% |
| **40'** | **64.18%** | — | — | 56.09% | 55.34% | 55.53% |
| **45'** | **66.60%** | 57.81% | 59.23% | 58.68% | 57.74% | 58.21% |

**Key takeaways from the first half:**

- **10' → 45' gain**: The best model (Gradient Boosting) improves from 57.20% to 66.60% during the first half alone — a **+9.4 percentage point** gain, demonstrating that early match events (first goal, early cards, tactical shape) carry substantial predictive value.
- **10' vs pre-match**: Even at 10 minutes, Gradient Boosting (57.20%) already exceeds the best pre-match model (50.21%) by **+7 percentage points**, showing that just 10 minutes of live data add more signal than 76 engineered pre-match features.
- **45' (half-time)**: At half-time, prediction accuracy reaches 57–67% depending on the model. This is practically useful — the match is only halfway done, yet the model already achieves a meaningful advantage over random guessing (33%) and pre-match prediction (~50%).
- **Model ranking is stable**: Gradient Boosting consistently leads at every first-half checkpoint. The gap between models narrows at later minutes as the signal (goal difference) becomes increasingly dominant.

#### 7.2.2 Full Match Accuracy Progression

For reference, the complete minute-by-minute progression (including second half) is shown below:

| Minute | Gradient Boosting | XGBoost | ResNet | Deep MLP | LSTM | Advanced Ensemble |
|--------|------------------|---------|--------|----------|------|------------------|
| 10' | 57.20% | 50.42% | 48.75% | 48.85% | 49.60% | 44.89% |
| 20' | 58.42% | 55.20% | 53.91% | 51.48% | 51.44% | 49.32% |
| 30' | 60.36% | 55.81% | 59.23% | 54.92% | 54.73% | 53.93% |
| 40' | 64.18% | — | — | 56.09% | 55.34% | 55.53% |
| 45' | 66.60% | 57.81% | 59.23% | 58.68% | 57.74% | 58.21% |
| 50' | 68.75% | — | — | 61.13% | 60.52% | 60.85% |
| 60' | 71.51% | 66.67% | 66.39% | 66.02% | 66.12% | 66.92% |
| 70' | 74.57% | — | — | 73.69% | 72.52% | 74.31% |
| 75' | — | 76.60% | 75.54% | — | — | — |
| 80' | 84.14% | — | — | 80.66% | 79.48% | 81.65% |
| 90' | **99.30%** | **98.31%** | **95.67%** | **95.53%** | **95.01%** | **97.98%** |

### 7.3 Key Observations

1. **Goal difference is king**: `goal_diff` accounts for **61.36%** of feature importance in the best in-play model (Gradient Boosting), dwarfing all other features. This makes intuitive sense — the current score is by far the strongest indicator of the final result.

2. **First-half predictions are the practical sweet spot**: At half-time (45'), the best model achieves **66.60%** — a **+16.4 percentage point** improvement over the best pre-match model (50.21%). This gain comes from observing only ~45 minutes of events (goals, cards, substitutions), yet it exceeds the total benefit of all 76 pre-match statistical features.

3. **Diminishing information gain in the second half**: The accuracy jump from 10' to 45' (+9.4pp) is comparable to the jump from 45' to 80' (+17.5pp), despite the second half spanning 35 more minutes. The first half provides disproportionate information relative to its duration.

4. **In-play feature importance** (Gradient Boosting, Top 10):

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `goal_diff` | 0.6136 |
| 2 | `minute` | 0.0528 |
| 3 | `impact_score_total` | 0.0453 |
| 4 | `minute_ratio` | 0.0422 |
| 5 | `h_rank` | 0.0261 |
| 6 | `a_rank` | 0.0247 |
| 7 | `importance_diff` | 0.0197 |
| 8 | `h2h_home_goal_diff_avg` | 0.0194 |
| 9 | `importance_sum` | 0.0175 |
| 10 | `a_days_since_last_match` | 0.0110 |

5. **Draw prediction improves dramatically**: In-play Draw recall reaches **65%** (vs ~23% pre-match) because the model can observe that no goals have been scored and both teams are evenly matched.

---

## 8. Paper Reproduction Experiments

### 8.1 Overview

We selected representative methods from each literature category (see Section 2) and reproduced them under our strict pre-match evaluation protocol. The results are organized by the type of issue encountered.

### 8.2 Reproducing Category A Methods (Leaked Features)

For methods that rely on in-match or half-time features, we ran two variants: (1) a faithful reproduction using the paper's original feature set (where data was available), and (2) a strict pre-match version that removes all features unavailable before kick-off.

| Reproduction | Original Features | Strict Pre-match | Drop |
|-------------|------------------|-----------------|------|
| Bayesian Network (Script `11`) | 53.95% (paper claims 75%) | 48.85% | -5.1pp |
| Bayesian Network + FT goals (leakage test) | **82.63%** | — | — |
| FNN ensemble (Script `16`, best model) | — | 50.73% | vs claimed 70%+ |
| 7-model comparison (Script `16`) | — | 44–51% range | vs claimed 70%+ |

Key findings:
- The Bayesian Network paper's 75% accuracy could not be replicated even with their own feature set (53.95%), likely due to implementation differences. However, adding full-time goals as input immediately pushes accuracy to 82.63%, confirming that high accuracy is driven by label leakage rather than genuine predictive modelling.
- The ML/DL ensemble paper's 7 models (FNN, RF, XGBoost, SVM, Voting, LR, NB) all fall to the **44–51% range** once half-time features are removed — no different from our own pre-match models.

### 8.3 Reproducing Category B Methods (Honest Prediction)

The most methodologically rigorous approach uses **only historical match scores** to derive 6 features. We reproduce this in Script `17`:

| Model | RPS | Accuracy | F1-macro |
|-------|-----|----------|----------|
| ANN (64-32) | **0.2036** | 51.74% | 39.72% |
| ANN (128-64) | 0.2053 | 51.55% | 39.47% |
| k-NN (k=50) | 0.2070 | 50.16% | 41.23% |
| k-NN (k=30) | 0.2089 | 50.09% | 42.25% |
| Naïve Bayes | 0.2223 | 50.66% | 41.97% |
| Null baseline | 0.2314 | 42.07% | 19.74% |

- **Optimal recency window**: Pearson correlation analysis found *n* = 30 matches (r = 0.427)
- Our best RPS (0.2036) slightly improves upon the paper's k-NN (RPS ≈ 0.211), likely because our dataset (10,627 matches, 5 leagues) is larger
- Accuracy remains firmly around 50%, consistent with our full model suite
- **Critical insight**: Even with 76 features (vs 6 score-only features), our best pre-match models barely improve upon this minimal baseline — suggesting that pre-match features beyond historical form have sharply diminishing returns

### 8.4 Reproducing Category C Methods (Novel Architectures)

| Method | Script | Our Result | Paper's Claim | Note |
|--------|--------|-----------|--------------|------|
| Edge-computing fusion (LR) | `14` | 49.41% | 87.5% | 8 edge nodes, probability fusion |
| Hybrid Quantum NN | `15` | 38.31% | ~60% | 6 qubits, 3 layers; underfits |
| Player-enhanced ensemble | `13` | 49.27% | — | Adapted from player-centric framework |

None of the novel architectures outperform standard methods on our dataset. The Quantum Neural Network performs significantly **worse** than a simple Logistic Regression baseline (38.31% vs 49.60%), suggesting that current quantum circuit models are not yet competitive for this class of tabular classification problems.

---

## 9. Misclassification Analysis

Script `10_analyze_misclassifications.py` provides detailed error analysis for both pre-match and in-play models.

### 9.1 Pre-Match Error Patterns

**Overall test accuracy**: 49.10%

| True / Predicted | Away Win | Draw | Home Win |
|-----------------|----------|------|----------|
| **Away Win** | **254** | 62 | 190 |
| **Draw** | 134 | **47** | 224 |
| **Home Win** | 147 | 59 | **486** |

**Per-class accuracy:**
- Home Win: 70.23% (relatively easy — strong home bias)
- Away Win: 50.20% (moderate difficulty)
- Draw: **11.60%** (extremely poor — nearly random)

**Top confusion pairs and feature analysis:**

1. **Draw → Home Win (224 errors)**: The model predicts Home Win when the match ends in a Draw. Characteristic pattern: the home team has a lower (better) league rank (9.09 vs 13.49) and a higher safety gap (11.26 vs 3.04). Essentially, when the home team "should" win but doesn't, the model gets it wrong.

2. **Away Win → Home Win (190 errors)**: The model picks the wrong winner. The home team's H2H record is better (0.287 vs -0.994 avg goal diff) and point differential is higher. These are matches where historical H2H advantage fails.

3. **Home Win → Away Win (147 errors)**: Reverse upsets. The away team has better points-per-game and win rate differential.

**Highest-confidence errors** (pre-match model was highly confident but wrong):
- 2020-09-19 Eintracht Frankfurt vs Arminia Bielefeld: Predicted H (97.76% confidence), actual D
- 2023-02-19 Union Berlin vs FC Schalke 04: Predicted H (97.34% confidence), actual D
- 2019-12-14 Leicester vs Norwich: Predicted H (97.27% confidence), actual D

All top-confidence errors are **strong favourites drawing** — a fundamental unpredictability in football.

### 9.2 In-Play Error Patterns

**Overall test accuracy**: 72.46%

| True / Predicted | Away Win | Draw | Home Win |
|-----------------|----------|------|----------|
| **Away Win** | **2260** | 548 | 228 |
| **Draw** | 389 | **1610** | 430 |
| **Home Win** | 312 | 741 | **3097** |

The most common in-play error is **Home Win → Draw (741 cases)** — the home team is winning at a checkpoint but the match eventually ends in a draw (late equaliser). This is followed by **Away Win → Draw (548 cases)** — same pattern for away team leads.

**Average model confidence:**
- Correct predictions: 0.7538
- Wrong predictions: 0.5341

The clear confidence gap (75% vs 53%) indicates the model is well-calibrated — it is less certain about its mistakes.

---

## 10. Discussion

### 10.1 The ~50% Pre-Match Ceiling

Our most important finding is the **robust accuracy ceiling at approximately 50%** for strict pre-match prediction. This result is consistent across:
- 15+ model architectures (from Logistic Regression to LSTM to Quantum Neural Networks)
- Multiple feature sets (6 features to 76 features)
- Different evaluation horizons (single-season to 6-season)
- Multiple papers' methodologies (Bayesian, ensemble, deep learning)

A 50% accuracy on a 3-class problem with base rates of [42%, 26%, 32%] translates to only marginal lift over the majority-class baseline (42%). This suggests that **football match outcomes are fundamentally difficult to predict before kick-off**, at least from the publicly available statistical information.

### 10.2 Why Published Papers Report Higher Numbers

Our reproducibility study reveals that papers claiming 70%+ pre-match accuracy almost universally rely on one or more of:

1. **In-match features disguised as pre-match**: Half-time goals (Atta Mills 2024), shots/corners/fouls (Razali 2017), or other in-play statistics are used as input features. These are not pre-match features.
2. **Label leakage**: Goal difference (Haruna 2022) is directly derived from the target variable.
3. **Random cross-validation**: Splitting data randomly rather than temporally allows the model to train on matches that occurred after its test matches, learning future trends.
4. **Publication bias**: Papers reporting ~50% accuracy (the honest result) are less likely to be published or cited than those claiming 80%+ accuracy.

### 10.3 The Value of In-Play Prediction

The in-play models demonstrate that **real-time match information is extremely predictive**:
- `goal_diff` alone explains 61% of prediction accuracy
- By minute 60, accuracy reaches 71%, and by minute 80 it reaches 84%
- At minute 90, accuracy is ~99% (trivially, knowing the current score near full-time is nearly sufficient)

This has practical implications for **live betting markets**, **broadcast analytics**, and **tactical decision-support systems** — applications where real-time prediction has genuine value, unlike pre-match prediction where the honest ceiling is ~50%.

### 10.4 Draw Prediction — The Unsolved Problem

Across all experiments, Draw remains the hardest outcome to predict:

| Model Type | Draw Recall | Draw F1 |
|-----------|-------------|---------|
| Pre-match best | 23% | 26% |
| Pre-match worst | 0% | 0% |
| In-play best | 65% | 58% |

Draws occur when two teams are evenly matched AND no team manages to score an extra goal — a conjunction of skill balance and luck. The low Draw recall is not a model failure but a reflection of the inherent unpredictability of this outcome.

### 10.5 Feature Engineering Insights

1. **Diminishing returns from more features**: Berrar's 6-feature score-only model (RPS=0.2036) is not significantly worse than our 76-feature models. Complex player-level and league-position features provide only marginal improvement.

2. **Player form rating is consistently important**: The `h_key_players_form_avg_rating` and `a_key_players_form_avg_rating` features consistently appear in the top-10 across multiple models, suggesting that individual player quality (as measured by historical match ratings) is one of the stronger pre-match signals.

3. **Motivation matters more than expected**: `importance_diff` (how much more one team needs the win) ranks consistently in the top-5 features, surpassing many traditional statistics like win rate or goals scored.

4. **Contextual features add value**: League position gaps (`h_gap_top4`, `a_gap_safety`) and scheduling factors (`diff_days_since_last_match`) contribute meaningfully, suggesting that the "external circumstances" of a match matter beyond just team quality.

### 10.6 Model Complexity vs Performance

| Model Category | Best Pre-Match Acc | Complexity |
|---------------|-------------------|------------|
| Logistic Regression | 49.60% | Very Low |
| Gradient Boosted Trees | 50.21% | Medium |
| Deep MLP | 49.36% | High |
| LSTM + Attention | 49.08% | Very High |
| Quantum Neural Network | 38.31% | Extreme |

There is no meaningful accuracy gain from increasing model complexity. The simplest models (Logistic Regression) perform within 1 percentage point of the most complex (LSTM + Attention). The Quantum Neural Network actually performs significantly worse, likely due to the difficulty of training quantum circuits on noisy classical data with limited qubits (6 qubits).

This strongly suggests the bottleneck is **information, not model capacity** — the pre-match features simply do not contain enough predictive signal to discriminate outcomes beyond ~50%.

---

## 11. Conclusion

### 11.1 Main Contributions

1. **Large-scale multi-league benchmarking**: We constructed a dataset of 10,627 fixtures across 5 major European leagues and 6+ seasons, with 76 principled pre-match features and 89 in-play features.

2. **Comprehensive model comparison**: We tested 15+ model architectures including classical ML, gradient-boosted trees, deep learning, recurrent models, Bayesian networks, edge-computing fusion, and quantum neural networks.

3. **Critical reproducibility study**: We systematically reproduced 6 published papers and demonstrated that most claimed high accuracies are artefacts of feature leakage, not genuine prediction capability.

4. **Honest performance assessment**: Strict pre-match prediction accuracy is robustly ~50% (±1%), regardless of model complexity. This is only marginally above the majority-class baseline of 42%.

5. **In-play prediction value**: Real-time match events dramatically improve prediction accuracy, reaching 70%+ overall and 99%+ near full-time.

### 11.2 Practical Implications

- **For researchers**: Pre-match football prediction papers should be evaluated with extreme scepticism. Any reported accuracy above 55% using only pre-match features warrants careful feature auditing for temporal leakage.
- **For practitioners**: In-play prediction models have genuine practical value for live applications. Pre-match prediction should be treated as a probabilistic tool (producing calibrated probabilities) rather than a deterministic classifier.
- **For the field**: The honest ~50% accuracy ceiling suggests that future progress in pre-match prediction may require fundamentally new information sources (e.g., tactical video analysis, detailed training data, psychological state monitoring) rather than more sophisticated models.

### 11.3 Limitations

1. **Five European leagues only**: Our dataset covers major leagues but does not include South American, Asian, or lower-tier competitions, which may have different predictability characteristics.
2. **No betting odds features**: Many top prediction models in the literature use bookmaker odds as features, which encode expert/market knowledge. We excluded these to focus on "pure" statistical prediction.
3. **No fine-grained in-play features**: Our in-play model uses checkpoint-based event statistics (goals, cards at minute T) but cannot access real-time tracking data (xG, possession trajectories, pressing intensity) which could further improve predictions.
4. **Temporal scope**: 2019–2025 covers a specific era of football; tactical evolution over longer periods was not studied.

### 11.4 Future Work

1. **Incorporating bookmaker odds** as additional features or as a benchmark baseline.
2. **Expected Goals (xG) integration** using StatsBomb-style detailed event data for more nuanced pre-match features.
3. **Transfer learning** across leagues to improve prediction for leagues with less data.
4. **Transformer architectures** for sequence modelling of team form trajectories.
5. **Explainable AI (XAI)** methods to better understand what drives model decisions.

---

## 12. Project Structure

```
CP3106/
├── 0_fetch_data.py              # Data acquisition from API-Football
├── 1_extract_features.py        # Feature engineering (pre-match + in-play)
├── 2_train_pretrain_model.py    # Baseline pre-match models (CatBoost/XGBoost/SVM)
├── 3_train_inplay_model.py      # Baseline in-play model (Gradient Boosting)
├── 4_prediction_demo.py         # Inference demo
├── 5_train_advanced_models.py   # Advanced ensemble (HistGB/ExtraTrees/Stacking)
├── 6_train_deep_models.py       # PyTorch MLP (pre-match + in-play)
├── 7_try_gradient_boost_models.py  # XGBoost/CatBoost comparison
├── 8_try_autogluon_deep_learning.py # ResNet-style deep learning
├── 9_train_lstm_inplay.py       # Bi-LSTM + Attention in-play model
├── 10_analyze_misclassifications.py # Error analysis
├── 11_reproduce_bayesian_epl_paper.py # Reproduce Razali 2017
├── 12_train_bayesian_prematch_strict.py # Strict Bayesian pre-match
├── 13_train_player_enhanced_ensemble.py # Player-level feature ensemble
├── 14_try_edge_ml_method.py     # Edge computing + ML fusion
├── 15_qnn_hybrid_comparison.py  # Quantum Neural Network comparison
├── 16_reproduce_attamills_prematch.py # Reproduce Atta Mills 2024
├── 17_reproduce_berrar_deepfoot.py  # Reproduce Berrar DeepFoot 2024
├── data/
│   ├── raw/
│   │   ├── api_football/        # Raw API data (~10,600 fixture directories)
│   │   └── statsbomb-open-data/ # StatsBomb open data
│   └── processed/
│       ├── pretrain_dataset.csv # 10,627 rows × 80+ columns
│       ├── inplay_dataset.csv   # 106,270 rows × 89+ columns
│       └── statsbomb/           # StatsBomb processed data
├── models/                      # Saved model files (.pkl, .pt)
├── papers/                      # 12 reference papers (PDF)
├── reports/                     # All experiment reports
│   ├── pretrain_report.txt
│   ├── pretrain_report_advanced.txt
│   ├── pretrain_report_deep.txt
│   ├── pretrain_report_bayesian_strict.txt
│   ├── pretrain_report_player_ensemble.txt
│   ├── pretrain_report_lstm.txt
│   ├── inplay_report.txt
│   ├── inplay_report_advanced.txt
│   ├── inplay_report_deep.txt
│   ├── inplay_report_lstm.txt
│   ├── gradient_boost_comparison.txt
│   ├── advanced_deep_learning_comparison.txt
│   ├── edge_ml_method_report.txt
│   ├── misclassification_analysis.txt
│   ├── 15_qnn_hybrid_comparison.txt
│   ├── 16_attamills_prematch_report.txt
│   ├── 17_berrar_deepfoot_report.txt
│   └── paper_bayesian_epl_paper_like*.txt
├── dataset_structure.md         # Data structure documentation
└── readme.md                    # This report
```

---

## 13. How to Run

### 13.1 Environment Setup

```bash
# Create conda environment
conda create -n cvlab python=3.12
conda activate cvlab

# Install core dependencies
pip install pandas numpy scikit-learn xgboost catboost torch torchvision
pip install requests tqdm pgmpy pennylane pennylane-lightning

# Optional: GPU acceleration
pip install cupy-cuda12x  # For CuPy
```

### 13.2 Data Pipeline

```bash
# Step 0: Fetch raw data from API-Football (requires API key)
python 0_fetch_data.py

# Step 1: Extract features
python 1_extract_features.py
```

### 13.3 Training Pipeline

```bash
# Baseline models
python 2_train_pretrain_model.py    # Pre-match baseline
python 3_train_inplay_model.py      # In-play baseline

# Advanced models
python 5_train_advanced_models.py   # Ensemble methods
python 6_train_deep_models.py       # PyTorch MLP
python 7_try_gradient_boost_models.py  # XGBoost/CatBoost
python 9_train_lstm_inplay.py       # LSTM

# Paper reproductions
python 11_reproduce_bayesian_epl_paper.py
python 16_reproduce_attamills_prematch.py
python 17_reproduce_berrar_deepfoot.py

# Analysis
python 10_analyze_misclassifications.py
python 4_prediction_demo.py
```

---

## 14. References

1. Razali, N. et al. (2017). "Predicting Football Matches Results using Bayesian Networks for English Premier League." *International Journal on Advanced Science, Engineering and Information Technology*.

2. Berrar, D., Lopes, P., & Dubitzky, W. (2024). "A data and knowledge-driven framework for developing machine learning models to predict soccer match outcomes." *Machine Learning*, 113:8165–8204.

3. Atta Mills, E.F.E. et al. (2024). "Data-driven prediction of soccer outcomes using enhanced machine and deep learning techniques." *Journal of Big Data*, 11:108.

4. Haruna, U. (2022). "Predicting the Outcomes of Football Matches Using Machine Learning Approach."

5. Moya, J. et al. (2025). "Artificial intelligence and football: A systematic review." *Mathematics*, 7(1):85.

6. Hubáček, O. et al. (2024). "Bayesian state-space modelling." *Journal of Quantitative Analysis in Sports*, qlae075.

7. Cho, K. et al. (2020). "A deep learning framework for football match prediction."

8. Edge Computing + ML Method (2025). "The outcome prediction method of football match."

9. Complex Networks for Soccer Prediction (2024). "Predicting soccer matches with complex networks and machine learning."

10. Player-based Framework (2025). "From Players to Champions: A Generalizable Framework."

---

*Report generated for CP3106 Independent Study. Last updated: April 2026.*
