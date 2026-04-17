# CP3106 Independent Study: Football Match Outcome Prediction — A Reproducibility Study and Multi-Model Comparison

## Abstract

This project conducts a systematic reproducibility study and multi-model comparison for football (soccer) match outcome prediction. We collect match data from five major European leagues (English Premier League, LaLiga, Serie A, Bundesliga, Ligue 1) spanning the 2019–2026 seasons via the API-Football (v3) service, totalling **12,095 fixtures** and **120,950 in-play checkpoint samples**. We engineer a rich set of **76+ pre-match features** and **89 in-play features**, then benchmark a comprehensive set of models: classical machine learning (Logistic Regression, SVM, KNN, Naïve Bayes, Random Forest, Extra Trees), gradient-boosted trees (XGBoost, CatBoost, HistGradientBoosting), deep learning (MLP, ResNet-style, LSTM with attention), a hybrid Quantum Neural Network (QNN), Bayesian Networks, and edge-computing fusion architectures. We also critically reproduce findings from multiple published papers, revealing that many reported high accuracies (70%+) rely on post-match features or data leakage, while strict pre-match prediction consistently plateaus at **~46–50% accuracy** across all model families. Only in-play models, which incorporate real-time match events, achieve meaningfully higher performance, reaching **69.0% overall accuracy** and **99.4% accuracy at minute 90**. These findings highlight a significant gap between claims in the literature and practically achievable pre-match prediction performance.

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

- **Class imbalance**: Home wins typically account for ~43% of outcomes, draws ~25%, and away wins ~32%, making draw prediction particularly challenging.
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

1. **Build a large-scale, multi-league dataset** covering 5 major European leagues across 7 seasons.
2. **Engineer principled pre-match and in-play feature sets** with clear temporal boundaries.
3. **Benchmark a comprehensive suite of models** from classical ML to deep learning and quantum computing.
4. **Critically reproduce representative published methods** with strict pre-match constraints.
5. **Provide honest assessment** of the fundamental predictability limits in football.

### 1.4 Key Findings Summary

| Setting | Best Model | Best Accuracy | Best Macro-F1 | Draw Recall |
|---------|-----------|---------------|---------------|-------------|
| Pre-match (strict) | Advanced Ensemble | ~49.8% | ~44.9% | ~18% |
| Pre-match (deep) | MLP (PyTorch) | ~47.3% | ~44.1% | ~26% |
| Pre-match (LSTM) | Bi-LSTM + Attention | ~45.7% | ~44.2% | ~43% |
| In-play (10') | Gradient Boosting | 54.0% | — | — |
| In-play (20') | Gradient Boosting | 56.6% | — | — |
| In-play (30') | Gradient Boosting | 59.4% | — | — |
| In-play (40') | Gradient Boosting | 61.5% | — | — |
| In-play (45', half-time) | Gradient Boosting | 64.9% | — | — |
| In-play (overall) | Gradient Boosting | 69.0% | 68.0% | 64% |

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
- A Bayesian Network method claiming 75% accuracy yields only **49.36%** under strict pre-match conditions, and requires full-time goals as input to reach 82.63% — clear evidence of leakage.
- An ensemble of FNN, Random Forest, XGBoost, and SVM claiming 70%+ accuracy drops to a maximum of **51.55%** once half-time features are removed.

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

**Seasons covered:** 2019/20 through 2025/26 (in API-Football, `season=2019` refers to the 2019/20 season, data through April 2026)

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
| Pre-match (`pretrain_dataset.csv`) | 12,095 | 76+ (numeric) + 4 (categorical) | 2019-08-09 to 2026-04-13 | 147 |
| In-play (`inplay_dataset.csv`) | 120,950 | 89 | Same | Same |

**Class distribution (pre-match):**

| Class | Count | Proportion |
|-------|-------|------------|
| Home Win (H) | 5,209 | 43.1% |
| Away Win (A) | 3,835 | 31.7% |
| Draw (D) | 3,051 | 25.2% |

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

In-play samples are generated at **10 checkpoint minutes**: [10, 20, 30, 40, 45, 50, 60, 70, 80, 90], creating 10 rows per fixture (12,095 × 10 = 120,950 samples).

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

A trivial baseline predicts every match as the majority class (Home Win), achieving **~43% accuracy**. Our **null model** using the training-set class distribution achieves **43.99% accuracy** with **RPS = 0.2317**.

---

## 6. Pre-Match Model Results

### 6.1 Comprehensive Model Comparison

The table below summarises all pre-match models tested, ordered by accuracy:

| # | Script | Model | Accuracy | Macro-F1 | ROC-AUC | Draw Recall | Note |
|---|--------|-------|----------|----------|---------|-------------|------|
| 1 | 16 | Atta Mills FNN (reproduction) | **51.55%** | 38.80% | — | 0.5% | Paper-style FNN |
| 2 | 14 | Edge ML Fusion (BP) | 50.56% | 37.11% | — | 0% | MLP fusion |
| 3 | 13 | Player Ensemble (LR+ET+RF) | 50.19% | 43.29% | 0.6562 | 12% | 64-dim SVD, voting |
| 4 | 5 | Advanced Ensemble (Stacking) | 49.81% | 44.91% | 0.6579 | 18% | HistGB + ExtraTrees + Stacking |
| 5 | 2 | XGBoost (baseline) | 49.40% | 41.00% | — | 10% | CUDA accelerated |
| 6 | 12 | Bayesian Network (strict) | 49.36% | 36.15% | 0.6253 | 0% | 40 features, BDeu |
| 7 | 16 | RandomForest (reproduction) | 49.94% | 43.58% | — | 13% | Paper-style RF |
| 8 | 7 | CatBoost (gradient boost) | 48.82% | 46.41% | — | — | GPU accelerated |
| 9 | 2 | CatBoost (baseline) | 48.78% | — | — | — | GPU accelerated |
| 10 | 14 | Edge ML Fusion (LR) | 48.37% | 46.39% | — | 15% | 8 edge nodes, probability fusion |
| 11 | 8 | ResNet (AutoGluon-style) | 48.25% | 41.51% | — | — | |
| 12 | 15 | Logistic Regression (QNN baseline) | 48.45% | 44.50% | — | — | PCA-reduced features |
| 13 | 7 | XGBoost (gradient boost) | 47.66% | 45.28% | — | — | CUDA accelerated |
| 14 | 6 | Deep MLP (PyTorch) | 47.29% | 44.06% | 0.6438 | 26% | 500 epochs, early stop @23 |
| 15 | 15 | KNN (QNN baseline) | 46.18% | 40.22% | — | — | |
| 16 | 9/LSTM | Bi-LSTM + Attention | 45.68% | 44.19% | 0.6556 | 43% | H2H + form sequences |
| 17 | 15 | BP-MLP (QNN baseline) | 39.77% | 37.19% | — | — | |
| 18 | 15 | Hybrid QNN (PennyLane) | 33.57% | 33.19% | — | — | 4 qubits, 1 layer |

### 6.2 Key Observations

1. **Accuracy ceiling at ~50%**: Despite using 18 different model architectures, 76 features, and 12,095 training matches, no pre-match model breaks the 52% accuracy barrier. The best result (51.55%, Atta Mills FNN reproduction) is only 8.5 percentage points above the majority-class baseline (43.1%).

2. **Draw prediction is nearly impossible**: Draw recall ranges from 0% (Bayesian Network, Edge BP) to 43% (LSTM). Models that boost Draw recall (e.g., LSTM at 43%) pay a steep accuracy penalty (45.68%). This is a fundamental trade-off — predicting everything as H or A yields better overall accuracy than attempting to predict the least frequent, most random outcome.

3. **Feature importance is diffuse**: No single feature dominates. The Top 10 features by importance (Advanced Ensemble) are all below 2.7%, indicating the information is spread across many weak signals:
   - H2H home goal diff average (~2.68%)
   - League position gaps (~2.68%)
   - Importance differential (~2.64%)
   - Key player form ratings (~2.61%)

4. **Complex models ≠ better results**: The Hybrid QNN (33.57%) performs significantly worse than simple Logistic Regression (48.45%). LSTM with attention (45.68%) does not outperform a basic Advanced Ensemble (49.81%) despite being far more complex. This suggests the problem is fundamentally data-limited, not model-limited.

5. **ROC-AUC consistently ~0.65**: The probability calibration metric hovers around 0.63–0.66 across all models, further confirming a hard ceiling on pre-match predictive information.

### 6.3 Top Feature Importance (Advanced Ensemble, Script 5)

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `h2h_home_goal_diff_avg` | 0.0268 |
| 2 | `h_gap_top4` | 0.0268 |
| 3 | `a_gap_top4` | 0.0264 |
| 4 | `importance_diff` | 0.0264 |
| 5 | `h_key_players_form_avg_rating` | 0.0261 |
| 6 | `h_key_players_form_avg_contrib` | 0.0257 |
| 7 | `a_key_players_form_avg_contrib` | 0.0256 |
| 8 | `a_rank` | 0.0250 |
| 9 | `a_gap_safety` | 0.0249 |
| 10 | `a_key_players_form_avg_rating` | 0.0247 |

---

## 7. In-Play Model Results

### 7.1 Model Comparison (at 30' checkpoint)

We compare in-play models at the **30-minute mark** — a point where the match is still in the first half with genuine outcome uncertainty, yet enough events have occurred to differentiate model quality.

| # | Script | Model | Accuracy @30' | Accuracy @45' |
|---|--------|-------|--------------|--------------|
| 1 | 3 | Gradient Boosting | **59.41%** | **64.88%** |
| 2 | 6 | Deep MLP (PyTorch) | 55.02% | 59.61% |
| 3 | 7 | XGBoost | 54.53% | 60.11% |
| 4 | 9 | LSTM + Attention | 54.69% | 58.66% |
| 5 | 8 | ResNet | 53.91% | 59.23% |
| 6 | 5 | Advanced Ensemble | 53.37% | 59.49% |

### 7.2 First-Half Prediction — The Most Valuable Window

From a practical standpoint, **first-half predictions (10'–45')** are the most meaningful in-play setting: the match outcome is still genuinely uncertain, yet real-time events have begun to provide actionable signals. By contrast, predictions at 80'–90' are trivially accurate (the score is nearly final), and pre-match predictions lack any in-game information. The first half represents the "sweet spot" where the model must reason under genuine uncertainty with partial information.

#### 7.2.1 First-Half Accuracy by Minute (All Models)

| Minute | Gradient Boosting | XGBoost | ResNet | Deep MLP | LSTM | Advanced Ensemble |
|--------|------------------|---------|--------|----------|------|------------------|
| **10'** | **54.01%** | 47.21% | — | 50.89% | 47.50% | 43.99% |
| **20'** | **56.56%** | 50.97% | — | 52.21% | 50.76% | 48.16% |
| **30'** | **59.41%** | 54.53% | 53.91% | 55.02% | 54.69% | 53.37% |
| **40'** | **61.54%** | 57.50% | — | 58.37% | 56.72% | 56.47% |
| **45'** | **64.88%** | 60.11% | 59.23% | 59.61% | 58.66% | 59.49% |

**Key takeaways from the first half:**

- **10' → 45' gain**: The best model (Gradient Boosting) improves from 54.01% to 64.88% during the first half alone — a **+10.9 percentage point** gain, demonstrating that early match events (first goal, early cards, tactical shape) carry substantial predictive value.
- **10' vs pre-match**: Even at 10 minutes, Gradient Boosting (54.01%) already exceeds the best pre-match model (49.81%) by **+4 percentage points**, showing that just 10 minutes of live data add more signal than 76 engineered pre-match features.
- **45' (half-time)**: At half-time, prediction accuracy reaches 59–65% depending on the model. This is practically useful — the match is only halfway done, yet the model already achieves a meaningful advantage over random guessing (33%) and pre-match prediction (~50%).
- **Model ranking is stable**: Gradient Boosting consistently leads at every first-half checkpoint. The gap between models narrows at later minutes as the signal (goal difference) becomes increasingly dominant.

#### 7.2.2 Full Match Accuracy Progression

For reference, the complete minute-by-minute progression (including second half) is shown below:

| Minute | Gradient Boosting | XGBoost | ResNet | Deep MLP | LSTM | Advanced Ensemble |
|--------|------------------|---------|--------|----------|------|------------------|
| 10' | 54.01% | 47.21% | — | 50.89% | 47.50% | 43.99% |
| 20' | 56.56% | 50.97% | — | 52.21% | 50.76% | 48.16% |
| 30' | 59.41% | 54.53% | 53.91% | 55.02% | 54.69% | 53.37% |
| 40' | 61.54% | 57.50% | — | 58.37% | 56.72% | 56.47% |
| 45' | 64.88% | 60.11% | 59.23% | 59.61% | 58.66% | 59.49% |
| 50' | 65.76% | 62.79% | — | 61.93% | 61.31% | 61.97% |
| 60' | 71.01% | 67.09% | 66.39% | 66.35% | 66.43% | 66.76% |
| 70' | 75.46% | 73.05% | — | 72.10% | 72.18% | 73.09% |
| 75' | — | — | 75.54% | — | — | — |
| 80' | 81.49% | 79.50% | — | 78.71% | 78.59% | 79.79% |
| 90' | **99.38%** | **99.21%** | **95.67%** | **97.48%** | **96.32%** | **99.21%** |

### 7.3 Key Observations

1. **Goal difference is king**: `goal_diff` accounts for **63.35%** of feature importance in the best in-play model (Gradient Boosting), dwarfing all other features. This makes intuitive sense — the current score is by far the strongest indicator of the final result.

2. **First-half predictions are the practical sweet spot**: At half-time (45'), the best model achieves **64.88%** — a **+15.1 percentage point** improvement over the best pre-match model (49.81%). This gain comes from observing only ~45 minutes of events (goals, cards, substitutions), yet it exceeds the total benefit of all 76 pre-match statistical features.

3. **Diminishing information gain in the second half**: The accuracy jump from 10' to 45' (+10.9pp) is comparable to the jump from 45' to 80' (+16.6pp), despite the second half spanning 35 more minutes. The first half provides disproportionate information relative to its duration.

4. **In-play feature importance** (Gradient Boosting, Top 10):

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | `goal_diff` | 0.6335 |
| 2 | `minute` | 0.0513 |
| 3 | `minute_ratio` | 0.0477 |
| 4 | `impact_score_total` | 0.0416 |
| 5 | `h_rank` | 0.0247 |
| 6 | `a_rank` | 0.0241 |
| 7 | `h2h_home_goal_diff_avg` | 0.0200 |
| 8 | `importance_diff` | 0.0178 |
| 9 | `importance_sum` | 0.0152 |
| 10 | `a_avg_gf` | 0.0107 |

5. **Draw prediction improves dramatically**: In-play Draw recall reaches **64%** (vs ~18% pre-match best with decent accuracy) because the model can observe that no goals have been scored and both teams are evenly matched.

---

## 8. Paper Reproduction Experiments

### 8.1 Overview

We selected representative methods from each literature category (see Section 2) and reproduced them under our strict pre-match evaluation protocol. The results are organized by the type of issue encountered.

### 8.2 Reproducing Category A Methods (Leaked Features)

For methods that rely on in-match or half-time features, we ran two variants: (1) a faithful reproduction using the paper's original feature set (where data was available), and (2) a strict pre-match version that removes all features unavailable before kick-off.

| Reproduction | Original Features | Strict Pre-match | Drop |
|-------------|------------------|-----------------|------|
| Bayesian Network (Script `11`) | 54.82% (paper claims 75%) | 49.36% | -5.5pp |
| Bayesian Network + FT goals (leakage test) | **82.63%** | — | — |
| FNN ensemble (Script `16`, best model) | — | 51.55% | vs claimed 70%+ |
| 7-model comparison (Script `16`) | — | 41–52% range | vs claimed 70%+ |

Key findings:
- The Bayesian Network paper's 75% accuracy could not be replicated even with their own feature set (54.82%), likely due to implementation differences. However, adding full-time goals as input immediately pushes accuracy to 82.63%, confirming that high accuracy is driven by label leakage rather than genuine predictive modelling.
- The ML/DL ensemble paper's 7 models (FNN, RF, XGBoost, SVM, Voting, LR, NB) all fall to the **41–52% range** once half-time features are removed — no different from our own pre-match models.

### 8.3 Reproducing Category B Methods (Honest Prediction)

The most methodologically rigorous approach uses **only historical match scores** to derive 6 features. We reproduce this in Script `17`:

| Model | RPS | Accuracy | F1-macro |
|-------|-----|----------|----------|
| ANN (64-32) | **0.2045** | 51.95% | 38.47% |
| ANN (128-64) | 0.2056 | 52.71% | 39.44% |
| k-NN (k=50) | 0.2062 | 52.06% | 42.60% |
| k-NN (k=30) | 0.2066 | 51.08% | 43.23% |
| Naïve Bayes | 0.2203 | 51.73% | 39.79% |
| Null baseline | 0.2317 | 43.99% | 20.37% |

- **Optimal recency window**: Pearson correlation analysis found *n* = 30 matches (r = 0.425)
- Our best RPS (0.2045) slightly improves upon the paper's k-NN (RPS ≈ 0.211), likely because our dataset (12,095 matches, 5 leagues) is larger
- Accuracy remains firmly around 51–53%, consistent with our full model suite
- **Critical insight**: Even with 76 features (vs 6 score-only features), our best pre-match models barely improve upon this minimal baseline — suggesting that pre-match features beyond historical form have sharply diminishing returns

### 8.4 Reproducing Category C Methods (Novel Architectures)

| Method | Script | Our Result | Paper's Claim | Note |
|--------|--------|-----------|--------------|------|
| Edge-computing fusion (BP) | `14` | 50.56% | 87.5% | 8 edge nodes, probability fusion |
| Hybrid Quantum NN | `15` | 33.57% | ~60% | 4 qubits, 1 layer; underfits |
| Player-enhanced ensemble | `13` | 50.19% | — | Adapted from player-centric framework |

None of the novel architectures outperform standard methods on our dataset. The Quantum Neural Network performs significantly **worse** than a simple Logistic Regression baseline (33.57% vs 48.45%), suggesting that current quantum circuit models are not yet competitive for this class of tabular classification problems.

---

## 9. Misclassification Analysis

Script `10_analyze_misclassifications.py` provides detailed error analysis for both pre-match and in-play models.

### 9.1 Pre-Match Error Patterns

**Overall test accuracy**: 49.40%

| True / Predicted | Away Win | Draw | Home Win |
|-----------------|----------|------|----------|
| **Away Win** | **359** | 78 | 330 |
| **Draw** | 210 | **60** | 340 |
| **Home Win** | 205 | 61 | **776** |

**Per-class accuracy:**
- Home Win: 74.47% (relatively easy — strong home bias)
- Away Win: 46.81% (moderate difficulty)
- Draw: **9.84%** (extremely poor — nearly random)

**Top confusion pairs and feature analysis:**

1. **Draw → Home Win (340 errors)**: The model predicts Home Win when the match ends in a Draw. Characteristic pattern: the home team has a lower (better) league rank (9.49 vs 12.87) and a higher safety gap. Essentially, when the home team "should" win but doesn't, the model gets it wrong.

2. **Away Win → Home Win (330 errors)**: The model picks the wrong winner. The home team's H2H record is better and point differential is higher. These are matches where historical H2H advantage fails.

3. **Draw → Away Win (210 errors)**: The away team appears stronger on paper. The away team has better ranking (8.13 vs 12.13) and lower gap to top 4.

**Highest-confidence errors** (pre-match model was highly confident but wrong):
- 2022-01-07 Bayern Munich vs Borussia Mönchengladbach: Predicted H (93.48% confidence), actual A
- 2025-12-14 Bayern München vs FSV Mainz 05: Predicted H (93.47% confidence), actual D
- 2022-05-08 Paris Saint Germain vs Estac Troyes: Predicted H (92.44% confidence), actual D

All top-confidence errors are **strong favourites failing** — a fundamental unpredictability in football.

### 9.2 In-Play Error Patterns

**Overall test accuracy**: 68.97%

| True / Predicted | Away Win | Draw | Home Win |
|-----------------|----------|------|----------|
| **Away Win** | **5387** | 1627 | 656 |
| **Draw** | 1099 | **3923** | 1080 |
| **Home Win** | 787 | 2256 | **7375** |

The most common in-play error is **Home Win → Draw (2256 cases)** — the home team is winning at a checkpoint but the match eventually ends in a draw (late equaliser). This is followed by **Away Win → Draw (1627 cases)** — same pattern for away team leads.

**Average model confidence:**
- Correct predictions: 0.7178
- Wrong predictions: 0.5245

The clear confidence gap (72% vs 52%) indicates the model is well-calibrated — it is less certain about its mistakes.

---

## 10. Discussion

### 10.1 The ~50% Pre-Match Ceiling

Our most important finding is the **robust accuracy ceiling at approximately 50%** for strict pre-match prediction. This result is consistent across:
- 18 model architectures (from Logistic Regression to LSTM to Quantum Neural Networks)
- Multiple feature sets (6 features to 76 features)
- Different evaluation horizons (single-season to 7-season)
- Multiple papers' methodologies (Bayesian, ensemble, deep learning)

A 50% accuracy on a 3-class problem with base rates of [43%, 25%, 32%] translates to only marginal lift over the majority-class baseline (43%). This suggests that **football match outcomes are fundamentally difficult to predict before kick-off**, at least from the publicly available statistical information.

### 10.2 Why Published Papers Report Higher Numbers

Our reproducibility study reveals that papers claiming 70%+ pre-match accuracy almost universally rely on one or more of:

1. **In-match features disguised as pre-match**: Half-time goals (Atta Mills 2024), shots/corners/fouls (Razali 2017), or other in-play statistics are used as input features. These are not pre-match features.
2. **Label leakage**: Goal difference (Haruna 2022) is directly derived from the target variable.
3. **Random cross-validation**: Splitting data randomly rather than temporally allows the model to train on matches that occurred after its test matches, learning future trends.
4. **Publication bias**: Papers reporting ~50% accuracy (the honest result) are less likely to be published or cited than those claiming 80%+ accuracy.

### 10.3 The Value of In-Play Prediction

The in-play models demonstrate that **real-time match information is extremely predictive**:
- `goal_diff` alone explains 63% of prediction accuracy
- By minute 60, accuracy reaches 71%, and by minute 80 it reaches 81%
- At minute 90, accuracy is ~99% (trivially, knowing the current score near full-time is nearly sufficient)

This has practical implications for **live betting markets**, **broadcast analytics**, and **tactical decision-support systems** — applications where real-time prediction has genuine value, unlike pre-match prediction where the honest ceiling is ~50%.

### 10.4 Draw Prediction — The Unsolved Problem

Across all experiments, Draw remains the hardest outcome to predict:

| Model Type | Draw Recall | Draw F1 |
|-----------|-------------|---------|
| Pre-match best | 43% | 34% |
| Pre-match worst | 0% | 0% |
| In-play best | 64% | 56% |

Draws occur when two teams are evenly matched AND no team manages to score an extra goal — a conjunction of skill balance and luck. The low Draw recall is not a model failure but a reflection of the inherent unpredictability of this outcome.

### 10.5 Feature Engineering Insights

1. **Diminishing returns from more features**: Berrar's 6-feature score-only model (RPS=0.2045) is not significantly worse than our 76-feature models. Complex player-level and league-position features provide only marginal improvement.

2. **Player form rating is consistently important**: The `h_key_players_form_avg_rating` and `a_key_players_form_avg_rating` features consistently appear in the top-10 across multiple models, suggesting that individual player quality (as measured by historical match ratings) is one of the stronger pre-match signals.

3. **Motivation matters more than expected**: `importance_diff` (how much more one team needs the win) ranks consistently in the top-5 features, surpassing many traditional statistics like win rate or goals scored.

4. **Contextual features add value**: League position gaps (`h_gap_top4`, `a_gap_safety`) and scheduling factors (`diff_days_since_last_match`) contribute meaningfully, suggesting that the "external circumstances" of a match matter beyond just team quality.

### 10.6 Model Complexity vs Performance

| Model Category | Best Pre-Match Acc | Complexity |
|---------------|-------------------|------------|
| Logistic Regression | 48.45% | Very Low |
| Gradient Boosted Trees | 49.81% | Medium |
| Deep MLP | 47.29% | High |
| LSTM + Attention | 45.68% | Very High |
| Quantum Neural Network | 33.57% | Extreme |

There is no meaningful accuracy gain from increasing model complexity. The simplest models (Logistic Regression) perform within 2 percentage points of the most complex (LSTM + Attention). The Quantum Neural Network actually performs significantly worse, likely due to the difficulty of training quantum circuits on noisy classical data with limited qubits (4 qubits).

This strongly suggests the bottleneck is **information, not model capacity** — the pre-match features simply do not contain enough predictive signal to discriminate outcomes beyond ~50%.

---

## 11. Conclusion

### 11.1 Main Contributions

1. **Large-scale multi-league benchmarking**: We constructed a dataset of 12,095 fixtures across 5 major European leagues and 7 seasons, with 76 principled pre-match features and 89 in-play features.

2. **Comprehensive model comparison**: We tested 18 model architectures including classical ML, gradient-boosted trees, deep learning, recurrent models, Bayesian networks, edge-computing fusion, and quantum neural networks.

3. **Critical reproducibility study**: We systematically reproduced 6 published papers and demonstrated that most claimed high accuracies are artefacts of feature leakage, not genuine prediction capability.

4. **Honest performance assessment**: Strict pre-match prediction accuracy is robustly ~50% (±2%), regardless of model complexity. This is only marginally above the majority-class baseline of 43%.

5. **In-play prediction value**: Real-time match events dramatically improve prediction accuracy, reaching 69%+ overall and 99%+ near full-time.

### 11.2 Practical Implications

- **For researchers**: Pre-match football prediction papers should be evaluated with extreme scepticism. Any reported accuracy above 55% using only pre-match features warrants careful feature auditing for temporal leakage.
- **For practitioners**: In-play prediction models have genuine practical value for live applications. Pre-match prediction should be treated as a probabilistic tool (producing calibrated probabilities) rather than a deterministic classifier.
- **For the field**: The honest ~50% accuracy ceiling suggests that future progress in pre-match prediction may require fundamentally new information sources (e.g., tactical video analysis, detailed training data, psychological state monitoring) rather than more sophisticated models.

### 11.3 Limitations

1. **Five European leagues only**: Our dataset covers major leagues but does not include South American, Asian, or lower-tier competitions, which may have different predictability characteristics.
2. **No betting odds features**: Many top prediction models in the literature use bookmaker odds as features, which encode expert/market knowledge. We excluded these to focus on "pure" statistical prediction.
3. **No fine-grained in-play features**: Our in-play model uses checkpoint-based event statistics (goals, cards at minute T) but cannot access real-time tracking data (xG, possession trajectories, pressing intensity) which could further improve predictions.
4. **Temporal scope**: 2019–2026 covers a specific era of football; tactical evolution over longer periods was not studied.

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
│   │   ├── api_football/        # Raw API data (~12,000 fixture directories)
│   │   └── statsbomb-open-data/ # StatsBomb open data
│   └── processed/
│       ├── pretrain_dataset.csv # 12,095 rows × 85 columns
│       ├── inplay_dataset.csv   # 120,950 rows × 96 columns
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
