<div align="center">

# 🏦 Portuguese Bank Telemarketing Intelligence
### AI-Powered Lead Propensity Scoring & Strategic Marketing Analytics

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-Champion_Model-2E7D32?style=for-the-badge&logo=lightning&logoColor=white)](https://lightgbm.readthedocs.io)
[![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-FF6F00?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Status](https://img.shields.io/badge/Status-Production_Ready-0284C7?style=for-the-badge)](https://github.com/babiazees007/Bank_Marketing_Intelligence_)
[![License](https://img.shields.io/badge/License-MIT-6B7280?style=for-the-badge)](LICENSE)

<br/>

**Transforming brute-force outbound telemarketing into an AI-guided lead scoring operation.**  
*By resolving subtle call-duration target leakage and deploying calibrated gradient-boosted decision trees, this solution eliminates 80% of wasted cold calls while preserving 72.5% of total term deposit acquisitions.*

<br/>

| 📞 Dialing Overhead Slashed | 🎯 Deposit Conversions Retained | ⚡ Telemarketer Strike Rate | 🛡️ Target Leakage |
| :---: | :---: | :---: | :---: |
| **-80%** *(32,950 calls saved)* | **72.46%** *(3,362 sales kept)* | **3.6x** *(11.27% ➔ 40.72%)* | **0%** *(Strict Pre-Call)* |

<br/>

[Explore Live Web App (index.html)](file:///c:/Users/Admin/Downloads/PRCP-1000-ProtugeseBank/index.html) • [View Executed Notebook](file:///c:/Users/Admin/Downloads/PRCP-1000-ProtugeseBank/Bank_Marketing_Term_Deposit_Prediction.ipynb) • [Methodology](#system-architecture--methodology) • [Model Benchmarks](#predictive-modeling--benchmark-results) • [Strategic Playbook](#the-7-pillar-strategic-action-playbook)

</div>

---

## 📋 Table of Contents
1. [Executive Summary & Problem Statement](#executive-summary--problem-statement)
2. [The Critical "Call Duration" Data Leakage Trap](#the-critical-call-duration-data-leakage-trap)
3. [System Architecture & Methodology](#system-architecture--methodology)
4. [Exploratory Data Analysis & Statistical Hypotheses](#exploratory-data-analysis--statistical-hypotheses)
5. [Predictive Modeling & Benchmark Results](#predictive-modeling--benchmark-results)
6. [Decile Prioritization & Cumulative Gains (80/20 Rule)](#decile-prioritization--cumulative-gains-8020-rule)
7. [The 7-Pillar Strategic Action Playbook](#the-7-pillar-strategic-action-playbook)
8. [Projected Financial & Operational ROI](#projected-financial--operational-roi)
9. [Repository Structure & Deliverables](#repository-structure--deliverables)
10. [Quickstart & Reproduction Guide](#quickstart--reproduction-guide)

---

## 🎯 Executive Summary & Problem Statement

Retail banks frequently deploy outbound telemarketing campaigns to sell **term deposits** (long-term, fixed-rate savings certificates). In the historical Portuguese banking dataset (41,188 interactions between May 2008 and November 2010), marketing was conducted as an un-targeted, brute-force cold calling operation.

```
Total Outbound Calls: 41,188 Interactions
├── Term Deposit Subscribed (y = 'yes'):   4,640 clients  (11.27%)  [Actual Sales]
└── Term Deposit Rejected   (y = 'no'):   36,548 clients  (88.73%)  [Wasted Labor & Overhead]
```

### Core Business Inefficiencies:
1. **Severe 88.73% Call Waste**: 7.88 out of every 8 telephone calls ended in rejection. Telemarketers spent almost 90% of their shifts speaking with uninterested prospects, demoralizing staff and burning customer goodwill.
2. **Seasonal Misallocation**: Over **45% of total annual call volume** was concentrated in **May (13,769 calls)** and **July (7,174 calls)**, where conversion rates were dismal (**6.4%** and **9.0%** respectively). Meanwhile, peak-converting autumn/winter months (**March at 50.5%**, **Sept at 44.9%**, **Oct at 43.8%**, and **Dec at 48.9%**) received less than 5% of total outreach.
3. **Channel Inefficiency**: Outbound dials to fixed landlines converted at only **5.23%**, whereas cellular mobile outreach converted at **14.74%** (a **2.82x efficiency lift**).
4. **Call Fatigue & Diminishing Returns**: Marginal conversion plummeted from **13.04% on call 1** to **10.77% on call 3**, collapsing to **2.12% after 6+ attempts**.
5. **Class Imbalance Trap**: Because buyers account for only 11.27% of instances, a naïve model predicting "No" achieves 88.73% accuracy while offering **zero economic utility**. Evaluation strictly requires **PR-AUC, Balanced Accuracy, Recall, and Decile Lift**.

---

## ⚠️ The Critical "Call Duration" Data Leakage Trap

A frequent pitfall in bank marketing analytics is the uncritical inclusion of the `duration` column (call duration in seconds). 

> [!CAUTION]
> **Why `duration` MUST be discarded in prospective lead generation:**
> - Call duration exhibits a high linear correlation ($r = 0.405$) with deposit subscriptions.
> - When a customer agrees to buy, the phone call takes 15–20 minutes to explain contractual terms and finalize account opening ($duration \approx 900s \rightarrow y = \text{'yes'}$).
> - When a customer is uninterested, they hang up after 10 seconds ($duration \approx 10s \rightarrow y = \text{'no'}$).
> - **The catch**: The duration of a phone call is **100% unknown before the call is placed**. Once duration is known, the call has completed and the sale is already won or lost.

A model trained with `duration` achieves an artificial, illusory ROC-AUC of **0.9472**, but **cannot score leads in CRM pipelines prior to dialing**.



## 🛠️ System Architecture & Methodology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DATA SCIENCE & ML ARCHITECTURE                        │
└─────────────────────────────────────────────────────────────────────────────┘
  1. Data Hygiene        ──► 41,188 rows, 21 attributes; deduplication;
                             explicit encoding of 'unknown' categories.
  2. Feature Engineering ──► pdays_contacted flag, pdays_group (recency cohorts:
                             0-7d, 8-14d, 15+d, never), age cohorts, campaign caps.
  3. Preprocessing Pipe  ──► ColumnTransformer: StandardScaler for numerics,
                             OneHotEncoder(drop='first') for categoricals.
  4. Class Balancing     ──► scale_pos_weight = 7.87, class_weight='balanced'.
  5. Cross-Validation    ──► 5-Fold Stratified CV encapsulated in Pipelines.
  6. Calibration         ──► Threshold optimization tuning F1 from 0.50 down to 0.28.
  7. Decile Scoring      ──► Ranked Cumulative Gains curve & Decile Lift profiling.
```

### Handled Technical Challenges:

| Challenge | Impact on Modeling | Technical Mitigation |
| :--- | :--- | :--- |
| **Duration Leakage** | Produces fake 0.947 AUC models that cannot score leads before dialing. | Discarded `duration` from production; built dual benchmark to prove the leakage mechanism. |
| **Class Imbalance (7.88:1)** | Standard loss functions learn to predict the majority negative class. | Cost-sensitive loss weighting (`scale_pos_weight = 7.87`), stratified CV, and threshold tuning. |
| **Macro Multicollinearity** | $\rho(\text{euribor3m}, \text{emp.var.rate}) = 0.972$, distorting regression weights. | Applied L2 Ridge shrinkage to linear baselines; deployed tree-based gradient boosters robust to collinearity. |
| **Masked Missingness** | Standard `.isnull()` was 0, but >20% of `default` was `'unknown'`. | Preserved `'unknown'` as an explicit informative category (`default='unknown'` converted 2.5x lower). |
| **Bimodal pdays Artifact** | 96.3% of `pdays` was `999` (never contacted), breaking continuous scaling. | Engineered binary indicator `pdays_contacted` and partitioned recency into discrete cohorts. |

---

## 📊 Exploratory Data Analysis & Statistical Hypotheses

### 1. Categorical Associations (Chi-Square & Cramer's V)
Statistical hypothesis tests were conducted across all categorical features against target outcome $y$:

| Feature | Chi-Square ($\chi^2$) | p-value | Cramer's V | Effect Size Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **poutcome** (Previous Campaign) | 4,230.5 | $< 10^{-15}$ | **0.320** | **Strong**: Past success is the #1 predictor |
| **month** (Seasonality) | 3,101.1 | $< 10^{-15}$ | **0.274** | **Moderate-High**: Extreme seasonal variation |
| **contact** (Communication Channel) | 863.2 | $< 10^{-15}$ | **0.145** | **Moderate**: Cellular vastly superior to landline |
| **default** (Credit Default) | 441.2 | $< 10^{-15}$ | **0.103** | **Moderate**: Known non-defaulters convert 2.5x higher |
| **job** (Occupation) | 961.2 | $< 10^{-15}$ | **0.153** | **Moderate**: Students & retirees convert at top rates |
| **education** (Education Level) | 163.2 | $< 10^{-15}$ | **0.063** | Weak-Moderate |
| **marital** (Marital Status) | 122.2 | $< 10^{-15}$ | **0.054** | Weak |

### 2. Numerical Divergence (Mann-Whitney U & Rank-Biserial Correlation)

| Feature | Mann-Whitney U ($U$) | p-value | Rank-Biserial ($r$) | Direction & Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **euribor3m** | 3.51e7 | $< 10^{-15}$ | **-0.474** | Inverse: Low interest rates trigger deposit demand |
| **nr.employed** | 3.56e7 | $< 10^{-15}$ | **-0.466** | Inverse: Lower employee index corresponds to easing |
| **emp.var.rate** | 3.78e7 | $< 10^{-15}$ | **-0.433** | Inverse: Economic contraction boosts fixed savings |
| **pdays** | 5.72e7 | $< 10^{-15}$ | **-0.142** | Inverse: Recent prior contact boosts conversion |
| **previous** | 6.02e7 | $< 10^{-15}$ | **+0.098** | Positive: Higher interaction history boosts conversion |
| **campaign** | 7.91e7 | $< 10^{-15}$ | **-0.069** | Inverse: Contact fatigue degrades conversion |

---

## 🔬 Predictive Modeling & Benchmark Results

### Holdout Test Set Evaluation (8,238 Unseen Clients — 20% Stratified Split)

Each operational model was evaluated strictly on pre-call attributes using **Stratified 5-Fold Cross-Validation** during training, and scored on the pristine holdout test set:

| Model Architecture | Duration Included? | Test ROC-AUC | Test PR-AUC | Balanced Acc. | F1-Score | Recall | Precision | Production Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Benchmark LightGBM** | **Yes (Leakage)** | **0.9472** | **0.6385** | **0.7812** | **0.6214** | **0.6125** | **0.6306** | 🚫 Infeasible offline ceiling |
| **Tuned LightGBM** | **No (Pre-Call)** | **0.7984** | **0.4621** | **0.7385** | **0.5342** | **0.6584** | **0.4491** | 🏆 **Champion (Deploy)** |
| **XGBoost Classifier** | No (Pre-Call) | 0.7932 | 0.4578 | 0.7321 | 0.5289 | 0.6519 | 0.4452 | 🥈 Strong Runner-Up |
| **Random Forest (150 Trees)** | No (Pre-Call) | 0.7891 | 0.4412 | 0.7248 | 0.5187 | 0.6401 | 0.4361 | 🥉 Bagging Baseline |
| **Logistic Regression (L2)** | No (Pre-Call) | 0.7854 | 0.4289 | 0.7210 | 0.5098 | 0.6328 | 0.4271 | ⚖️ Interpretable Linear |
| **Decision Tree (Pruned)** | No (Pre-Call) | 0.7412 | 0.3541 | 0.6854 | 0.4421 | 0.5912 | 0.3534 | ⚠️ High Variance |

> [!TIP]
> **Why Tuned LightGBM Won:**
> 1. **Superior PR-AUC (0.4621)**: In highly imbalanced banking domains, PR-AUC is the gold standard for separating true buyers from noise.
> 2. **Calibrated Threshold ($\tau = 0.28$)**: Shifts the operational operating point to achieve **65.84% recall** without inundating agents with false alarms.
> 3. **Sub-millisecond Inference**: Capable of batch scoring 100,000 CRM client records in under 3 seconds.

---

## 📈 Decile Prioritization & Cumulative Gains (80/20 Rule)

By sorting prospective leads into 10 equal deciles based on predicted LightGBM probability, the bank achieves massive efficiency gains:

```
DECILE CUMULATIVE GAINS CURVE:
Decile 1 (Top 10% Leads) ──────────────► Captures 48.16% of ALL Subscribers  (Lift: 4.80x)
Decile 2 (Top 20% Leads) ──────────────► Captures 72.46% of ALL Subscribers  (Lift: 2.42x)  ◄── OPTIMAL CUTOFF
Decile 3 (Top 30% Leads) ──────────────► Captures 82.83% of ALL Subscribers  (Lift: 1.03x)
Deciles 4–10 (Remaining 70% Leads) ────► Dwindling returns (avg conversion < 4.0%)
```

### Empirical Decile Lift Breakdown:

| Decile Tier | Total Leads | Actual Subscribers | Cumulative Leads % | Cumulative Responders % | Conversion Rate | Lift Factor |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decile 1** | 824 | 446 | 10.0% | **48.16%** | **54.13%** | **4.80x** |
| **Decile 2** | 824 | 225 | **20.0%** | **72.46%** | **27.31%** | **2.42x** |
| **Decile 3** | 824 | 96 | 30.0% | 82.83% | 11.65% | 1.03x |
| **Decile 4** | 824 | 57 | 40.0% | 88.99% | 6.92% | 0.61x |
| **Decile 5** | 824 | 35 | 50.0% | 92.76% | 4.25% | 0.38x |
| **Deciles 6–10** | 4,118 | 67 | 100.0% | 100.00% | 1.63% | 0.14x |

> [!IMPORTANT]
> **The Bottom Line Rule**: Restricting telemarketers strictly to **Deciles 1 and 2 (Top 20% of leads)** eliminates **80% of call center dialing volume** while retaining **72.46% of total deposit subscriptions**, boosting agent conversion from **11.27% to 40.72%**.

---

## 🏛️ The 7-Pillar Strategic Action Playbook

| Pillar | Operational Action | Expected Business Lift |
| :--- | :--- | :--- |
| **1. Strict Decile Targeting** | Pre-score all CRM leads monthly with LightGBM. Route only Deciles 1 and 2 to telemarketers. Discontinue cold outbound dials to the lower 80%. | **+3.6x Strike Rate Lift**; cuts 80% of dial overhead |
| **2. Seasonal Budget Pivot** | Reallocate marketing headcount away from May and July (&lt;9% conv.) toward March, September, October, and December (&gt;43% conv.). | **5.2x Seasonal Efficiency Gain** |
| **3. Cellular-First Mandate** | Mandate outbound calls to verified mobile numbers (14.7% conv.) over landlines (5.2% conv.). Prompt online/mobile app users to verify cell numbers. | **2.82x Channel Multiplier** |
| **4. The "3-Call Fatigue" Rule** | Enforce an automated CRM lock after 3 unsuccessful dials per campaign. Conversion drops to 2% on call 6+ while creating brand fatigue. | **Zero Agent Burnout**; saves ~12% call capacity |
| **5. 7-Day Warm Queue** | Customers who previously subscribed (`poutcome='success'`) convert at 65.1%. Queue these warm leads for follow-up within 3–7 days. | **65.1% Repeat Conversion Rate** |
| **6. Macro Counter-Cyclical Trigger** | Trigger automated deposit campaigns when 3-month Euribor drops below 1.5%—investors naturally seek fixed, guaranteed returns during uncertainty. | **Counter-Cyclical Liquidity Capture** |
| **7. Persona-Specific Value Props** | Pitch capital security and monthly dividend yields to seniors (>60 yrs, 45.5% conv.); pitch automated micro-savings to students (<25 yrs, 31.4% conv.). | **Maximized Segment Resonance** |

---

## 💰 Projected Financial & Operational ROI

Assuming an institutional campaign sizing of 41,188 leads at an average call center fully-loaded cost of **$6.50 per call attempt**:

| Campaign Metric | Traditional Cold Calling (100% Volume) | AI-Prioritized (Top 2 Deciles) | Variance / Delta |
| :--- | :---: | :---: | :---: |
| **Outbound Calls Placed** | 41,188 calls | **8,238 calls** | **-32,950 calls (-80.0%)** |
| **Term Deposits Sold** | 4,640 deposits | **3,362 deposits** | **72.46% sales captured** |
| **Agent Conversion Rate** | 11.27% | **40.72%** | **+29.45% (+3.6x lift)** |
| **Total Dialing Expenditure** | $267,722 | **$53,547** | **+$214,175 budget saved (-80%)** |
| **Cost per Acquired Customer (CAC)** | **$57.70** | **$15.93** | **-$41.77 per customer (-72.4%)** |
| **Agent Hours Required (@10 calls/hr)** | 4,118 hours | **824 hours** | **3,294 agent hours saved** |

---

## 📂 Repository Structure & Deliverables

```
Bank_Marketing_Intelligence_/
├── index.html                                   # Executive Web Presentation & Interactive Decision Simulator
├── Bank_Marketing_Term_Deposit_Prediction.ipynb # Fully executed, reproducible Jupyter Notebook (1.55 MB)
├── generate_bank_notebook.py                    # Programmatic Python generator using nbformat & nbclient
├── pyproject.toml                               # Python project dependencies & environment metadata
├── README.md                                    # Comprehensive technical documentation & playbook
└── Data/
    ├── bank-additional/
    │   ├── bank-additional-full.csv             # Primary dataset (41,188 rows x 21 columns)
    │   ├── bank-additional.csv                  # 10% sample dataset (4,119 rows)
    │   └── bank-additional-names.txt            # Dataset dictionary & metadata
    ├── bank-full.csv                            # Older dataset version (45,211 rows)
    └── bank.csv                                 # Older 10% sample
```

### Key Deliverables:
- **[index.html](file:///c:/Users/Admin/Downloads/PRCP-1000-ProtugeseBank/index.html)**:
  - Custom modern typography (**Sora** display + **Inter** body + **JetBrains Mono** data).
  - Dual Theme: **Dark Obsidian Nebula** & **Executive Light Banking Mode** with live theme toggle.
  - Interactive **Decile ROI Simulator**: dynamically calculates calls eliminated, deposits retained, and cost savings.
  - Live **Lead Propensity Sandbox**: test prospect age, channel, month, and past outcome to receive immediate priority tier and conversation guidance.
  - Interactive **Chart.js Visualizations**: Cumulative gains, seasonality vs call volume, pre-call feature importance, call fatigue.
- **[Bank_Marketing_Term_Deposit_Prediction.ipynb](file:///c:/Users/Admin/Downloads/PRCP-1000-ProtugeseBank/Bank_Marketing_Term_Deposit_Prediction.ipynb)**:
  - Complete, fully executed notebook with all outputs, statistical tables, seaborn charts, and model evaluations saved directly inside.

---

## 🚀 Quickstart & Reproduction Guide

### 1. Launch the Executive Decision Dashboard
Open `index.html` in any modern web browser:
```bash
# Windows
start index.html

# macOS
open index.html

# Linux
xdg-open index.html
```

### 2. Run the Machine Learning Jupyter Notebook
```bash
# Clone the repository
git clone https://github.com/babiazees007/Bank_Marketing_Intelligence_.git
cd Bank_Marketing_Intelligence_

# Install required dependencies
pip install numpy pandas scikit-learn lightgbm xgboost matplotlib seaborn scipy jupyterlab nbclient nbformat

# Launch JupyterLab
jupyter lab Bank_Marketing_Term_Deposit_Prediction.ipynb
```

### 3. Programmatically Rebuild the Notebook (Optional)
To regenerate and execute the entire notebook from scratch with clean outputs:
```bash
python generate_bank_notebook.py
```

---

<div align="center">

### Developed for Retail Banking Telemarketing Optimization
*Data Source: Portuguese Bank Direct Marketing Campaigns (UCI Machine Learning Repository)*

[![GitHub Stars](https://img.shields.io/github/stars/babiazees007/Bank_Marketing_Intelligence_?style=social)](https://github.com/babiazees007/Bank_Marketing_Intelligence_)
[![GitHub Forks](https://img.shields.io/github/forks/babiazees007/Bank_Marketing_Intelligence_?style=social)](https://github.com/babiazees007/Bank_Marketing_Intelligence_)

</div>
