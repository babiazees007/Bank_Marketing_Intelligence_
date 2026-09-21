# Portuguese Bank Marketing: Term Deposit Prediction & Strategic Analytics

An end-to-end data science, machine learning, and strategic business advisory solution analyzing direct telemarketing campaigns for long-term deposit subscriptions by a Portuguese banking institution (May 2008 – November 2010).

---

## Deliverables & Repository Structure

- **[index.html](file:///c:/Users/Admin/Downloads/PRCP-1000-ProtugeseBank/index.html)**: Interactive Executive Web Presentation & Decision Simulator featuring a live ROI & Decile Lead Prioritization Calculator, Real-Time Propensity Scoring Sandbox, dynamic Chart.js visualizations, and the 7-Pillar Strategic Playbook. Open directly in any modern browser.
- **[Bank_Marketing_Term_Deposit_Prediction.ipynb](file:///c:/Users/Admin/Downloads/PRCP-1000-ProtugeseBank/Bank_Marketing_Term_Deposit_Prediction.ipynb)**: Fully executed, self-contained Jupyter Notebook (1.55 MB) containing all Python code, statistical hypothesis tests, charts, operational model pipelines, and decile evaluations.
- **[generate_bank_notebook.py](file:///c:/Users/Admin/Downloads/PRCP-1000-ProtugeseBank/generate_bank_notebook.py)**: Reproducible Python script using `nbformat` and `nbclient` to programmatically assemble and execute the entire notebook.
- **`Data/bank-additional/bank-additional-full.csv`**: Primary dataset containing 41,188 interactions and 21 demographic, operational, and macroeconomic attributes.

---

## 1. What is the Problem Statement?

The bank's telemarketing division relies on outbound phone calls to sell fixed-rate **term deposits** (long-term savings instruments). Under historical operations, the campaign was run as a brute-force, spray-and-pray exercise that suffered from severe inefficiencies:

### 1. The 88.7% Wasted Call Trap
- Across 41,188 outbound customer calls, only **4,640 interactions (11.27%)** resulted in a successful subscription (`y='yes'`).
- The remaining **36,548 calls (88.73%)** produced negative outcomes, consuming hundreds of call center agent hours, inflating acquisition costs, and fatiguing uninterested clients.
- Naïve accuracy is deceptive: a dumb model predicting "No" on every call attains 88.73% accuracy while producing **zero business value**.

### 2. Severe Seasonal Budget Misallocation
- The bank deployed over **45% of its entire call budget in May (13,769 calls) and July (7,174 calls)**—months with poor conversion rates (**6.4%** and **9.0%** respectively).
- Conversely, peak-converting months—**March (50.5%)**, **September (44.9%)**, **October (43.8%)**, and **December (48.9%)**—received less than 5% of total call volume combined.

### 3. Channel Inefficiency
- Outbound outreach to fixed telephone landlines converted at only **5.23%**, whereas cellular mobile outreach converted at **14.74%** (a **2.82x multiplier**). Landline calling consumed substantial agent bandwidth with negligible return.

### 4. Contact Fatigue & Diminishing Returns
- Calling clients repeatedly produced rapidly diminishing returns: conversion dropped from **13.04% on call 1** to **10.77% on call 3**, collapsing to **2.12% after 6+ attempts**. Continuing to call past 3 attempts destroyed brand goodwill without generating sales.

### 5. The Call Duration Target Leakage Trap
- In raw historical logs, the feature `duration` (call duration in seconds) has a strong correlation ($r = 0.405$) with term deposit sales: an interested customer stays on the line for 15 minutes to finalize paperwork, whereas an uninterested customer hangs up within seconds.
- **The Catch**: Call duration is **100% unknown before dialing**. A model trained with `duration` achieves an artificial ~0.947 ROC-AUC in testing, but is completely unusable in production because telemarketers must prioritize leads *prior* to placing the call.

---

## 2. What Have We Solved? (Technical Mitigations)

We resolved 5 critical data science obstacles to engineer a reliable, leak-free operational lead scoring engine:

| Challenge | Nature of the Obstacle | Engineering Solution & Rationale |
| :--- | :--- | :--- |
| **1. Duration Target Leakage** | `duration` leaks the post-call outcome, falsely inflating offline metrics ($r = 0.405$). | **Bifurcated Architecture**: Built an explicit Leakage Benchmark model to document the ceiling, while stripping `duration` completely from the Operational Production Pipeline to score leads pre-call. |
| **2. Severe Class Imbalance (7.88:1)** | With only 11.27% positive cases, standard cross-entropy and accuracy ignore the minority class. | Deployed cost-sensitive weighting (`scale_pos_weight = 7.87` in LightGBM/XGBoost, `class_weight='balanced'`), Stratified 5-Fold Cross-Validation, and calibrated the decision threshold down to **0.28**. |
| **3. Macro Multicollinearity ($\rho = 0.972$)** | `euribor3m`, `emp.var.rate`, and `nr.employed` move in lockstep, inflating variance in linear coefficients. | Applied L2 ridge regularization to linear baselines and deployed tree-based gradient boosters (LightGBM/XGBoost) that handle collinear splits natively. |
| **4. Masked Missingness (`'unknown'`)** | Standard `.isnull()` audits returned 0, yet >20% of `default` was masked as `'unknown'`. | Preserved `'unknown'` as an explicit informative category. Statistical tests proved `default='unknown'` clients convert at 2.5x lower rates (5.15% vs 12.83%) than known non-defaulters. |
| **5. Bimodal `pdays` Artifact (96.3% = 999)** | 96.3% of `pdays` was encoded as `999` (never previously contacted), distorting numerical distributions. | Engineered binary indicator `pdays_contacted` and recency cohorts (`0-7 days`, `8-14 days`, `15+ days`, `never contacted`). |

---

## 3. What Have We Done? (Implementation & Results)

### Task 1: Complete Data Analysis Report (EDA & Statistical Insights)
1. **Statistical Significance Testing**:
   - Categoricals: Chi-Square tests with Cramer's V effect size confirmed `poutcome` (Cramer's V = 0.312), `month` (0.267), and `contact` (0.145) as primary categorical drivers ($p < 0.001$).
   - Numerics: Mann-Whitney U tests confirmed `euribor3m` (rank-biserial = 0.462), `nr.employed` (0.457), and `emp.var.rate` (0.448) as top macroeconomic indicators.
2. **Prior Campaign Track Record**:
   - Clients with previous campaign `success` converted at an extraordinary **65.11%** (a 5.78x lift over baseline).
   - Clients contacted within the prior 0–7 days (`pdays <= 7`) converted at **64.2%**.
3. **Demographic Sweet Spots**:
   - **Retirees (>60 yrs, 25.2%–45.5% conversion)** and **Students (<25 yrs, 31.4% conversion)** convert at double or triple the rate of middle-aged working clients.

---

### Task 2: Predictive Modeling & Benchmarks (Holdout Test Set: 8,238 Clients)

We evaluated 5 operational algorithms (strictly without call duration) alongside the leakage benchmark:

| Model Architecture | Duration Included? | Test ROC-AUC | Test PR-AUC | Balanced Acc. | F1-Score | Recall | Precision | Status / Decision |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Benchmark LightGBM** | **Yes (Leakage)** | **0.9472** | **0.6385** | **0.7812** | **0.6214** | **0.6125** | **0.6306** | Infeasible offline ceiling |
| **Tuned LightGBM (Operational)** | **No (Pre-Call Pure)** | **0.7984** | **0.4621** | **0.7385** | **0.5342** | **0.6584** | **0.4491** | **Champion for Deployment** |
| **XGBoost Classifier** | No (Pre-Call Pure) | 0.7932 | 0.4578 | 0.7321 | 0.5289 | 0.6519 | 0.4452 | Strong Runner-Up |
| **Random Forest (150 Trees)** | No (Pre-Call Pure) | 0.7891 | 0.4412 | 0.7248 | 0.5187 | 0.6401 | 0.4361 | Reliable Bagging Baseline |
| **Logistic Regression (L2)** | No (Pre-Call Pure) | 0.7854 | 0.4289 | 0.7210 | 0.5098 | 0.6328 | 0.4271 | Linear Baseline |
| **Decision Tree (Pruned)** | No (Pre-Call Pure) | 0.7412 | 0.3541 | 0.6854 | 0.4421 | 0.5912 | 0.3534 | High Variance |

#### Why Tuned LightGBM Won:
- Achieved the highest **PR-AUC (0.4621)** and **Balanced Accuracy (73.85%)**.
- Maximized minority recall (**65.84%**) under calibrated thresholding ($\tau = 0.28$) without drowning telemarketers in false positive leads.
- Sub-millisecond inference time capable of scoring entire CRM lead lists instantaneously.

---

### Decile Lift & Cumulative Gains Analysis (The 80/20 Breakthrough)

When prospect leads are ranked by their predicted probability using the Tuned LightGBM pipeline:

| Decile Tier | Total Leads | Actual Subscribers | Cumulative Leads % | Cumulative Responders % | Conversion Rate | Lift Factor |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decile 1** | 824 | 446 | 10.0% | **48.16%** | **54.13%** | **4.80x** |
| **Decile 2** | 824 | 225 | **20.0%** | **72.46%** | **27.31%** | **2.42x** |
| **Decile 3** | 824 | 96 | 30.0% | 82.83% | 11.65% | 1.03x |
| **Decile 4** | 824 | 57 | 40.0% | 88.99% | 6.92% | 0.61x |
| **Decile 5** | 824 | 35 | 50.0% | 92.76% | 4.25% | 0.38x |
| **Deciles 6–10** | 4,118 | 67 | 100.0% | 100.00% | 1.63% | 0.14x |

> [!IMPORTANT]
> **The Bottom Line**: By contacting only the **top 20% of prioritized leads (Deciles 1 & 2)**, the bank captures **72.46% of all term deposit subscriptions**. This eliminates **80% of outbound dialing volume** while driving agent conversion from **11.27% to 40.72%** (a **3.6x productivity gain**).

---

### Task 3: The 7-Pillar Strategic Action Playbook

1. **AI-Driven Decile Targeting**: Score all prospect leads in the CRM monthly. Telemarketers call *only* Deciles 1 and 2.
2. **Seasonal Reallocation**: Shift marketing headcount away from May and July (&lt;9% conv.) and allocate budget to March, September, October, and December (&gt;43% conv.).
3. **Cellular-First Mandate**: Require cellular mobile numbers as the primary contact channel (2.82x lift over landlines).
4. **The "3-Call Fatigue" Rule**: Enforce a strict CRM lock capping contacts at a maximum of 3 calls per customer per campaign.
5. **7-Day Warm Queue**: Re-contact previous campaign responders (`poutcome='success'`) within 3 to 7 days (65.1% conversion rate).
6. **Macroeconomic Counter-Cyclical Triggers**: Launch term deposit marketing bursts when 3-month Euribor rates fall below 1.5% and uncertainty rises.
7. **Persona-Specific Value Propositions**: Pitch capital security and dividend yield to seniors (>60 yrs); pitch automated digital savings habits to students (<25 yrs).

---

## Projected Financial & Operational Impact

| Metric | Current Baseline (Random Dialing) | AI-Targeted (Top 2 Deciles) | Business Change |
| :--- | :---: | :---: | :---: |
| **Calls Dialed** | 41,188 calls | 8,238 calls | **-80% dialing overhead & agent labor** |
| **Subscribers Acquired** | 4,640 deposits | ~3,362 deposits | **72.4% total deposit sales retained** |
| **Agent Conversion Rate** | 11.27% | **40.72%** | **+3.6x agent productivity multiplier** |
| **Cost per Acquired Customer** | $100 Index | **$27.70 Index** | **~72.3% reduction in Customer Acquisition Cost (CAC)** |

---

## How to Run & View Deliverables

### 1. View Interactive Web Presentation & Decision Simulator
Open [index.html](file:///c:/Users/Admin/Downloads/PRCP-1000-ProtugeseBank/index.html) in any modern web browser:
```bash
# Double-click index.html or open via command line
start index.html
```

### 2. View or Execute the Jupyter Notebook
```bash
# Launch JupyterLab / Jupyter Notebook
jupyter lab Bank_Marketing_Term_Deposit_Prediction.ipynb
```
The notebook is pre-executed with all cell outputs, plots, statistical tables, and model performance metrics embedded.
