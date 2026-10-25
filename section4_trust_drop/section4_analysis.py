#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 10:17:09 2026

@author: timo
"""

#============================= DATA PREPARATION ==============================

import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import seaborn as sns

# import dataset

os.chdir('/Users/timo/Desktop/MSBA Related/Coursework/AN6002 ANALYTICS AND MACHINE LEARNING IN BUSINESS/Group Project')
df = pd.read_csv('dataset2.csv')

# handle null values
    
""" 
99 rows of blanks for columns where error_detected == 'no' : 
    ['error_detected_by', 'error_type', 'error_severity', 'error_domain', 'user_emotional_reaction','user_reported_consequence',
    'trust_recovered','recovery_trigger']
"""                              
                              
# impute na values
df = df.fillna('na')
  
# list all columns of significance
columns = ['age',
'gender',
'education_level',
'country',
'occupation',
'tech_literacy_score',
'prior_ai_experience',
'ai_tools_used_before',
'trust_disposition_score',
'risk_tolerance_score',
'critical_thinking_score',
'anxiety_level_baseline',
'openness_to_technology',
'past_human_betrayal_incidents',
'ai_tool_category',
'task_type',
'task_importance',
'interaction_duration_seconds',
'error_detected_by',
'error_type',
'error_severity',
'error_domain',
'user_emotional_reaction',
'user_reported_consequence',
'media_coverage_of_ai_errors_that_week',
'personal_stress_level_at_time',
'regulatory_news_exposure']


# ================== First-pass exploration  =============================

# first-pass exploration of error impact on trust

sns.catplot(data = df, y = "trust_drop_magnitude", x = "error_detected", kind = "box", color = "skyblue")
sns.stripplot(data = df, y = "trust_drop_magnitude", x = "error_detected", size=4, jitter=True)
plt.title("Impact of Errors on Trust Loss")
plt.show();

# obviously error_detected will cause loss in trust while no error will not, so for the analysis of trust loss factors,
# we only consider cases where an error is already detected
df_error = df[df["error_detected"]=="yes"]

# ================== Trying to analyze significance of categorical variables  =============================

experience_mapping = {"low": "moderate experience", 
                      "medium": "moderate experience", 
                      "high": "high experience", 
                      "none": "no experience"}

df_error["ai_experience_level"] = df_error["prior_ai_experience"].map(experience_mapping).astype("category")

education_mapping = {
    'High School': "no degree",
    'Associate': "no degree",
    'Bachelor': "degree",
    'Master': "degree",
    'PhD': "degree"
}

df_error['degree'] = df_error['education_level'].map(education_mapping)

cat_columns = ['gender',
'education_level',
'country',
'occupation',
'ai_tools_used_before',
'ai_tool_category',
'task_type',
'task_importance',
'error_detected_by',
'error_type',
'error_severity',
'error_domain',
'user_emotional_reaction',
'user_reported_consequence',
'media_coverage_of_ai_errors_that_week',
'regulatory_news_exposure',
'ai_experience_level',
'degree']

# visualize all cat columns vs trust drop magnitude
for column in cat_columns:
    sns.catplot(data =df_error, y = "trust_drop_magnitude", x = column, kind = "box", color = "skyblue")
    sns.stripplot(data=df_error, y = "trust_drop_magnitude", x = column, size=4, jitter=True)
    plt.show();

# ANOVA test for significant categorical variables   
from scipy import stats
results = []
for col in cat_columns:
    groups = [group["trust_drop_magnitude"].values for _, group in df_error.groupby(col)]

    # Only test if there are at least 2 non-empty groups
    if len(groups) >= 2:
        f_stat, p_val = stats.f_oneway(*groups)
        results.append(
            {
                "Feature": col,
                "ANOVA_F_Stat": f_stat,
                "ANOVA_p_value": p_val,
            }
        )

# Sort features by lowest p-value (most statistically significant)
results_df = pd.DataFrame(results).sort_values("ANOVA_p_value")
print(results_df)

# plotting ai_experience_level (p = 0.016)    
sns.catplot(data =df_error, 
            y = "trust_drop_magnitude", x = "ai_experience_level", 
            kind = "box", 
            order = ["no experience", "moderate experience", "high experience"],
            color="skyblue")
sns.stripplot(data=df_error, y = "trust_drop_magnitude", x = "ai_experience_level", size=4, jitter=True)
plt.ylim(0, 4) 
plt.show();

# ================== Trying to analyze significance of numeric variables  =============================

num_columns = ['age',
'tech_literacy_score',
'trust_disposition_score',
'risk_tolerance_score',
'critical_thinking_score',
'anxiety_level_baseline',
'openness_to_technology',
'past_human_betrayal_incidents',
'interaction_duration_seconds',
'error_domain',
'user_emotional_reaction',
'user_reported_consequence',
'media_coverage_of_ai_errors_that_week',
'personal_stress_level_at_time',
'regulatory_news_exposure']


import statsmodels.api as sm

# linear regression model to check for variable correlation/significance

# Prepare X and y
X = df_error.select_dtypes(include=["number"]).drop(
    columns=["trust_drop_magnitude"], errors="ignore")
y = df_error.loc[X.index,'trust_drop_magnitude']

# Add constant for intercept
X_const = sm.add_constant(X)

# Fit OLS model
model = sm.OLS(y, X_const).fit()

# Extract coefficients, t-stats, and p-values into a tidy DataFrame
ols_summary = pd.DataFrame(
    {
        'Coefficient': model.params,
        'Std_Error': model.bse,
        't_statistic': model.tvalues,
        'p_value': model.pvalues,
    }
).drop(
    'const', errors='ignore'
)  # Drop intercept row for feature comparison

# Sort by lowest p-value (or highest absolute t-stat)
ols_summary['abs_t'] = ols_summary['t_statistic'].abs()
ols_summary = ols_summary.sort_values(by='p_value')
print(ols_summary[['Coefficient', 'p_value']])

# check corr coefficients
corr_matrix = df.select_dtypes(include='number').corr()
corr_matrix = corr_matrix.replace(1,np.nan)
print(corr_matrix["trust_drop_magnitude"].sort_values())

# relationship between trust_loss_permanent and trust_drop_magnitude
sns.regplot(data = df_error, x = "trust_drop_magnitude", y= "total_trust_loss_permanent", lowess = True)


# ===================== TRAIN TEST SPLIT FOR MACHINE LEARNING MODEL TRAINING ========================

from sklearn.model_selection import train_test_split

# create target and features dataset
y = df_error["trust_drop_magnitude"].astype(float)
X = df_error[columns]

X_encoded = pd.get_dummies(X, drop_first=True, dtype = float)

# train test-split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size = 0.2, random_state = 42)
print(f"Training shapes: X={X_train.shape}, y={y_train.shape}")
print(f"Testing shapes:  X={X_test.shape}, y={y_test.shape}")

# ==================== Random Forest Regression ==============================

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# instantiate and fit RF model
rf_model = RandomForestRegressor(n_estimators=200,
                                 max_depth=6,
                                 min_samples_split=5,
                                 random_state=42)

rf_model.fit(X_train,y_train)

# predict on train/test sets
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# check model fit
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_rmse = np.sqrt(mean_squared_error(y_train,y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test,y_test_pred))
print("====== RANDOM FOREST PERFORMANCE ======")
print(f"Train R²: {train_r2:.4f} | Train RMSE: {train_rmse:.4f}")
print(f"Test R²:  {test_r2:.4f}  | Test RMSE:  {test_rmse:.4f}")

#===================== Ridge/Lasso Regression ===============================

from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# Standardize features and fit Ridge regression with Cross-Validation
ridge_pipe = make_pipeline(
    StandardScaler(),
    RidgeCV(alphas=np.logspace(-3, 3, 100), cv=5)
)
ridge_pipe.fit(X_train, y_train)

print(f"Ridge Train R²: {ridge_pipe.score(X_train, y_train):.4f}")
print(f"Ridge Test R²:  {ridge_pipe.score(X_test, y_test):.4f}")

lasso_pipe = make_pipeline(
    StandardScaler(),
    LassoCV(alphas=np.logspace(-3, 3, 100), cv=5)
)
lasso_pipe.fit(X_train, y_train)

print(f"Lasso Train R²: {lasso_pipe.score(X_train, y_train):.4f}")
print(f"Lasso Test R²:  {lasso_pipe.score(X_test, y_test):.4f}")


#===================== Classification  ======================

# classify trust loss into high and low
threshold = df_error["trust_drop_magnitude"].median()
y_binary = (df_error["trust_drop_magnitude"] > threshold).astype(int)

# list all columns of significance
feature_cols = ['age',
'gender',
'education_level',
'country',
'occupation',
'tech_literacy_score',
'prior_ai_experience',
'ai_tools_used_before',
'trust_disposition_score',
'risk_tolerance_score',
'critical_thinking_score',
'anxiety_level_baseline',
'openness_to_technology',
'past_human_betrayal_incidents',
'ai_tool_category',
'task_type',
'task_importance',
'interaction_duration_seconds',
'error_detected_by',
'error_type',
'error_severity',
'error_domain',
'user_emotional_reaction',
'user_reported_consequence',
'media_coverage_of_ai_errors_that_week',
'personal_stress_level_at_time',
'regulatory_news_exposure']

#prepare X and y feature sets and train-test splits

X = df_error[feature_cols].copy()
X_encoded = pd.get_dummies(X, drop_first=True, dtype=float)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_binary, test_size=0.20, random_state=42, stratify=y_binary
)

print(f"Training shape: {X_train.shape}, Testing shape: {X_test.shape}")
print("Class Distribution in Test Set:", np.bincount(y_test))

# train and compare classification models

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# Initialize Classifiers
models = {
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, random_state=42)),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=5, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(max_depth=3, min_samples_leaf=5, random_state=42)
}

# Iterate, Train, and Evaluate
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
    
    auc = roc_auc_score(y_test, y_proba)
    results[name] = {"model": model, "auc": auc, "pred": y_pred}
    
    print(f"=== {name} ===")
    print(f"ROC-AUC Score: {auc:.4f}")
    print(classification_report(y_test, y_pred))
    print(f"Model Accuracy: {accuracy_score(y_test,y_pred)}")
    print("-" * 50)

# all 3 models perform extremely poorly

#===================== Clustering ======================

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# 1. Select clustering feature columns from dataset
cluster_cols = [
    'trust_drop_magnitude',
    'critical_thinking_score', 
    'openness_to_technology',
]

# Extract feature matrix
X_cluster = df_error[cluster_cols].copy()

# 2. Standardize features (Essential for distance-based clustering like K-Means)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

inertia = []
silhouette_scores = []
K_range = range(2, 10)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

# Plot diagnostic curves
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Elbow Curve
ax[0].plot(K_range, inertia, marker='o', color='navy', linewidth=2)
ax[0].set_title('Elbow Method (Inertia)', fontsize=12)
ax[0].set_xlabel('Number of Clusters (K)')
ax[0].set_ylabel('Inertia')
ax[0].axvline(x=3, color = 'orange')
ax[0].grid(True, linestyle='--', alpha=0.5)

# Silhouette Curve
ax[1].plot(K_range, silhouette_scores, marker='s', color='darkgreen', linewidth=2)
ax[1].set_title('Silhouette Score Analysis', fontsize=12)
ax[1].set_xlabel('Number of Clusters (K)')
ax[1].set_ylabel('Silhouette Score')
ax[1].axvline(x=3, color = 'orange')
ax[1].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Print scores to identify highest peak
for k, score in zip(K_range, silhouette_scores):
    print(f"K = {k}: Silhouette Score = {score:.4f}")

# Train KMeans with chosen K
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df_error['Trust_Persona_Cluster'] = kmeans.fit_predict(X_scaled)

# Calculate mean trait profiles for each persona
persona_profiles = df_error.groupby('Trust_Persona_Cluster')[cluster_cols].mean().round(2)

print("=== AI Trust Resilience Profiles ===")
print(persona_profiles.T)







