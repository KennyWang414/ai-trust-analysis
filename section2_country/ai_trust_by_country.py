# ------------------------------------------------------------
# Import Libraries
# ------------------------------------------------------------
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    RepeatedKFold,
    cross_validate,
    GridSearchCV
)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

### Regression models ###
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.neural_network import MLPRegressor

# ------------------------------------------------------------
# Importing data
# ------------------------------------------------------------
geo_df = gpd.read_file("world_by_iso_geo.json")
ai_trust_df = pd.read_csv('ai_trust_by_country.csv')
country_df1 = pd.read_csv('WB_WDI_WIDEF.csv')
country_df2 = pd.read_excel('HDR25_Statistical_Annex_HDI_Table_with_ISO3.xlsx',skiprows = 4)

df_list = [geo_df, ai_trust_df, country_df1, country_df2]

# ------------------------------------------------------------
# Cleaning Data
# ------------------------------------------------------------

### Trim column names ###

geo_df.columns = geo_df.columns.str.strip()
ai_trust_df.columns = ai_trust_df.columns.str.strip()
country_df1.columns = country_df1.columns.str.strip()
country_df2.columns = country_df2.columns.str.strip()

### Cleaning Geo DataFrame ###
geo_df.dropna(inplace = True)
geo_df.drop(columns=['id'], inplace=True)

geo_df["iso"] = (geo_df["iso"].str.strip().str.upper())

### Cleaning Labelled DataFrame ###
ai_trust_df = ai_trust_df.drop(columns=['region', 'economy_type', 'gdp_per_capita_usd','population' ])
ai_trust_df["iso3"] = (ai_trust_df["iso3"].str.strip().str.upper())

### Cleaning Unlabelled DataFrame 1 ###

# Filtering rows to only show listed indicators 
indicators = [
    "Age dependency ratio (% of working-age population)",
    "Access to electricity (% of population)",
    "GDP per capita (current US$)",
    "Government Effectiveness - Governance score (0-100)",
    "Individuals using the Internet (% of population)",
    "Population density (people per sq. km of land area)",
    "Population, total",
    "Secure Internet servers (per 1 million people)"
]
country_df1 = country_df1[country_df1["INDICATOR_LABEL"].isin(indicators)].reset_index(drop = True)

# Filtering columns to only show relevant columns
country_df1 = country_df1[
    [
        "REF_AREA",
        "REF_AREA_LABEL",
        "INDICATOR_LABEL",
        "2023"
    ]
]

# Pivoting the DataFrame to have indicators as columns and countries as rows
country_df1 = country_df1.pivot(index=["REF_AREA", "REF_AREA_LABEL"],columns="INDICATOR_LABEL",values="2023").reset_index()
country_df1["REF_AREA"] = (country_df1["REF_AREA"].str.strip().str.upper())

### Cleaning Unlabelled DataFrame 2 ###

# Filtering rows to only show countries
country_df2 = country_df2[country_df2['Unnamed: 0'].notna()]

# Filtering columns to only show relevant columns
country_df2 = country_df2[
    [
        "Unnamed: 1",
        "Unnamed: 2",
        "Human Development Index (HDI)",
        "Life expectancy at birth",
        "Expected years of schooling",
        "Mean years of schooling",
        "Gross national income (GNI) per capita"
    ]
]
# Cleaning column names and resetting index
country_df2 = country_df2.rename(columns={"Unnamed: 1": "iso3","Unnamed: 2": "Country"})
country_df2.drop(country_df2.index[0], inplace = True)
country_df2.reset_index(drop = True, inplace = True)

# Converting data types
country_df2[
    [
        'Human Development Index (HDI)',
        'Life expectancy at birth',
        'Expected years of schooling',
        'Mean years of schooling',
        'Gross national income (GNI) per capita'
        ]
    ] = country_df2[
    [
        'Human Development Index (HDI)',
        'Life expectancy at birth',
        'Expected years of schooling',
        'Mean years of schooling',
        'Gross national income (GNI) per capita'
        ]
    ].astype(float)
    
    
country_df2["iso3"] = (
    country_df2["iso3"]
    .str.strip()
    .str.upper()
)

### Check for missing values ###
for df in df_list:
    print(df.isnull().sum())


### Check Data Types ###
for df in df_list:
   print(df.dtypes)
    
# ------------------------------------------------------------
# Merging DataFrames to create labeled dataset
# ------------------------------------------------------------
labelled_df = pd.merge(ai_trust_df, country_df1, how = 'left', left_on = 'iso3', right_on = 'REF_AREA').drop(columns=['REF_AREA', 'REF_AREA_LABEL'])
labelled_df = pd.merge(labelled_df, country_df2, how = 'left', left_on = 'iso3', right_on = 'iso3').drop(columns=['Country'])

# Export clean labeled data to csv
labelled_df.to_csv("labeled_data_clean.csv", index = False)

### Merging DataFrames to create unlabeled dataset ###
unlabeled_df = pd.merge(country_df2, country_df1, how="left", left_on="iso3", right_on="REF_AREA").drop(columns=["REF_AREA", "REF_AREA_LABEL"])
unlabeled_df = unlabeled_df[~unlabeled_df["iso3"].isin(ai_trust_df["iso3"])]
unlabeled_df = unlabeled_df.reset_index(drop=True)

# Export clean unlabeled data to csv
unlabeled_df.to_csv("unlabeled_data_clean.csv", index = False)

# ------------------------------------------------------------
# Selecting best machine learning model
# ------------------------------------------------------------
### Predictor variables ###
features = [
    "Age dependency ratio (% of working-age population)",
    "Government Effectiveness - Governance score (0-100)",
    "Individuals using the Internet (% of population)",
    "Population density (people per sq. km of land area)",
    "Life expectancy at birth",
    "Mean years of schooling",
    "Gross national income (GNI) per capita"
]

X_train = labelled_df[features]
y_train = labelled_df["trust_ai_pct"]
X_test = unlabeled_df[features]

seed_num = 12345

### Cross-validation ###
regression_cv = RepeatedKFold(
    n_splits=5,
    n_repeats=20,
    random_state= seed_num
)

### Establishing a dictionary of Regression Models to evaluate ###

regression_models = {

    "Linear Regression":
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", LinearRegression())
        ]),

    "KNN Regressor":
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", KNeighborsRegressor(
                n_neighbors=5
            ))
        ]),

    "Linear SVR":
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", SVR(
                kernel="linear"
            ))
        ]),

    "RBF SVR":
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", SVR(
                kernel="rbf"
            ))
        ]),

    "Decision Tree":
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DecisionTreeRegressor(
                random_state=seed_num
            ))
        ]),

    "Random Forest":
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=500,
                min_samples_leaf=2,
                random_state=seed_num
            ))
        ]),

    "Gradient Boosting":
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingRegressor(
                random_state=seed_num
            ))
        ]),

    "AdaBoost":
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", AdaBoostRegressor(
                random_state=seed_num
            ))
        ]),

    "MLP":
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", MLPRegressor(
                hidden_layer_sizes=(50,),
                max_iter=5000,
                random_state=seed_num
            ))
        ])
}

### Running Machine learning models ###

regression_results = []

for name, model in regression_models.items():

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=regression_cv,
        scoring=[
            "neg_mean_absolute_error",
            "neg_mean_squared_error",
            "r2"
        ],
    )

    mae = (-scores["test_neg_mean_absolute_error"]).mean()
    rmse = np.sqrt(-scores["test_neg_mean_squared_error"]).mean()
    r2 = scores["test_r2"].mean()
    regression_results.append({"Model": name,"MAE": mae,"RMSE": rmse,"R2": r2})

regression_results = pd.DataFrame(regression_results)
regression_results = regression_results.sort_values("RMSE")

print("\nREGRESSION RESULTS")
print(regression_results)

# Best regression model based on RMSE is K-Nearest Neighbors.

# ------------------------------------------------------------
# Tuning the machine learning model and testing
# ------------------------------------------------------------

# Tuning K-Nearest Neighbors Regression Model

# KNN pipeline
knn_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
    ("model", KNeighborsRegressor())
])

# Values of k to test
knn_param_grid = {
    "model__n_neighbors": list(range(2, 16)),
    "model__weights": [
        "uniform",
        "distance"
    ]
}

# Grid search
knn_grid = GridSearchCV(
    estimator=knn_pipeline,
    param_grid=knn_param_grid,
    scoring="neg_root_mean_squared_error",
    cv=regression_cv,
)

# Fit model with different k values
knn_grid.fit(X_train,y_train)

print("Best k:",knn_grid.best_params_["model__n_neighbors"])
print("Best weighting:",knn_grid.best_params_["model__weights"])
print("Best RMSE:",round(-knn_grid.best_score_, 4))

best_knn = knn_grid.best_estimator_
y_test = best_knn.predict(X_test)

# Add predictions to unlabeled dataframe
unlabeled_df["Predicted Trust AI (%)"] = (y_test.round(2))

# View results
print(unlabeled_df[["iso3", "Country", "Predicted Trust AI (%)"]])

# ------------------------------------------------------------
# Identifying most important predictor variables
# ------------------------------------------------------------

### Regression Model Feature Importance ###

regression_importance = permutation_importance(
    best_knn,
    X_train,
    y_train,
    scoring="neg_root_mean_squared_error",
    n_repeats=30,
    random_state=seed_num
)

regression_importance_df = pd.DataFrame({
    "Predictor": features,
    "Importance": regression_importance.importances_mean,
    "Importance SD": regression_importance.importances_std
})

regression_importance_df = (regression_importance_df.sort_values("Importance",ascending=False).reset_index(drop=True))

print("\nMOST INFLUENTIAL PREDICTORS FOR TRUST AI %")
print(regression_importance_df)


# ------------------------------------------------------------
# Plotting importance charts
# ------------------------------------------------------------

### Plotting Regression Importance Chart ###
top_regression = (regression_importance_df.head(10).sort_values("Importance")
                  )
plt.figure(figsize=(9, 6))
plt.barh(top_regression["Predictor"],top_regression["Importance"])
plt.xlabel("Permutation Importance")
plt.ylabel("Predictor")
plt.title("Most Influential Predictors of AI Trust")
plt.tight_layout()
plt.show()

# ------------------------------------------------------------
# Combining Results and Labelled dataframes for use in Tableau
# ------------------------------------------------------------

# Rename columns in labelled data to match test results
labelled_df = labelled_df.rename(columns={"country": "Country","trust_ai_pct": "Trust AI (%)"})

# Rename predicted columns in test results
unlabeled_df = unlabeled_df.rename(columns={"Predicted Trust AI (%)": "Trust AI (%)"})

# Add column showing data source
labelled_df["Dataset"] = "Labelled"
unlabeled_df["Dataset"] = "Unlabeled"

# Select the common columns
common_columns = [
    "iso3",
    "Country",
    "Age dependency ratio (% of working-age population)",
    "Government Effectiveness - Governance score (0-100)",
    "Individuals using the Internet (% of population)",
    "Population density (people per sq. km of land area)",  
    "Life expectancy at birth",
    "Mean years of schooling",
    "Gross national income (GNI) per capita",
    "Trust AI (%)",
    "Dataset"
]

# Keep only matching columns
labelled_df = labelled_df[common_columns]
unlabeled_df = unlabeled_df[common_columns]

# Combine the datasets vertically
final_df = pd.concat([labelled_df, unlabeled_df],ignore_index=True)

# Check result
print(final_df.shape)
print(final_df.head())

# Export file
final_df.to_csv("final.csv",index = False)

# Export to geojson
final_geodf = pd.merge(final_df, geo_df, how = 'left', left_on = 'iso3', right_on = 'iso').drop(columns=['name', 'iso'])
gdf = gpd.GeoDataFrame(final_geodf, geometry='geometry', crs="EPSG:4326")
gdf.to_file("final_geodata.geojson", driver="GeoJSON")
