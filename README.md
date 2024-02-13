# Reducing Maize Post-Harvest Losses in Sub-Saharan Africa: A Data-Driven Approach
![Maize farm 1](https://github.com/Kelvin-Rotich824/Phase_5_Capstone_Project/assets/142001883/dfbe22aa-1499-4e86-b95b-872f5e73309e)

## Problem Statement

Post-harvest losses of Maize in Sub-Saharan Africa are substantial across the supply chain, ranging from 30% to 50% of total yield. Traditional monitoring and mitigation methods require enhancement to be more effective and resource-efficient.

## Introduction

Maize is widely considered to be a vital staple food across Sub-Saharan Africa, but post-harvest losses do pose a major threat to food security and economic stability in the region. This project background aims to explore the magnitude of the issue, key contributing factors, and measures put in place as well as efforts underway to address it.

### Project Overview

Approximately 13.5% of all maize harvested in Sub-Saharan Africa is lost post-harvest. This project aims to tackle the significant challenge of post-harvest losses incurred by maize farmers throughout the region by utilizing a data-driven approach. By leveraging various data science techniques such as time-series modeling, anomaly detection, and inferential regression, we seek to gain insight into the factors contributing to post-harvest losses and develop strategies to mitigate them

## Business Understanding

Stakeholders including farmers, policymakers, workers in the agricultural sector, food security organizations, and private sector stakeholders are only a few of the people who stand to benefit from this project. By reducing post-harvest losses, we aim to reduce environmental impact, increase the availability of foodstuffs, stabilize markets, and boost the incomes of small, medium, and large-scale farmers.

## 1. Data Transformation and Preparation
Our data was sourced from the Food and Agriculture Organisation Statistics (FAOSTAT) database which has global statistics on multiple aspects of agriculture and the African Post-Harvest Losses Information System(APHLIS) which has data on cereal postharvest loss in Africa but we narrowed our search down to Sub-Saharan countries. The first order of business was to transform our collected data and store it in one data frame making it more user-friendly for analysis and modeling. Additionally, we worked on data cleaning, handling missing values, handling duplicates, data reshaping, and other processes to ensure that we have a clean, structured, and suitable format for analysis and modeling

## 2. Exploratory Data Analysis

Secondly, we embarked on an exploration of the different features within the dataset to gain a better understanding. We utilized data visualization to uncover trends and patterns as well as Feature Engineering to create new features from existing ones and perform One-Hot Encoding on categorical variables that we will require for analysis.

### 2.1 Univariate Analysis

The exploration begins with univariate analysis, where histograms are created for selected columns to observe their distributions. All histograms, except for temperature change, display positive skewness, indicating that their means are lower than their medians, and they do not follow a normal distribution. Below is just one example of the various histograms. 
![Hist dry weight loss](https://github.com/Kelvin-Rotich824/Phase_5_Capstone_Project/assets/142001883/9a75e8bd-f795-4b27-b0a3-aefc521a3032)

### 2.2 Bivariate Analysis

Moving on to bivariate analysis, scatter plots are generated to compare numerical features with the target variable, revealing no linear relationship.
![Bivariate Analysis](https://github.com/Kelvin-Rotich824/Phase_5_Capstone_Project/assets/142001883/261b0583-7d59-40c5-be54-ac8f1e29531d)

### 2.3 Multivariate Analysis

As we can see, there is a strong positive correlation between import value and import quantity, cropland nitrogen per unit area, and cropland potassium per unit area production, and area harvested. The others correlate 0.65.
![Multivariate analysis](https://github.com/Kelvin-Rotich824/Phase_5_Capstone_Project/assets/142001883/d9b2d08d-4d09-4f1d-8d6d-8b63e015ed99)

### 2.4 Time Series Analysis

This is where the data is resampled to daily frequency and null values are interpolated. A comparison is made between the original and resampled datasets, confirming similar characteristics.
![Line plot 2 TSA](https://github.com/Kelvin-Rotich824/Phase_5_Capstone_Project/assets/142001883/04565c35-ca7d-4862-b696-54eec9089353)
## 3. Modeling

The goal is to build a regression model that can accurately predict crop dry weight loss based on various predictor variables like climate, yields, production, etc. Multiple regression algorithms such as Linear, Ridge, and Lasso, DecisionTree, and RandomForest are evaluated one by one to determine the best approach. Random Forest Regression achieves the lowest error scores out of all the models, which makes it the best approach for predicting crop dry weight loss. The base model was further improved after employing Optuna as a hyperparameter tuning technique, allowing for the creation of more optimized RandomForest and XGBoost models with the former still having the best performance.
![ARIMA 2](https://github.com/Kelvin-Rotich824/Phase_5_Capstone_Project/assets/142001883/f4c84041-db3c-46f5-9a2b-956c1db512c9)
## 4. Conclusion

- Regional Trends: Western African countries experience the highest maize post-harvest losses, followed by Southern Africa, which also boasts the highest maize production quantity among the regions.
- Temporal Patterns: Maize post-harvest losses exhibited a rising trend from 2000 up to around 2010, followed by a subsequent decline.
- Regression Modeling: A regression model was developed to uncover insights into the relationship between various features and maize loss, providing valuable analytical depth.
- Anomaly Detection: Anomalies detected within the data may have contributed to the observed maize losses, highlighting areas for further investigation and intervention.
- Time Series Prediction: A time series model was successfully constructed, enabling the prediction of maize loss across sub-Saharan Africa over multiple years.
#### This project holds the potential to make a significant impact on food security and economic development in Sub-Saharan Africa. By harnessing the power of data science and machine learning, we aim to reduce post-harvest losses, increase food availability and the scale of grain farming, and contribute to sustainable development in the region.

### Project Team 
- Kelvin Rotich (https://github.com/Kelvin-Rotich824)
- Shadrack Muchai (https://github.com/ShadrackMu)
- Ruth Nanjala (https://github.com/RuthNanjala)
- MaryImmaculate Kariuki (https://github.com/mumbikariuki)
- Dennis Kimani (https://github.com/dennismathu)
