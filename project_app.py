import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.svm import OneClassSVM
from prophet import Prophet
import plotly

st.title("Maize Post-Harvest Loss in Sub-Saharan Africa")

# Data loading and preprocessing (replace with your data and preprocessing steps)
# Loading the datasets
df1 = pd.read_csv("CPI.csv")
df2= pd.read_csv("Dry weight loss.csv")
df3 = pd.read_csv("GPI.csv")
df4 = pd.read_csv("Import.csv")
df5 = pd.read_csv("Nutrients use.csv")
df6 = pd.read_csv("Pesticide use.csv")
df7 = pd.read_csv("temperature change.csv")
df8 = pd.read_csv("Yield.csv")

# Merging df1 with df3
data = df1.merge(df3,left_on=['country','year'],right_on=['country','year'], how="inner")
data = data.drop(["Unnamed: 0_x","Unnamed: 0_y"],axis = 1)

# Merging df1 with df4
data = data.merge(df4,left_on=['country','year'],right_on=['country','year'], how="inner")
data = data.drop(["Unnamed: 0"],axis = 1)

# Merging df1 with df5
data = data.merge(df5,left_on=['country','year'],right_on=['country','year'], how="inner")
data = data.drop(["Unnamed: 0.1", "Unnamed: 0"],axis = 1)

# Merging df1 with df6
data = data.merge(df6,left_on=['country','year'],right_on=['country','year'], how="inner")
data= data.drop(["Unnamed: 0"],axis = 1)

# Merging df1 with df7
data = data.merge(df7,left_on=['country','year'],right_on=['country','year'], how="inner")
data= data.drop(["Unnamed: 0"],axis = 1)

# Merging df1 with df8
data = data.merge(df8,left_on=['country','year'],right_on=['country','year'], how="inner")
data= data.drop(["Unnamed: 0"],axis = 1)

# Merging df1 with df2 which has the target variable
data = data.merge(df2,left_on=['country','year'],right_on=['country','year'], how="inner")
data= data.drop(["Unnamed: 0","Unnamed: 0.1","Unnamed: 0.1.1","Unnamed: 0.2"],axis = 1)

# Creating the 'Region' column

# Dividing the countries into lists of their positions in Africa
central = ["Burundi", 
           "Cameroon", 
           "Central African Republic", 
           "Chad", 
           "Congo", 
           "Democratic Republic of the Congo",
           "Equatorial Guinea",  
           "Gabon"]
east = ["Eritrea", 
        "Ethiopia", 
        "Kenya", 
        "Madagascar", 
        "Rwanda", 
        "Somalia",
        "Sudan",
        "Uganda",
        "United Republic of Tanzania"]
south = ["Angola", 
        "Botswana", 
        "Eswatini", 
        "Lesotho", 
        "Malawi", 
        "Mozambique",
        "Namibia",
        "South Africa",
        "Zambia",
        "Zimbabwe"]
west = ["Benin", 
        "Burkina Faso", 
        "Côte d'Ivoire", 
        "Gambia", 
        "Ghana", 
        "Guinea", 
        "Guinea-Bissau",  
        "Liberia",
        "Mali",
        "Niger",
        "Nigeria",
        "Senegal",
        "Sierra Leone",
        "Togo"]

north = ["Mauritania"]

# Creating an empty 'Region' column in the DataFrame
data['Region'] = ""

# Iterating over each row in the DataFrame and assign the region based on the country
for index, row in data.iterrows():
    if row['country'] in central:
        data.at[index, 'Region'] = "Central Africa"
    elif row['country'] in east:
        data.at[index, 'Region'] = "Eastern Africa"
    elif row['country'] in south:
        data.at[index, 'Region'] = "Southern Africa"
    elif row['country'] in west:
        data.at[index, 'Region'] = "Western Africa"
    elif row['country'] in north:
        data.at[index, 'Region'] = "Northern Africa"
    else:
        data.at[index, 'Region'] = "Other"

# Creating the 'years_to_2050' column
data['years_to_2050'] = 2050 - data['year']

# Reordering the columns
column_order = ['country', 'Region', 'year', 'cpi', 'maize GPI', 'Import Quantity', 'Import Value', 'Cropland nitrogen per unit area', 'Cropland potassium per unit area', 'pesticide use per area of cropland', 'temperature change', 'Area harvested', 'Production', 'Yield', 'years_to_2050', 'dry weight loss']

# Reordering the DataFrame columns
data = data[column_order]

# Convert 'year' column to integer type
data["year"] = data["year"].astype(int)
data.info()

# Selecting the necessary columns
X_fourth = data[['cpi', 'Import Value','Cropland nitrogen per unit area', 'pesticide use per area of cropland', 'Production', 'Yield']]
# Converting their values to the fourth root
for column in X_fourth.columns:
    X_fourth[column] = X_fourth[column] ** 0.25

# Selecting the column
X_cat = data['Region']
# OneHotEncoding 
X_cat = pd.get_dummies(X_cat, columns=['Region'], drop_first=True, dtype=int)
# Combining the two dataframes
X_transformed = pd.concat([X_fourth, X_cat], axis=1)

# combining to our transformed dataset
X_transformed = pd.concat([X_transformed, data['years_to_2050']], axis=1)
X_scaled = StandardScaler().fit_transform(X_transformed)

# Stating the target variable
y = data['dry weight loss']

# VotingRegressor Model Training and Evaluation
model_1 = joblib.load('regression_model.pkl')
model_1.fit(X_scaled, y)
y_pred1 = model_1.predict(X_scaled)
residuals = y - y_pred1

predictions = pd.DataFrame(y_pred1, index=X_transformed.index)
predictions = predictions.rename(columns={0:'Predicted dry weight loss'})
df = pd.concat([data, predictions], axis=1)

st.header("Inferential Regression Model")
st.subheader("Regression Results")
st.dataframe(df)
st.subheader("Evaluation Metrics")
st.write(f"R-squared: {r2_score(y, y_pred1)}")
r2 = r2_score(y, y_pred1)
n = len(y)
p = len(X_scaled.T)
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p)
st.write(f"Adjusted R-Squared: {adj_r2}")
st.write(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred1))}")
st.write(f"MAE: {mean_absolute_error(y, y_pred1)}")

# Residuals Plot
st.subheader("Residuals Plot")
fig, ax = plt.subplots()
ax.scatter(y_pred1, residuals)
ax.set_xlabel("Fitted Values")
ax.set_ylabel("Residuals")
st.pyplot(fig)

# OneClass Model Training and Evaluation
svm = joblib.load("anomaly_model.pkl")
svm.fit(X_scaled)
y_pred_outlier = svm.predict(X_scaled)

st.header("Anomaly Detection")
st.subheader("Anomaly Detection Results")
st.write(f"Number of Inliers: {len(y_pred_outlier[y_pred_outlier == 1])}")
st.write(f"Number of Outliers: {len(y_pred_outlier[y_pred_outlier == -1])}")
st.subheader("Evaluation Metrics")
st.write(f"RMSE: {np.sqrt(mean_squared_error(y, y_pred_outlier))}")
st.write(f"MAE: {mean_absolute_error(y, y_pred_outlier)}")

# Anomaly Score Distribution
st.subheader("Anomaly Score Distribution")
fig, ax = plt.subplots()
ax.hist(svm.decision_function(X_scaled), bins="auto")
ax.set_xlabel("Values")
ax.set_ylabel("Frequency")
# Plotting the histogram in Streamlit
st.pyplot(fig)

# Prophet Forecast

# Load your Prophet model
ts_model = joblib.load('ts_model.pkl')

# Get starting and ending dates from user input
start_date = st.date_input(
    "Select a starting date:",
    value=pd.to_datetime("2022-01-01"),
    min_value=pd.to_datetime("2022-01-01"),
    max_value=pd.to_datetime("2050-12-31")
)

end_date = st.date_input(
    "Select an ending date:",
    value=pd.to_datetime("2023-01-01"),
    min_value=start_date,  # Ensure end date is after start date
    max_value=pd.to_datetime("2050-12-31")
)

# Define a function to make predictions
def make_prediction(date):
    # Copying the dataframe.
    ts = data.copy()
    # Setting 'year' as the index
    ts['year'] = pd.to_datetime(ts['year'], format='%Y')
    # Dropping the null values
    ts = ts.dropna(subset=['dry weight loss'])
    # Grouping the dataframe
    ts = ts.groupby('year').aggregate({'dry weight loss':'mean'})
    # Resampling the data to daily
    ts = ts.resample('D').asfreq()
    # Filling the null values
    ts = ts.interpolate(method='linear', axis=0, limit_direction='forward')
    # Resetting the index and renaming the columns
    ts_prophet = ts.reset_index()
    ts_prophet = ts_prophet.rename(columns={'year': 'ds', 'dry weight loss': 'y'})
    # Fit the model to your data
    ts_model.fit(ts_prophet)
    future_data = ts_model.make_future_dataframe(periods=1, freq="D", include_history=True)
    forecast = ts_model.predict(future_data)
    forecast = pd.DataFrame({
        "ds": forecast["ds"],
        "yhat": forecast["yhat"],
        "yhat_lower": forecast["yhat_lower"],
        "yhat_upper": forecast["yhat_upper"]})
    return forecast

# Streamlit interface
st.header("Time Series Forecast")

selected_date = st.date_input(
    "Select a date for prediction:",
    value = pd.to_datetime("2022-01-01"),
    min_value = pd.to_datetime("2022-01-01"),
    max_value = pd.to_datetime("2050-12-31"),
)

if selected_date:
    if st.button("Predict"):
        forecast = make_prediction(selected_date)
        st.write(f"Predicted dry weight loss for {selected_date}: {forecast['yhat'].iloc[0]} tonnes")
else:
    st.write("Please select a date for prediction")
