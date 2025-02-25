import pandas as pd
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv('churn.csv')

#Pipeline
def wow_predict(dataframe):
  # Yeni özellik oluşturma
  dataframe['density_timestamp'] = dataframe['total_timestamps'] / dataframe['Average_Playing_density']

  # Outlier Handling
  dataframe.loc[dataframe['total_timestamps'] < 3, 'Binned_Timestamps'] = 0
  dataframe.loc[(dataframe['total_timestamps'] >= 3) & (dataframe['total_timestamps'] < 28), 'Binned_Timestamps'] = 1
  dataframe.loc[(dataframe['total_timestamps'] >= 28) & (dataframe['total_timestamps'] < 43), 'Binned_Timestamps'] = 2
  dataframe.loc[(dataframe['total_timestamps'] >= 43) & (dataframe['total_timestamps'] < 128), 'Binned_Timestamps'] = 3
  dataframe.loc[(dataframe['total_timestamps'] >= 128), 'Binned_Timestamps'] = 4

  dataframe['Binned_Timestamps'] = dataframe['Binned_Timestamps'].astype(int)

  dataframe.loc[dataframe['max_level'] < 3, 'Binned_Level'] = 0
  dataframe.loc[(dataframe['max_level'] >= 3) & (dataframe['max_level'] < 14), 'Binned_Level'] = 1
  dataframe.loc[(dataframe['max_level'] >= 14) & (dataframe['max_level'] < 18), 'Binned_Level'] = 2
  dataframe.loc[(dataframe['max_level'] >= 18) & (dataframe['max_level'] < 60), 'Binned_Level'] = 3
  dataframe.loc[(dataframe['max_level'] >= 60) & (dataframe['max_level'] < 70), 'Binned_Level'] = 4
  dataframe.loc[(dataframe['max_level'] >= 70), 'Binned_Level'] = 5

  dataframe['Binned_Level'] = dataframe['Binned_Level'].astype(int)

  dataframe.loc[dataframe['unique_days'] < 3, 'Binned_Unique_Days'] = 0
  dataframe.loc[(dataframe['unique_days'] >= 3) & (dataframe['unique_days'] < 11), 'Binned_Unique_Days'] = 1
  dataframe.loc[(dataframe['unique_days'] >= 11) & (dataframe['unique_days'] < 25), 'Binned_Unique_Days'] = 2
  dataframe.loc[(dataframe['unique_days'] >= 25) & (dataframe['unique_days'] < 40), 'Binned_Unique_Days'] = 3
  dataframe.loc[(dataframe['unique_days'] >= 40) & (dataframe['unique_days'] < 105), 'Binned_Unique_Days'] = 4
  dataframe.loc[(dataframe['unique_days'] >= 105), 'Binned_Unique_Days'] = 5

  dataframe['Binned_Unique_Days'] = dataframe['Binned_Unique_Days'].astype(int)

  dataframe.loc[dataframe['Average_Hour'] < 0.33, 'Binned_Average_Hour'] = 0
  dataframe.loc[(dataframe['Average_Hour'] >= 0.33) & (dataframe['Average_Hour'] < 0.43), 'Binned_Average_Hour'] = 1
  dataframe.loc[(dataframe['Average_Hour'] >= 0.43) & (dataframe['Average_Hour'] < 1.0), 'Binned_Average_Hour'] = 2
  dataframe.loc[(dataframe['Average_Hour'] >= 1) & (dataframe['Average_Hour'] < 2.6), 'Binned_Average_Hour'] = 3
  dataframe.loc[(dataframe['Average_Hour'] >= 2.6) & (dataframe['Average_Hour'] < 6.0), 'Binned_Average_Hour'] = 4
  dataframe.loc[(dataframe['Average_Hour'] >= 5.0), 'Binned_Average_Hour'] = 5

  dataframe['Binned_Average_Hour'] = dataframe['Binned_Average_Hour'].astype(int)

  X = dataframe[['density_timestamp', 'Binned_Timestamps', 'Binned_Level', 'Binned_Unique_Days', 'Binned_Average_Hour']]

  if 'Playing_after_6_months'  in dataframe.columns:
    y = dataframe['Playing_after_6_months']
    return X,y
  else:
    return X


X,y = wow_predict(df)


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)

model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.2, max_depth=3)

model.fit(x_train,y_train)


def feature(Total_timestamp,Average_Playing_density,max_level,unique_days,Average_hour):
  # Kullanıcı girdilerini DataFrame'e dönüştürme
  df = pd.DataFrame({
  'total_timestamps': [Total_timestamp],
  'Average_Playing_density': [Average_Playing_density],
  'max_level': [max_level],
  'unique_days': [unique_days],
  'Average_Hour': [Average_hour]
})
  return df


# Streamlit Titles
st.title('CHURN PREDICTION of WoW')
st.header('Features', divider='rainbow')

# User Inputs
st.subheader('Total TimeStamp')
Total_timestamp = st.slider("The total number of timestamps recorded for the player's activities in the game.", 0, 17000, 1)

st.subheader('Average_Playing_Density')
Average_Playing_density = st.slider("Average Playing Density:", 0.0, 1.0, 0.1)

st.subheader('Max_level')
max_level = st.slider("Max Level:", 0, 80, 1)

st.subheader('Unique_Days')
unique_days = st.slider("Unique days:", 0, 200, 1)

st.subheader('Average_Hour')
Average_hour = st.slider("Average hours:", 0.0, 10.0, 0.1)

# Data Processing and Prediction
X = wow_predict(feature(Total_timestamp,Average_Playing_density,max_level,unique_days,Average_hour))
y = model.predict(X)

# Show Results
if y[0] == 0:
  st.write('YOUR PLAYER CHURNED')
else:
  st.write('YOUR PLAYER STAYING')

st.markdown('This app is created by Cagan Demir')
