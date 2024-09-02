import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import streamlit as st 
from sklearn.preprocessing import MinMaxScaler
import keras
import tensorflow as tf
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential


start_date = '2010-01-01'
end_date = '2019-12-31'
st.title('Stock Trend Prediction')
user_input=st.text_input('Enter Stock Ticker', 'AAPL')
df = yf.download(user_input, start=start_date, end=end_date)

#Describing data
st.subheader('Data from 2010 - 2019')
st.write(df.describe())

st.subheader('Closing Price vs Time Chart')
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close,'b')
st.pyplot(fig) 

#Calculating and plotting moving averages
st.subheader('Closing Price vs Time Chart with 100MA')
ma100=df.Close.rolling(100).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close,'b')
plt.plot(ma100,'r')
st.pyplot(fig) 

st.subheader('Closing Price vs Time Chart with 100MA & 200MA')
ma200=df.Close.rolling(200).mean()
fig=plt.figure(figsize=(12,6))
plt.plot(df.Close,'b')
plt.plot(ma100,'r')
plt.plot(ma200,'g')
st.pyplot(fig) 

# Splitting data into training and testing sets
data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):])

# Normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
data_training_array = scaler.fit_transform(data_training)

# Prepare training data - splitting it into x_train and y_train
x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

#load my model 

model = load_model('./keras_model.h5')

#Prepare testing data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

# Normalize the testing data
input_data = scaler.transform(final_df)

# Prepare x_test and y_test
x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)


# Make predictions
y_predicted = model.predict(x_test)

# Rescale the predicted values back to original scale
y_predicted = scaler.inverse_transform(y_predicted)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot the predictions
st.subheader('Prediction vs Original')
fig2 =plt.figure(figsize=(12,6))
plt.plot(y_test, 'b')
plt.plot(y_predicted, 'r')
plt.xlabel('Time')
plt.ylabel('Price')
st.pyplot(fig2)
