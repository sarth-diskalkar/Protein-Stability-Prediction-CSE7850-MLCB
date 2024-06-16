# -*- coding: utf-8 -*-
"""KaggleCompetitionML.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1BwJ4n4jT41-WvtQCZrKoIzdls8yhvr_d
"""

from google.colab import drive
import pandas as pd
drive.mount('/content/drive')

train_path = '/content/drive/MyDrive/train.csv'
test_path = '/content/drive/MyDrive/test.csv'

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Training Data Info:")
print(train_df.info())
print("\nTraining Data Description:")
print(train_df.describe())

print("\nTesting Data Info:")
print(test_df.info())

print("\nFirst 5 rows of Training Data:")
print(train_df.head())
print("\nFirst 5 rows of Testing Data:")
print(test_df.head())
#sum of samples that are not Nan/null in training data
print(train_df.notna().sum())
#sum of samples that are  Nan/null in test data
print(test_df.isna().sum())
#sum of samples that are  Nan/null in training data
print(train_df.isna().sum())

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


vocab = "ARNDCQEGHILKMFPSTWYVU"

def one_hot_encode_seq(seq, vocab=vocab, max_length=None):
    """ One-hot encode a protein sequence, padding to max_length. """
    if max_length is None:
        max_length = len(seq)
    encoding = np.zeros((max_length, len(vocab)), dtype=int)
    aa_to_index = {aa: idx for idx, aa in enumerate(vocab)}
    for i, aa in enumerate(seq[:max_length]):  # Slice sequence if longer than max_length
        if aa in aa_to_index:
            encoding[i, aa_to_index[aa]] = 1
    return encoding.flatten()

def frequency_encode_seq(seq, vocab=vocab):
    """ Calculate frequency of each amino acid in a sequence. """
    encoding = np.zeros(len(vocab), dtype=float)
    aa_to_index = {aa: idx for idx, aa in enumerate(vocab)}
    seq_length = len(seq)
    for aa in seq:
        if aa in aa_to_index:
            encoding[aa_to_index[aa]] += 1
    if seq_length > 0:
        encoding /= seq_length
    return encoding

def encode_dataset(df, encoding_func, max_length=None):
    """ Apply encoding function to the 'sequence' column of a dataframe. """
    if 'max_length' in encoding_func.__code__.co_varnames:
        encoded_features = np.array([encoding_func(seq, vocab, max_length) for seq in df['sequence']])
    else:
        encoded_features = np.array([encoding_func(seq, vocab) for seq in df['sequence']])
    return encoded_features

max_length = max(train_df['sequence'].str.len().max(), test_df['sequence'].str.len().max())

train_features_one_hot = encode_dataset(train_df, one_hot_encode_seq, max_length)
test_features_one_hot = encode_dataset(test_df, one_hot_encode_seq, max_length)

train_features_freq = encode_dataset(train_df, frequency_encode_seq)
test_features_freq = encode_dataset(test_df, frequency_encode_seq)

train_targets = train_df['target'].values

model_one_hot = LinearRegression()
model_one_hot.fit(train_features_one_hot, train_targets)
train_predictions_one_hot = model_one_hot.predict(train_features_one_hot)

test_predictions_one_hot = model_one_hot.predict(test_features_one_hot)

model_freq = LinearRegression()
model_freq.fit(train_features_freq, train_targets)
train_predictions_freq = model_freq.predict(train_features_freq)

rmse_one_hot = np.sqrt(mean_squared_error(train_targets, train_predictions_one_hot))

rmse_freq = np.sqrt(mean_squared_error(train_targets, train_predictions_freq))

print("RMSE with One-Hot Encoding:", rmse_one_hot)
print("RMSE with Frequency Encoding:", rmse_freq)

submission_df = pd.DataFrame({
    'id': test_df['id'],
    'target': test_predictions_one_hot
})


submission_df.to_csv('/content/drive/MyDrive/linear_regression_prediction.csv', index=False)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(train_features_one_hot, train_targets)
train_predictions_rf = model_rf.predict(train_features_one_hot)
test_predictions_rf = model_rf.predict(test_features_one_hot)

rmse_rf = np.sqrt(mean_squared_error(train_targets, train_predictions_rf))
print("Training RMSE with RandomForest:", rmse_rf)
submission_df_rf = pd.DataFrame({
    'id': test_df['id'],
    'target': test_predictions_rf
})
submission_df_rf.to_csv('/content/drive/MyDrive/withoutPCA_prediction_rf.csv', index=False)
print("RandomForest CSV file has been created and saved.")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA


pca = PCA(n_components=200)
train_features_pca = pca.fit_transform(train_features_one_hot)
test_features_pca = pca.transform(test_features_one_hot)

model_rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
model_rf.fit(train_features_pca, train_targets)

train_predictions_rf = model_rf.predict(train_features_pca)
rmse_rf = np.sqrt(mean_squared_error(train_targets, train_predictions_rf))
print("Training RMSE with RandomForest:", rmse_rf)

test_predictions_rf = model_rf.predict(test_features_pca)
submission_df_rf = pd.DataFrame({
    'id': test_df['id'],
    'target': test_predictions_rf
})
submission_df_rf.to_csv('/content/drive/MyDrive/estimators50_PCAprediction_rf.csv', index=False)
print("RandomForest CSV file has been created and saved.")

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import PCA


pca = PCA(n_components=200)
train_features_pca = pca.fit_transform(train_features_one_hot)
test_features_pca = pca.transform(test_features_one_hot)

model_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
model_rf.fit(train_features_pca, train_targets)

train_predictions_rf = model_rf.predict(train_features_pca)
rmse_rf = np.sqrt(mean_squared_error(train_targets, train_predictions_rf))
print("Training RMSE with RandomForest:", rmse_rf)

test_predictions_rf = model_rf.predict(test_features_pca)
submission_df_rf = pd.DataFrame({
    'id': test_df['id'],
    'target': test_predictions_rf
})
submission_df_rf.to_csv('/content/drive/MyDrive/estimators100_PCAprediction_rf.csv', index=False)
print("RandomForest CSV file has been created and saved.")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

model_gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

model_gb.fit(train_features_one_hot, train_targets)

train_predictions_gb = model_gb.predict(train_features_one_hot)

test_predictions_gb = model_gb.predict(test_features_one_hot)

rmse_gb = np.sqrt(mean_squared_error(train_targets, train_predictions_gb))
print("Training RMSE with Gradient Boosting:", rmse_gb)

submission_df_gb = pd.DataFrame({
    'id': test_df['id'],
    'target': test_predictions_gb
})

submission_df_gb.to_csv('/content/drive/MyDrive/prediction_gb.csv', index=False)
print("Gradient Boosting CSV file has been created and saved to '/content/drive/MyDrive/prediction_gb.csv'")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features_one_hot)
test_features_scaled = scaler.transform(test_features_one_hot)

model = Sequential([
    Dense(128, activation='relu', input_shape=(train_features_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)  #no activation function
])

model.compile(optimizer='adam', loss='mean_squared_error')

history = model.fit(train_features_scaled, train_targets, epochs=50, batch_size=32, validation_split=0.2)

train_predictions_nn = model.predict(train_features_scaled)
rmse_nn = np.sqrt(mean_squared_error(train_targets, train_predictions_nn))
print("Training RMSE with Neural Network:", rmse_nn)

test_predictions_nn = model.predict(test_features_scaled)

submission_df_nn = pd.DataFrame({
    'id': test_df['id'],
    'target': test_predictions_nn.flatten()  #convert 2D array to 1D
})
submission_df_nn.to_csv('/content/drive/MyDrive/50epoch_prediction_nn.csv', index=False)
print("Neural Network CSV file has been created and saved.")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features_one_hot)
test_features_scaled = scaler.transform(test_features_one_hot)


model = Sequential([
    Dense(128, activation='relu', input_shape=(train_features_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(train_features_scaled, train_targets, epochs=300, batch_size=32, validation_split=0.2)
train_predictions_nn = model.predict(train_features_scaled)
rmse_nn = np.sqrt(mean_squared_error(train_targets, train_predictions_nn))
print("Training RMSE with Neural Network:", rmse_nn)

test_predictions_nn = model.predict(test_features_scaled)

submission_df_nn = pd.DataFrame({
    'id': test_df['id'],
    'target': test_predictions_nn.flatten()
})
submission_df_nn.to_csv('/content/drive/MyDrive/300epoch_prediction_nn.csv', index=False)
print("Neural Network CSV file has been created and saved.")

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features_one_hot)
test_features_scaled = scaler.transform(test_features_one_hot)

model = Sequential([
    Dense(128, activation='relu', input_shape=(train_features_scaled.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])


model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(train_features_scaled, train_targets, epochs=125, batch_size=32, validation_split=0.2)
train_predictions_nn = model.predict(train_features_scaled)
rmse_nn = np.sqrt(mean_squared_error(train_targets, train_predictions_nn))
print("Training RMSE with Neural Network:", rmse_nn)
test_predictions_nn = model.predict(test_features_scaled)

submission_df_nn = pd.DataFrame({
    'id': test_df['id'],
    'target': test_predictions_nn.flatten()
})
submission_df_nn.to_csv('/content/drive/MyDrive/125_epoch_prediction_nn.csv', index=False)
print("Neural Network CSV file has been created and saved.")