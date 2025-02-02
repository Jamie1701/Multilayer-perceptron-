'''
-------------------------------------------
Data scalling and encoding. 
Standard scaler applied to all numerical features. One hot encoding applied to the target variable.
144 space groups are one hot encoded.
139 features are scaled. 
-------------------------------------------
'''

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, "data", "magpie_features_6.xlsx")

df_final = pd.read_excel(file_path)

df_final.dropna(inplace=True)

X = df_final.drop(columns=["sg", "cleaned_formula", "alpha", "beta", "gamma"]).values
y = df_final["sgNumber"].values.reshape(-1, 1)  # Reshape for encoder

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

encoder = OneHotEncoder(sparse_output=False)
y_one_hot_encoded = encoder.fit_transform(y)

X_scaled_df = pd.DataFrame(X_scaled, columns=df_final.drop(columns=["sg", "cleaned_formula","alpha", "beta", "gamma"]).columns)
y_encoded_df = pd.DataFrame(y_one_hot_encoded, columns=[f"SG_{category}" for category in encoder.categories_[0]]) # Helps enumerate the space groups. Not in order. Number corresponds to sg number.

df_scaled_encoded = pd.concat([X_scaled_df, y_encoded_df], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_one_hot_encoded, test_size=0.2, random_state=42)

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
file_path = os.path.join(data_dir, "scaled_encoded_data.csv")

df_scaled_encoded.to_csv(file_path, index=False)

if __name__ == "__main__":
    row_sums = np.sum(y_test, axis=1)
    print("Min label row sum:", row_sums.min(), "Max label row sum:", row_sums.max())
    
    print(f"NaNs in X_train: {np.isnan(X_train).sum()}, NaNs in X_test: {np.isnan(X_test).sum()}") # For debugging the implementation. 
    print(f"NaNs in y_train: {np.isnan(y_train).sum()}, NaNs in y_test: {np.isnan(y_test).sum()}") # NaN values in train set caused model errors, now removed. 