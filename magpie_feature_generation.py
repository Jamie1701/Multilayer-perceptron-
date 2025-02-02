'''
Potential bug with multiple featurizers, it gets stuck in an infinite error loop. 
For the purpose of this project, ElementProperty has proven sufficient to generate meaningful results in the implementation.

'''

from cubic_sg_pred_data import df_valid
import os
import pandas as pd
import numpy as np
from pymatgen.core.composition import Composition
from matminer.featurizers.composition import (
    ElementProperty, Stoichiometry, ValenceOrbital, IonProperty, ElectronegativityDiff, ElementFraction, BandCenter
)

df_valid["composition"] = df_valid["cleaned_formula"].apply(lambda x: Composition(x))

feature_dfs = [df_valid[["sg", "sgNumber", "cleaned_formula"]]]  # Start with space group + formula


# Includes atomic number, weight, melting temp, etc.
try:
    ep_featurizer = ElementProperty.from_preset("magpie")
    df_ep = ep_featurizer.featurize_dataframe(df_valid, col_id="composition", ignore_errors=True)
    feature_dfs.append(df_ep.drop(columns=["composition"]))
except Exception as e:
    print(f"Error in Element Properties: {e}") # Initially used to combat the infinite error loop stemming from multiple featurizer. 

df_final = pd.concat(feature_dfs, axis=1)

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
os.makedirs(data_dir, exist_ok=True)
file_path = os.path.join(data_dir, "magpie_features.csv") 
# Be mindful, if the ElementProperty featurizer isn't run to 100% completion, the file will only contain space group, space group number and cleaned formula. 
# I have put the resultant file in the folder and converted it to a xlsx file for ease of viewing, the data is not scaled or encoded in this file. (magpie_features_6.xlsx) 

df_final.to_csv(file_path, index=False) 

if __name__ == "__main__":
    print(df_final.shape)
    print(df_final.head())