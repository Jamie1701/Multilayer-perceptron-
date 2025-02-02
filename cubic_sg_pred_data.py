'''
----------------------------------------
Cleaning data obtained from the crystallography open database (COD) for cubic crystal structures.
The data is cleaned to ensure valid chemical formulas and stoichiometry to be used with matminer and magpie.
Maximum element ratios for structures are defined for runtime convenience.
----------------------------------------
'''

import pandas as pd
import os
from pymatgen.core.composition import Composition

base_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(base_dir, "data")
file_path = os.path.join(data_dir, "COD-selection.xlsx")

# Sheet_1 contains the full data set obtained from the COD database. The cubic_data sheet contains approx 3300 entries.
df = pd.read_excel(file_path, engine="openpyxl", sheet_name="cubic_data")

# Don't necessarily need all columns. Given the focus is on cubic crystal structures, alpha, beta, and gamma can be dropped.
columns_to_keep = ["a", "b", "c", "alpha", "beta", "gamma", "sg","sgNumber", "cellformula", "Z", "Zprime", "formula", "calcformula"]
df_clean = df[columns_to_keep].copy()  # Ensure we work on a copy

df_cleaned = df_clean.dropna(subset=["cellformula"])

# Reasonable stoichiometry
max_subscript = 100

def clean_cellformula(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans chemical formulas and maintains corresponding space group (SG) & other data.
    - Removes invalid formulas.
    - Ensures integer subscripts and reasonable values.
    - Returns a filtered DataFrame with valid formulas.
    """
    valid_rows = []

    for idx, row in df.iterrows():
        entry = str(row["cellformula"]).strip("- ").replace(" ", "")
        try:
            # pymatgen composition
            comp = Composition(entry)

            valid_formula = True 
            for element, amount in comp.get_el_amt_dict().items():
                if not amount.is_integer() or amount > max_subscript: # Check that stoichiometry is reasonable -- nothing more than runtime convenience. 
                    valid_formula = False  
                    break 

            if valid_formula:
                row["cleaned_formula"] = comp.reduced_formula
                valid_rows.append(row)

        except Exception:
            continue 

    df_filtered = pd.DataFrame(valid_rows)

    return df_filtered

df_valid = clean_cellformula(df_cleaned)

df_valid = df_valid.drop(["cellformula", "formula", "calcformula"], axis=1) # Dropped as these are redundant given we have cleaned formulas.

if __name__ == "__main__":
    print(df_valid.head())
    print(df_valid.shape) # Shape (3237, 11)
