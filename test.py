import pandas as pd

drug = pd.read_csv("Data/drug.csv")

new_drug = drug.sample(frac=1).reset_index(drop=True)

new_drug.to_csv("Data/drug.csv", index=False)
