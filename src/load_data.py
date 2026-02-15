import pandas as pd
import os
from pathlib import Path


# get the directory of this file src/
base_dir = os.path.dirname(os.path.abspath(__file__))

data_path = os.path.join(base_dir,"..","..","Documents","creditcardfraud","creditcard.csv")

df = pd.read_csv(data_path)



df['Class'].value_counts()
df['Class'].value_counts(normalize = True)

print(df.info())









