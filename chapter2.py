# %% 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
auto = pd.read_csv('./datasets/Auto.csv')
auto.head()

# %%
auto.describe()

# %%
auto['year'].unique()

# %%
fig, ax = plt.subplots(figsize=(8,8))
ax.bar(auto['year'].unique(), auto['year'].value_counts().values)
ax.set_xlabel("Year")
ax.set_ylabel("Quantity")

# %%
auto_re = auto.set_index('name')
auto_re

# %%
wanted_rows = ['amc rebel sst', 'ford torino']
auto_re.loc[wanted_rows, ['mpg', 'origin']]

# %%
# Boolean selection
auto_re.loc[auto_re['year'] > 80, ['weight', 'origin']] 

# Boolean selection with anonymous function Lambda
auto_re.loc[lambda df: df['year'] > 80, ['weight', 'origin']] 

# %%
auto_re.loc[
    lambda df: (df['year'] > 80) & (df['mpg'] > 30), ['weight', 'origin']
]
# %%
