# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
from ISLP import load_data
from ISLP.models import (
    ModelSpec as MS,
    summarize,
    poly
)

# %%
boston = load_data("Boston")

terms = boston.columns.drop('medv')
terms

# %%
X = MS(terms).fit_transform(boston)
y = boston['medv']
model = sm.OLS(y, X)
results = model.fit()
summarize(results).values
# %%
