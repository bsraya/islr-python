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
boston.head()

# %%
X = pd.DataFrame({
    'intercept': np.ones(boston.shape[0]),
    'lstat': boston['lstat']
})

X[:5]
# %%
"""
Columns in the Boston House Pricing dataset:
crim: per capita crime rate by town.
zn: proportion of residential land zoned for lots over 25,000 sq.ft.
indus: proportion of non-retail business acres per town.
chas: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
nox: nitrogen oxides concentration (parts per 10 million).
rm: average number of rooms per dwelling.
age: proportion of owner-occupied units built prior to 1940.
dis: weighted mean of distances to five Boston employment centres.
ptratio: pupil-teacher ratio by town.
lstat: lower status of the population (percent).
medv: median value of owner-occupied homes in $1000s.
"""

X = pd.DataFrame({
    'intercept': np.ones(boston.shape[0]),
    'lstat': boston['lstat']
})
y = boston['medv']
model = sm.OLS(y, X)
results = model.fit()
summarize(results) # results.summary()

# %%
def abline(ax , b, m, *args , ** kwargs):
  xlim = ax.get_xlim ()
  ylim = [m * xlim [0] + b, m * xlim [1] + b]
  ax.plot(xlim , ylim , *args , ** kwargs)

# %%
fig, ax = plt.subplots()
ax.scatter(boston['lstat'], boston['medv'])
abline(
  ax,
  results.params['intercept'],
  results.params['lstat'],
  color='red'
)
ax.set_xlabel("Lower status of the population (percent).")
ax.set_ylabel("median value of owner-occupied homes (in $1000).")

# %%
fig, ax = plt.subplots()
ax.scatter(results.fittedvalues, results.resid)
ax.set_xlabel("Fitted values")
ax.set_ylabel("Residuals")
ax.axhline(0, color='red', linestyle='--')

# %%
comparison = pd.DataFrame({
  'fitted': results.fittedvalues,
  'real': boston['medv'],
  'residual': results.resid
})
comparison.head()
# %%
