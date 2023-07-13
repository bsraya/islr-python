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
y = boston['medv']
model = sm.OLS(y, X)
results = model.fit()
summarize(results)

# %%
design = MS(['lstat'])
design = design.fit(boston)
X = design.transform(boston)
X[:5]


# %%
new_df = pd.DataFrame({ 'lstat': [5, 10, 15] })
newX = design.transform(new_df)
newX

# %%
new_predictions = results.get_prediction(newX)
new_predictions.predicted_mean

# %%
