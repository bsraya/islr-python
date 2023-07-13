# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %%
random_generator = np.random.default_rng(3)

# %%
y = random_generator.standard_normal(10)
y, np.mean(y), y.mean()

# %%
y = random_generator.standard_normal((10,3))
np.mean(y), y.mean()

# %%
fig, ax = plt.subplots(figsize=(8,8))
x = random_generator.standard_normal(100)
y = random_generator.standard_normal(100)
ax.scatter(x, y, marker = 'o')

# %%
