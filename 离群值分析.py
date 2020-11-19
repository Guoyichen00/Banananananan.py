# -*- coding: utf-8 -*-
"""
Created on Mon Sep  7 17:57:38 2020

@author: Gyc
"""

import pandas as pd
import numpy as np

data2 = {"one":pd.Series(np.random.rand(2), index = ["a", "b"]),
        "two":pd.Series(np.random.rand(3), index = ["a", "b", "c"])}
df2 = pd.DataFrame(data2)
print(df2)
print(df2.shape[1])
print(len(df2))