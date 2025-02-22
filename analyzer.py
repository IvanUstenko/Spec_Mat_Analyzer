import numpy as np
import pandas as pd

def rating_points(df: pd.DataFrame):
    df['Weights'] = 1/np.sqrt(df.sum(axis=1))
    df.iloc[:, :-1] = df.iloc[:, :-1].mul(df['Weights'], axis=0)
    return df.iloc[:, :-1].sum()
