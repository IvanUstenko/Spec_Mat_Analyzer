import numpy as np
import pandas as pd

def competative_points(df: pd.DataFrame, topic="all", subtopic="all", weight_func="sqrt") -> pd.Series:
    df = df.drop(columns=['Процент порядка', 'Процент номера'])
    if topic != "all":
        df = df.loc[df["Тема"] == topic]
    if subtopic != "all":
        df = df.loc[df["Подтема"] == subtopic]
    df = df.drop(columns=['Тема', 'Подтема'])
    df = df.astype(float)
    df = df.replace([2,3,4,5,6,7,8], 1)
    df = df[df.sum(axis=1) != 0]
    if weight_func == "sqrt":
        df['Weights'] = 1/np.sqrt(df.sum(axis=1))
        df.iloc[:, :-1] = df.iloc[:, :-1].mul(df['Weights'], axis=0)
        return df.iloc[:, :-1].sum()
    elif weight_func == "linear":
        df['Weights'] = df.shape[1] - df.sum(axis=1)
        df.iloc[:, :-1] = df.iloc[:, :-1].mul(df['Weights'], axis=0)
        return df.iloc[:, :-1].sum()

def total_problems(df: pd.DataFrame) -> pd.Series:
    return df.sum()

def work_points(df: pd.DataFrame, topic="all", subtopic="all", weight_func="linear") -> pd.Series:
    df = df.drop(columns=['Процент порядка', 'Процент номера'])
    if topic != "all":
        df = df.loc[df["Тема"] == topic]
    if subtopic != "all":
        df = df.loc[df["Подтема"] == subtopic]
    df = df.drop(columns=['Тема', 'Подтема'])
    df = df.astype(float)
    df = df.replace([2,3,4,5,6,7,8], 1)
    df = df[df.sum(axis=1) != 0]
    if weight_func == "sqrt":
        df['Weights'] = np.sqrt(df.sum(axis=1))
        df.iloc[:, :-1] = df.iloc[:, :-1].mul(df['Weights'], axis=0)
        return df.iloc[:, :-1].sum()
    elif weight_func == "linear":
        df['Weights'] = df.sum(axis=1)
        df.iloc[:, :-1] = df.iloc[:, :-1].mul(df['Weights'], axis=0)
        return df.iloc[:, :-1].sum()