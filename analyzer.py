import numpy as np
import pandas as pd

def topic_loc(df: pd.DataFrame, topic="all", subtopic="all") -> pd.DataFrame:
    if topic != "all":
        df = df.loc[df["Тема"] == topic]
    if subtopic != "all":
        df = df.loc[df["Подтема"] == subtopic]
    df = df.drop(columns=['Тема', 'Подтема'])
    return df

def bin_conduit(df: pd.DataFrame, days = "all") -> pd.DataFrame:
    df = df.drop(columns=['Процент порядка', 'Процент номера'])
    df = df.astype(float)
    if days == "all":
        df = df.replace([2,3,4,5,6,7,8], 1)
    else:
        df = df.replace(days, -1)
        df = df.replace([1,2,3,4,5,6,7,8], 0)
        df = df.replace(-1,1) #потом переделать
    return df

def competative_points(df: pd.DataFrame, topic="all", subtopic="all", weight_func="sqrt", days="all") -> pd.Series:
    df = topic_loc(df, topic, subtopic)
    df = bin_conduit(df, days)
    df = df[df.sum(axis=1) != 0]
    if weight_func == "sqrt":
        df['Weights'] = 1/np.sqrt(df.sum(axis=1))
        df.iloc[:, :-1] = df.iloc[:, :-1].mul(df['Weights'], axis=0)
        return df.iloc[:, :-1].sum()
    elif weight_func == "linear":
        df['Weights'] = df.shape[1] - df.sum(axis=1)
        df.iloc[:, :-1] = df.iloc[:, :-1].mul(df['Weights'], axis=0)
        return df.iloc[:, :-1].sum()

def total_problems(df: pd.DataFrame, topic="all", subtopic="all", days="all") -> pd.Series:
    return bin_conduit(topic_loc(df, topic, subtopic), days).sum()

def work_points(df: pd.DataFrame, topic="all", subtopic="all", weight_func="linear", days="all") -> pd.Series:
    df = bin_conduit(topic_loc(df, topic, subtopic), days)
    df = df[df.sum(axis=1) != 0]
    if weight_func == "sqrt":
        df['Weights'] = np.sqrt(df.sum(axis=1))
        df.iloc[:, :-1] = df.iloc[:, :-1].mul(df['Weights'], axis=0)
        return df.iloc[:, :-1].sum()
    elif weight_func == "linear":
        df['Weights'] = df.sum(axis=1)
        df.iloc[:, :-1] = df.iloc[:, :-1].mul(df['Weights'], axis=0)
        return df.iloc[:, :-1].sum()