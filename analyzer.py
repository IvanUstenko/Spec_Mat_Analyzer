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
    df = df.astype(float)
    if days == "all":
        df = df.replace([2,3,4,5,6,7,8], 1)
    else:
        df = df.replace(days, -1)
        df = df.replace([1,2,3,4,5,6,7,8], 0)
        df = df.replace(-1,1) 
    return df

def total_problems(df: pd.DataFrame, topic="all", subtopic="all", days="all") -> pd.Series:
    return bin_conduit(topic_loc(df, topic, subtopic), days).sum()

def competative_points(df: pd.DataFrame, topic="all", subtopic="all", weight_func="sqrt", days="all") -> pd.Series:
    df = topic_loc(df, topic, subtopic)
    df = bin_conduit(df, days)
    if not('Comp_Weights' in df.columns):
        df = add_comp_weight(df, weight_func = weight_func)
    df = df[df.sum(axis=1) != 0]
    df.iloc[:, :-1] = df.iloc[:, :-1].mul(df['Comp_Weights'], axis=0)
    return df.iloc[:, :-1].sum()

def work_points(df: pd.DataFrame, topic="all", subtopic="all", days="all", weight_func="linear") -> pd.Series:
    df = topic_loc(df, topic, subtopic)
    df = bin_conduit(df, days)
    if not('Work_Weights' in df.columns):
        df = add_work_weight(df, weight_func = weight_func)
    df.iloc[:, :-1] = df.iloc[:, :-1].mul(df['Work_Weights'], axis=0)
    return df.iloc[:, :-1].sum()
    
def add_comp_weight(df: pd.DataFrame, weight_func="sqrt") -> pd.DataFrame:
    df = df[df.sum(axis=1) != 0]
    if weight_func == "sqrt":
        df.loc[:, 'Comp_Weights'] = 1/np.sqrt(df.sum(axis=1))
    elif weight_func == "const":
        df.loc[:, 'Comp_Weights'] = 1
    elif weight_func == "linear":
        df.loc[:, 'Comp_Weights'] = df.shape[1] - df.sum(axis=1)
    elif weight_func == "trifecta":
        df.loc[:, 'Comp_Weights'] = np.where(df.sum(axis=1) <= 3, 1, 0)
    elif weight_func == "sigm":
        n = df.shape[1]
        wmax = 0.999
        wmin = 0.001
        b = -np.log(1/wmax - 1)
        k = (np.log(1/wmin - 1) - np.log(1/wmax - 1))/n
        df.loc[:, 'Comp_Weights'] = 1 / (1 + 2.71 ** (df.sum(axis=1) * k - b))
    return df

def add_work_weight(df: pd.DataFrame, weight_func="linear") -> pd.DataFrame:
    df = df[df.sum(axis=1) != 0]
    if weight_func == "sqrt":
        df.loc[:, 'Work_Weights'] = np.sqrt(df.sum(axis=1))
    elif weight_func == "linear":
        df.loc[:, 'Work_Weights'] = df.sum(axis=1)
    return df

def stats(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(data={
                            "Соревновательные очки" : competative_points(df, weight_func = "sigm"),
                            "Среднее кол-во очков за задачу" : competative_points(df, weight_func = "sigm").to_numpy() / total_problems(df),
                            "Соревновательные очки в комбинаторике" : competative_points(df, topic='Комбинаторика', weight_func="sigm"),
                            "Среднее кол-во очков за задачу в комбинаторике" : competative_points(df, topic='Комбинаторика', weight_func="sigm")  / total_problems(df, topic='Комбинаторика'),
                            "Соревновательные очки в ТЧ" : competative_points(df, topic='ТЧ', weight_func="sigm"),
                            "Среднее кол-во очков за задачу в ТЧ" : competative_points(df, topic='ТЧ', weight_func="sigm")  / total_problems(df, topic='ТЧ'),
                            "Соревновательные очки в графах" : competative_points(df, topic='Графы', weight_func="sigm"),
                            "Среднее кол-во очков за задачу в графах" : competative_points(df, topic='Графы', weight_func="sigm")  / total_problems(df, topic='Графы'),
                            "Стартовые соревновательные очки" : competative_points(df, days=[1,2], weight_func='sigm')})

def personal_stats(stats: pd.DataFrame, name: str):
    print(name)
    for stat in stats.columns:
        print(f"{stat}: {np.round(stats[stat][name], 2)}, #{stats.shape[0] - np.argsort(stats[stat].sort_values(ascending=False))[name]}") 



