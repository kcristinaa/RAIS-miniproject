import pandas as pd

def f(row):
    if row['steps'] < 500:
        val = 0
    else:
        val = 1
    return val

def stress_quantile(df):
    df["stress_quantile"] = pd.qcut(df["stress_score"].rank(method='first'), [0, .25, .75, 1],
                                    labels=["low", "medium", "high"])