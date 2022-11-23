import pandas as pd

def f(row):
    if row['steps'] < 500:
        val = 0
    else:
        val = 1
    return val

