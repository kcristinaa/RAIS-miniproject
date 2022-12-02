import warnings

import holidays
import pandas as pd
import datetime

# ----------------------------------------------------------------------------------------------- #

# adds wear_day
def f(row):
    if row['steps'] < 500:
        val = 0
    else:
        val = 1
    return val

# ----------------------------------------------------------------------------------------------- #

# Creates 2 columns that represent if a user has tracked at least once its spo2 or eda
def use_EDA_SpO2_ECG(df):
    df['spo2_tracking'] = ""
    df['EDA_tracking'] = ""
    df['ECG_tracking'] = ""
    users = list(df['id'].unique())

    for user in users:
        user_df = df.loc[df['id'] == user]
        # spo2
        if user_df['spo2'].isnull().sum() == len(user_df):
            df.loc[df['id'] == user, 'spo2_tracking'] = 0
        else:
            df.loc[df['id'] == user, 'spo2_tracking'] = 1
        # EDA
        if user_df['scl_avg'].isnull().sum() == len(user_df):
            df.loc[df['id'] == user, 'EDA_tracking'] = 0
        else:
            df.loc[df['id'] == user, 'EDA_tracking'] = 1
        # ECG
        if user_df['heart_rate_alert'].isnull().sum() == len(user_df):
            df.loc[df['id'] == user, 'ECG_tracking'] = 0
        else:
            df.loc[df['id'] == user, 'ECG_tracking'] = 1
    return df


# Creates a new column that represents how many (different) activity types a user has done
def different_activity_types(data):
    users = list(data['id'].unique())
    data['different_activity_types'] = ""
    for user in users:
        different_types = 0
        user_data = data.loc[data['id'] == user]
        user_data = user_data.loc[:, 'Aerobic Workout':'Yoga/Pilates']
        cols = user_data.columns
        for col in cols:
            if 1.0 in user_data[col].values:
                different_types = different_types + 1
        data.loc[data['id'] == user, 'different_activity_types'] = different_types
    return data


# Creates a new column with the percentage of fitbit usage while sleeping for each user
def use_during_sleep(data):
    users = list(data['id'].unique())
    data['used_during_night'] = ""
    for user in users:
        user_df = data.loc[data['id'] == user]
        user_df = user_df[["nightly_temperature", "full_sleep_breathing_rate", "sleep_duration", "minutesToFallAsleep",
                           "minutesAsleep", "minutesAwake", "minutesAfterWakeup", "sleep_efficiency",
                           "sleep_deep_ratio", "sleep_wake_ratio", "sleep_light_ratio", "sleep_rem_ratio"]]
        all_days = len(user_df)
        user_df = user_df.dropna(how='all')
        days_used = len(user_df)
        data.loc[data['id'] == user, 'used_during_night'] = (days_used / all_days)
    return data

# Creates a new column with True for weekend dates and False for weekdays
def is_weekend(df):
    df.date = pd.to_datetime(df.date, infer_datetime_format=True)
    df.loc[:, "is_weekend"] = df.date.dt.dayofweek  # returns 0-4 for Monday-Friday and 5-6 for Weekend
    df.is_weekend = df.is_weekend > 4
    return df

# Creates a new column with True if the date is a public holiday in Greece, Cyprus, Sweden or Italy, False otherwise
def is_holiday(df):
    gr_holidays = list(holidays.GR(years=[2021, 2022]).keys())
    swe_holdidays = list(holidays.SWE(years=[2021, 2022]).keys())
    cy_holidays = list(holidays.CY(years=[2021, 2022]).keys())
    it_holidays = list(holidays.IT(years=[2021, 2022]).keys())

    df.loc[:, 'is_holiday'] = df.date.apply(lambda d: True if (
            (d in gr_holidays) or (d in swe_holdidays) or (d in cy_holidays) or (d in it_holidays)) else False)

    return df


def interdaily_stability(data, column_name=None):
    r"""Calculate the interdaily stability"""
    if not column_name:
        warnings.warn("WARNING: No column name passed, returning unprocessed dataframe.")
        return data

    d_24h = data.groupby([
        data.index.hour,
        data.index.minute,
        data.index.second]
    ).mean().var()

    d_1h = data.var()

    return (d_24h / d_1h)
