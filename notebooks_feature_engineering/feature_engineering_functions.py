import warnings
import holidays
import datetime
import numpy as np
import pandas as pd
from scipy.stats import stats


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

    df['early_features'] = np.where((df['spo2_tracking'] == 1) | (df['EDA_tracking'] == 1) | (df['ECG_tracking'] == 1),
                                    1, 0)
    df = df.drop(columns=['spo2_tracking', 'EDA_tracking', 'ECG_tracking'])

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


# Creates a new column that represents how many (different) badge types a user has gain
def different_badge_types(data):
    users = list(data['id'].unique())
    data['different_badge_types'] = ""
    for user in users:
        different_types = 0
        user_data = data.loc[data['id'] == user]
        user_data = user_data.loc[:, 'DAILY_FLOORS':'LIFETIME_WEIGHT_GOAL_SETUP']

        cols = user_data.columns
        for col in cols:
            if 1.0 in user_data[col].values:
                different_types = different_types + 1
        data.loc[data['id'] == user, 'different_badge_types'] = different_types
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


def intradaily_variability(data, column_name=None):
    r"""Calculate the intradaily variability"""
    if not column_name:
        warnings.warn("WARNING: No column name passed, returning unprocessed dataframe.")
        return data

    c_1h = data.diff(1).pow(2).mean()

    d_1h = data.var()

    return (c_1h / d_1h)


def social_jet_lag(data):
    # split weekend and weekdays
    data = is_weekend(data)
    data.startTime = pd.to_datetime(data.startTime)
    data.loc[:, 'startHour'] = data.startTime.dt.hour

    w = data.loc[data.is_weekend == False, :]
    f = data.loc[data.is_weekend == True, :]

    so_w = stats.mode(w.startHour, keepdims=True).mode[0]
    so_f = stats.mode(f.startHour, keepdims=True).mode[0]

    so_diff = so_f - so_w
    # if any sleep time is after 00:00 the calculation is slightly different
    if abs(so_diff) > 8:
        if so_diff < 0:
            so_diff = 24 - abs(so_diff)
        else:
            so_diff = - (24 - so_diff)

    sd_w = np.mean(w.sleep_duration) / 3600000
    sd_f = np.mean(f.sleep_duration) / 3600000

    sjl = so_diff + 0.5 * (sd_f - sd_w)
    return sjl


def prob_stability(ts, threshold):
    r''' Compute the probability that any two consecutive time
    points are in the same state (wake or sleep)'''

    # Construct binarized data if requested
    data = np.where(ts > threshold, 1, 0) if threshold is not None else ts

    # Compute stability as $\delta(s_i,s_{i+1}) = 1$ if $s_i = s_{i+}$
    # Two consecutive values are equal if the 1st order diff is equal to zero.
    # The 1st order diff is either +1 or -1 otherwise.
    prob = np.mean(1-np.abs(np.diff(data)))

    return prob


def sri_profile(data, threshold):
    r''' Compute daily profile of sleep regularity indices '''
    # Group data by hour/minute/second across all the days contained in the
    # recording
    data_grp = data.groupby([
        data.index.hour,
        data.index.minute,
        data.index.second
    ])
    # Apply prob_stability to each data group (i.e series of consecutive points
    # that are 24h apart for a given time of day)
    sri_prof = data_grp.apply(prob_stability, threshold=threshold)
    # sri_prof.index = pd.timedelta_range(
    #     start='0 day',
    #     end='1 day',
    #     freq=data.index.freq,
    #     closed='left'
    # )
    return sri_prof


def sri(data, threshold=None):
    r''' Compute sleep regularity index (SRI)'''

    # Compute daily profile of sleep regularity indices
    sri_prof = sri_profile(data, threshold)

    # Calculate SRI coefficient
    sri_coef = 200*np.mean(sri_prof.values)-100

    return sri_coef






