import os
import warnings
import holidays
import datetime
import numpy as np
import pandas as pd
from scipy.stats import stats


# ----------------------------------------------------------------------------------------------- #
# Add new features

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


# adds stress quantile
def add_stress_quantile(df):
    df = df.astype({"id": str})
    ids = list(np.unique((df[['id']])))

    df["stress_quantile"] = pd.qcut(df["stress_score"].rank(method='first'), [0, .25, .75, 1], labels=["low", "medium", "high"])
    df['stress_quantile'].replace(to_replace=['low', 'medium', 'high'], value=[0, 1, 2], inplace=True)

    d = pd.DataFrame()
    for user in ids:
        user_df = df[(df["id"] == user)]
        user_df["user_stress_quantile"] = pd.qcut(user_df["stress_score"].rank(method='first'), [0, .25, .75, 1],
                                                  labels=[0, 1, 2])
        d = pd.concat([d, user_df])
    df = d

    return df


# adds sleep duration average values for each user
def add_sleep_average(df):
    df = df.astype({"id": str})
    ids = list(np.unique((df[['id']])))

    d = pd.DataFrame()
    for user in ids:
        user_df = df[(df["id"] == user)]
        user_df['average_sleep_duration'] = user_df['sleep_duration'].mean()
        d = pd.concat([d, user_df])
    df = d

    return df


# adds steps average values for each user
def add_steps_average(df):
    df = df.astype({"id": str})
    ids = list(np.unique((df[['id']])))

    d = pd.DataFrame()
    for user in ids:
        user_df = df[(df["id"] == user)]
        user_df['average_steps'] = user_df['steps'].mean()
        d = pd.concat([d, user_df])
    df = d

    return df


# Creates a new column with 1.0 for weekend dates and 0.0 for weekdays
def is_weekend(df):
    df.date = pd.to_datetime(df.date, infer_datetime_format=True)
    df.loc[:, "is_weekend"] = df.date.dt.dayofweek  # returns 0-4 for Monday-Friday and 5-6 for Weekend
    df.loc[:, 'is_weekend'] = df['is_weekend'].apply(lambda d: 1.0 if d > 4 else 0.0)

    return df


# Creates a new column with 1.0 if the date is a public holiday in Greece, Cyprus, Sweden or Italy, 0.0 otherwise
def is_holiday(df):
    gr_holidays = list(holidays.GR(years=[2021, 2022]).keys())
    swe_holdidays = list(holidays.SWE(years=[2021, 2022]).keys())
    cy_holidays = list(holidays.CY(years=[2021, 2022]).keys())
    it_holidays = list(holidays.IT(years=[2021, 2022]).keys())

    df.loc[:, 'is_holiday'] = df.date.apply(lambda d: 1.0 if (
            (d in gr_holidays) or (d in swe_holdidays) or (d in cy_holidays) or (d in it_holidays)) else 0.0)

    return df


# Single function to integrate all sleep indices
def add_sleep_regularity_indices(data):

    data.date = pd.to_datetime(data.date)  # convert to datetime
    # indices
    users_is = pd.read_pickle('../data/user_level_data/is_index.pkl')
    users_isp = pd.read_pickle('../data/user_level_data/isp_index.pkl')
    users_iv = pd.read_pickle('../data/user_level_data/iv_index.pkl')
    users_sri = pd.read_pickle('../data/user_level_data/sri_index.pkl')
    users_sjl = pd.read_pickle('../data/user_level_data/sjl_index.pkl')
    users_sleep_variability = pd.read_pickle('../data/user_level_data/sleep_variability.pkl')

    merged = data.merge(users_is, on='id', how='left')
    merged = merged.merge(users_iv, on='id', how='left')
    merged = merged.merge(users_sri, on='id', how='left')
    merged = merged.merge(users_sjl, on='id', how='left')
    merged = merged.merge(users_sleep_variability, on='id', how='left')

    # add ISP per week
    merged = merged.merge(users_isp, how='left', left_on=['id', 'date'], right_on=['id', 'startDate'])
    merged = merged.sort_values(by=['id', 'date'])
    merged.isp_index = merged.isp_index.ffill(limit=6)

    merged.drop(['startDate', 'endDate'], axis=1, inplace=True)

    return merged


# Single function to integrate all step indices
def add_steps_regularity_indices(data):

    data.date = pd.to_datetime(data.date)  # convert to datetime
    # indices
    users_is = pd.read_pickle('../data/steps_indices/steps_is_index')
    users_isp = pd.read_pickle('../data/steps_indices/steps_isp_index')
    users_iv = pd.read_pickle('../data/steps_indices/steps_iv_index')
    users_sri = pd.read_pickle('../data/steps_indices/steps_sri_index')

    merged = data.merge(users_is, on='id', how='left')
    merged = merged.merge(users_iv, on='id', how='left')
    merged = merged.merge(users_sri, on='id', how='left')

    # add ISP per week
    merged = merged.merge(users_isp, how='left', left_on=['id', 'date'], right_on=['id', 'startDate'])
    merged = merged.sort_values(by=['id', 'date'])
    merged.steps_isp_index = merged.steps_isp_index.ffill(limit=6)

    merged.drop(['startDate', 'endDate'], axis=1, inplace=True)

    return merged


# Single function to integrate all exercise indices
def add_exercise_regularity_indices(data):

    data.date = pd.to_datetime(data.date)  # convert to datetime
    # indices
    users_is = pd.read_pickle('../data/exercise_indices/exercise_is_index')
    users_iv = pd.read_pickle('../data/exercise_indices/exercise_iv_index')
    users_sri = pd.read_pickle('../data/exercise_indices/exercise_sri_index')

    merged = data.merge(users_is, on='id', how='left')
    merged = merged.merge(users_iv, on='id', how='left')
    merged = merged.merge(users_sri, on='id', how='left')

    return merged


# Adds all the new features
def add_features(data, frequency):
    # add different activity types
    data = different_activity_types(data)

    # add different badge types
    data = different_badge_types(data)

    # add sleep regularity indeces
    data = add_sleep_regularity_indices(data)

    # add steps regularity indeces
    data = add_steps_regularity_indices(data)

    # add exercise regularity indeces
    data = add_exercise_regularity_indices(data)

    # replace steps < 500
    data = replace_low_steps(data)

    # add stress quantile
    if frequency == 'daily':
        data = add_stress_quantile(data)

    # add averages for sleep and steps
    data = add_sleep_average(data)
    data = add_steps_average(data)

    # add if it is weekend
    data = is_weekend(data)

    # add if it is holiday
    data = is_holiday(data)

    # date engineering for start and end sleep time
    if frequency == 'daily':
        data = start_end_sleep_time(data)

    # steps 24h encoding
    data = steps_24(data)

    return data

# ----------------------------------------------------------------------------------------------- #

# ----------------------------------------------------------------------------------------------- #
# CODE FROM pyActigraphy PACKAGE
def hour_diff(so_f, so_w):
    so_diff = so_f - so_w
    # if any sleep time is after 00:00 the calculation is slightly different
    if abs(so_diff) > 8:
        if so_diff < 0:
            so_diff = 24 - abs(so_diff)
        else:
            so_diff = - (24 - so_diff)
    return so_diff


def social_jet_lag(data):
    # split weekend and weekdays
    data = is_weekend(data)
    data.startTime = pd.to_datetime(data.startTime)
    data.loc[:, 'startHour'] = data.startTime.dt.hour

    w = data.loc[data.is_weekend == False, :]
    f = data.loc[data.is_weekend == True, :]

    so_w = stats.mode(w.startHour, keepdims=True).mode[0]
    so_f = stats.mode(f.startHour, keepdims=True).mode[0]

    so_diff = hour_diff(so_f, so_w)

    sd_w = np.mean(w.sleep_duration) / 3600000
    sd_f = np.mean(f.sleep_duration) / 3600000

    sjl = so_diff + 0.5 * (sd_f - sd_w)
    return sjl


def interdaily_stability(data, column_name='sleep'):
    r"""Calculate the interdaily stability"""

    d_24h = data.groupby([
        data.index.hour,
        data.index.minute,
        data.index.second]
    ).mean().var()

    d_1h = data.var()

    try:
        is_index = (d_24h[column_name] / d_1h[column_name])
    except:
        is_index = (d_24h / d_1h)

    return is_index


def intradaily_variability(data):
    r"""Calculate the intradaily variability"""

    c_1h = data.diff(1).pow(2).mean()

    d_1h = data.var()

    return (c_1h / d_1h)


def prob_stability(ts, threshold):
    r''' Compute the probability that any two consecutive time
    points are in the same state (wake or sleep)'''

    # Construct binarized data if requested
    data = np.where(ts > threshold, 1, 0) if threshold is not None else ts

    # Compute stability as $\delta(s_i,s_{i+1}) = 1$ if $s_i = s_{i+}$
    # Two consecutive values are equal if the 1st order diff is equal to zero.
    # The 1st order diff is either +1 or -1 otherwise.
    prob = np.mean(1 - np.abs(np.diff(data)))

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
    sri_coef = 200 * np.mean(sri_prof.values) - 100

    return sri_coef


def interval_maker(index, period, verbose):
    (num_periods, td) = divmod(
        (index[-1] - index[0]), pd.Timedelta(period)
    )
    if verbose:
        print("Number of periods: {0}\n Time unaccounted for: {1}".format(
            num_periods,
            '{} days, {}h, {}m, {}s'.format(
                td.days,
                td.seconds // 3600,
                (td.seconds // 60) % 60,
                td.seconds % 60
            )
        ))

    intervals = [(
        index[0] + (i) * pd.Timedelta(period),
        index[0] + (i + 1) * pd.Timedelta(period))
        for i in range(0, num_periods)
    ]

    return intervals


def ISp(data, period='7D', freq='1H', binarize=True, threshold=4, verbose=False):
    # data = resampled_data(freq, binarize, threshold)

    intervals = interval_maker(data.index, period, verbose)

    results = [
        interdaily_stability(data[time[0]:time[1]]) for time in intervals
    ]

    results.append(intervals)
    return results


def get_mode_sleep_per_user(user_id, df_user):
    df_user.loc[:, 'startHour'] = df_user.startTime.dt.hour
    df_user.loc[:, 'endHour'] = df_user.endTime.dt.hour
    mode_sleep_time = stats.mode(df_user.startHour, keepdims=True).mode[0]
    mode_awake_time = stats.mode(df_user.endHour, keepdims=True).mode[0]
    row = pd.Series([user_id, mode_sleep_time, mode_awake_time], index=['id', 'mode_startTime', 'mode_endTime'])
    return row


def get_variability_per_day(df_sleep):
    # read sleep modes
    dir = ".\\..\\data\\user_level_data"
    sleep_times = pd.read_pickle(os.path.join(dir, 'sleep_variability.pkl'))
    # add temporary variables required for computation
    df_sleep.loc[:, 'startHour'] = df_sleep.startTime.dt.hour
    df_sleep.loc[:, 'endHour'] = df_sleep.endTime.dt.hour
    # df_sleep = df_sleep.merge(sleep_times[['mode_startTime', 'mode_endTime', 'id']], on='id', how='left')
    # compute difference
    df_sleep.loc[:, 'variability_startTime'] = df_sleep.apply(lambda row: hour_diff(row.startHour, row.mode_startTime),
                                                              axis=1)
    df_sleep.loc[:, 'variability_endTime'] = df_sleep.apply(lambda row: hour_diff(row.endHour, row.mode_endTime),
                                                            axis=1)
    # drop temporary variables
    df_sleep.drop(['startHour', 'endHour', 'mode_startTime', 'mode_endTime'], axis=1, inplace=True)
    return df_sleep

# ----------------------------------------------------------------------------------------------- #

# ----------------------------------------------------------------------------------------------- #
# Extra preprocessing actions

# Replace steps < 500 with user's median
def replace_low_steps(df):
    df = df.astype({"id": str})
    ids = list(np.unique((df[['id']])))

    d = pd.DataFrame()

    for user in ids:
        user_df = df[(df["id"] == user)]
        user_df.loc[user_df['steps'] < 500, 'steps'] = user_df['steps'].median()
        d = pd.concat([d, user_df])

    d = d.reset_index()

    df = d.drop(columns='index')

    return df

# date engineering for start and end sleep time
def sin_transform(values):
    return np.sin(2 * np.pi * values / len(set(values)))
def cos_transform(values):
    return np.cos(2 * np.pi * values / len(set(values)))
def start_end_sleep_time(df):
    cols = ['startTime', 'endTime']
    for col in cols:
        df[col] = pd.to_datetime(df[col], format='%Y-%m-%dT%H:%M:%S.%f')
    df.loc[:, 'startHour'] = df.startTime.dt.hour
    df.loc[:, 'endHour'] = df.endTime.dt.hour

    cols = ['startHour', 'endHour']
    for col in cols:
        # Sin transformation in time features
        df["%s_sin" % col] = sin_transform(df[col])
        # Cos transformation in time features
        df["%s_cos" % col] = cos_transform(df[col])

    return df

# encode steps 24h
def createList(r1, r2):
    return [item for item in range(r1, r2 + 1)]
def steps_24(data):
    # read hourly steps
    df = pd.read_pickle("../data/daily_hourly_fitbit_types/users_steps_hourly.pkl")
    df['date'] = pd.to_datetime(df.date, format='%m/%d/%y')
    df = df.sort_values('date')
    df = df.reset_index()
    df = df.drop(columns='index')

    l = createList(0, 23)
    for col in l:
        df[col] = 0

    cols = df.iloc[:, 4:].columns
    for col in cols:
        df.loc[df['hour'] == col, col] = df['steps']
        df = df.rename(columns={col: "Steps_hour%s" % col})

    df = df.groupby(['date', 'id'])['Steps_hour0', 'Steps_hour1',
                                    'Steps_hour2', 'Steps_hour3', 'Steps_hour4', 'Steps_hour5',
                                    'Steps_hour6', 'Steps_hour7', 'Steps_hour8', 'Steps_hour9',
                                    'Steps_hour10', 'Steps_hour11', 'Steps_hour12', 'Steps_hour13',
                                    'Steps_hour14', 'Steps_hour15', 'Steps_hour16', 'Steps_hour17',
                                    'Steps_hour18', 'Steps_hour19', 'Steps_hour20', 'Steps_hour21',
                                    'Steps_hour22', 'Steps_hour23'].sum().reset_index()
    df = df[['id', 'date', 'Steps_hour0', 'Steps_hour1',
             'Steps_hour2', 'Steps_hour3', 'Steps_hour4', 'Steps_hour5',
             'Steps_hour6', 'Steps_hour7', 'Steps_hour8', 'Steps_hour9',
             'Steps_hour10', 'Steps_hour11', 'Steps_hour12', 'Steps_hour13',
             'Steps_hour14', 'Steps_hour15', 'Steps_hour16', 'Steps_hour17',
             'Steps_hour18', 'Steps_hour19', 'Steps_hour20', 'Steps_hour21',
             'Steps_hour22', 'Steps_hour23']]

    df["date"] = pd.to_datetime(pd.to_datetime(df["date"]).dt.date)
    df = df.astype({"id": str})

    data['date'] = data['date'].astype('datetime64')
    df['date'] = df['date'].astype('datetime64')

    data = data.merge(df, how='left', on=['id', 'date'])

    return data
