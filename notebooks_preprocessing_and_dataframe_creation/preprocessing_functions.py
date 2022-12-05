import json
import statistics
import numpy as np
import pandas as pd
from datetime import datetime

from pymongo import MongoClient


# --------------------------------------------------------------------------- #

# Pre-processing actions for baseline dataframe

def fitbit_basic_preprocessing(df):
    # selecting the experiment days
    df = df.sort_values(by='date', ascending=True)
    df['date'] = pd.to_datetime(df['date'].astype("str"), format='%Y-%m-%d')
    df = df.loc[((df['date'] > '2021-05-23') & (df['date'] < '2021-07-27')) | (
            (df['date'] > '2021-11-14') & (df['date'] < '2022-01-18'))]
    df.reset_index(inplace=True, drop=True)

    # drop duplicates
    df = df.loc[df.astype(str).drop_duplicates().index]

    # convert data types falsely described as categorical
    df[["lightly_active_minutes", "moderately_active_minutes", "very_active_minutes", "sedentary_minutes"]] = df[
        ["lightly_active_minutes", "moderately_active_minutes", "very_active_minutes", "sedentary_minutes"]].apply(
        pd.to_numeric)

    return df


# Adding sleep startTime and endTime columns to the dataframe
def fitbit_intraday_sleep(df):
    # setup mongo connection for reading extra data
    with open('.\\..\\credentials.json') as f:
        data = json.load(f)
        username = data['username']
        password = data['password']
    client = MongoClient('mongodb://%s:%s@127.0.0.1' % (username, password))
    db = client.rais_anonymized
    col = db.fitbit

    # read intra-day data from Mongo
    df_mongo = pd.DataFrame(list(col.find({"$and": [
        {'type': 'sleep'},
        {'data.mainSleep': True}
    ]},
        {'_id': 0, 'id': 1, 'data.dateOfSleep': 1, 'data.startTime': 1, 'data.endTime': 1}
    )))
    df_mongo.loc[:, "date"] = df_mongo.data.str.get('dateOfSleep')
    df_mongo.loc[:, "startTime"] = df_mongo.data.str.get('startTime')
    df_mongo.loc[:, "endTime"] = df_mongo.data.str.get('endTime')
    df_mongo.drop(columns=['data'], inplace=True)
    df_mongo["date"] = pd.to_datetime(pd.to_datetime(df_mongo["date"]).dt.date)
    df = df.merge(df_mongo, how='left', on=['id', 'date'])
    return df


# --------------------------------------------------------------------------- #

def sin_transform(values):
    """
    Applies SIN transform to a series value.
    Args:
        values (pd.Series): A series to apply SIN transform on.
    Returns
        (pd.Series): The transformed series.
    """

    return np.sin(2 * np.pi * values / len(set(values)))


def cos_transform(values):
    """
    Applies COS transform to a series value.
    Args:
        values (pd.Series): A series to apply SIN transform on.
    Returns
        (pd.Series): The transformed series.
    """
    return np.cos(2 * np.pi * values / len(set(values)))


def date_engineering(data):  # data could be any dataframe that needs date engineering

    data['date'] = pd.to_datetime(data.date, format='%m/%d/%y %H:%M:%S')
    data = data.astype({"date": str})

    # Extract features from date
    data["year"] = data["date"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').year)
    data["month"] = data["date"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').month)
    data["weekday"] = data["date"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').weekday())
    data["week"] = data["date"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').isocalendar()[1])
    data["day"] = data["date"].apply(lambda x: datetime.strptime(x, '%Y-%m-%d').day)

    # Sin transformation in date features
    data["month_sin"] = sin_transform(data["month"])
    data["weekday_sin"] = sin_transform(data["weekday"])
    data["week_sin"] = sin_transform(data["week"])
    data["day_sin"] = sin_transform(data["day"])

    # Cosine transformation in date features
    data["month_cos"] = cos_transform(data["month"])
    data["weekday_cos"] = cos_transform(data["weekday"])
    data["week_cos"] = cos_transform(data["week"])
    data["day_cos"] = cos_transform(data["day"])

    data = data.drop(columns=['year', 'month', 'weekday', 'week', 'day'])

    return data


# --------------------------------------------------------------------------- #

def sema_basic_preprocessing(df):
    df["negative_feelings"] = np.where(df['TENSE/ANXIOUS'] == 1, 1, np.where(df['ALERT'] == 1, 1,
                                                                             np.where(df['SAD'] == 1, 1,
                                                                                      np.where(df['TIRED'] == 1, 1,
                                                                                               0))))
    df["positive_feelings"] = np.where(df['HAPPY'] == 1, 1,
                                       np.where(df['NEUTRAL'] == 1, 1, np.where(df['RESTED/RELAXED'] == 1, 1, 0)))
    df = df.drop(columns=['ALERT', 'HAPPY', 'NEUTRAL', 'RESTED/RELAXED', 'SAD', 'TENSE/ANXIOUS', 'TIRED'])

    return df


# --------------------------------------------------------------------------- #

def one_hot_encoding(fitbit):
    # badgeType encoding
    s = fitbit['badgeType']
    dum = pd.get_dummies(s.apply(pd.Series).stack()).sum(level=0)
    df = pd.concat([s, dum], axis=1)
    fitbit = pd.concat([fitbit, df], axis=1)
    fitbit = fitbit.drop(columns='badgeType')

    # activity type encoding
    s = fitbit['activityType']
    dum = pd.get_dummies(s.apply(pd.Series).stack()).sum(level=0)
    df = pd.concat([s, dum], axis=1)
    fitbit = pd.concat([fitbit, df], axis=1)
    fitbit = fitbit.drop(columns='activityType')

    # mindfulness session encoding - highly imbalanced
    fitbit['mindfulness_session'].replace(to_replace=['False', True], value=[0, 1], inplace=True)

    # age encoding
    fitbit['age'].replace(to_replace=['<30', '>=30'], value=[0, 1], inplace=True)

    # gender encoding
    fitbit['gender'].replace(to_replace=['MALE', 'FEMALE'], value=[0, 1], inplace=True)

    # bmi encoding
    fitbit['bmi'] = fitbit['bmi'].fillna(fitbit['bmi'].mode().iloc[0])
    fitbit["bmi"] = fitbit["bmi"].apply(lambda x: 31.0 if x == '>=30' else x)
    fitbit["bmi"] = fitbit["bmi"].apply(lambda x: 18.0 if x == '<19' else x)
    fitbit["bmi"] = fitbit["bmi"].apply(lambda x: 26.0 if x == '>=25' else x)  # it belongs to overweight
    fitbit["bmi"] = fitbit["bmi"].apply(lambda x: 31 if x == '>=30' else x)
    fitbit['bmi'] = fitbit.bmi.apply(lambda bmi: 'Underweight' if bmi < 18.5 else ('Normal' if bmi < 25 else (
        'Overweight' if bmi < 30 else 'Obese')))  # 0: Underweight, 1: Normal, 2: Overweight, 3: Obese

    # ECG alert encoding
    fitbit['heart_rate_alert'].replace(to_replace=['NONE', 'LOW_HR'], value=[0, 1], inplace=True)

    return fitbit


# --------------------------------------------------------------------------- #

# Creates 1 column that represent if a user has tracked at least once its spo2 or eda or ecg

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

    df['early_features'] = np.where((df['spo2_tracking'] == 1) | (df['EDA_tracking'] == 1) | (df['ECG_tracking'] == 1), 1, 0)
    df = df.drop(columns=['spo2_tracking', 'EDA_tracking', 'ECG_tracking'])
    return df


# --------------------------------------------------------------------------- #

def post_preprocessing(df):
    df = use_EDA_SpO2_ECG(df)

    categorical = ['mindfulness_session', 'age', 'gender', 'bmi', 'heart_rate_alert', 'DAILY_FLOORS', 'DAILY_STEPS',
                   'GOAL_BASED_WEIGHT_LOSS', 'LIFETIME_DISTANCE', 'LIFETIME_FLOORS', 'LIFETIME_WEIGHT_GOAL_SETUP',
                   'Aerobic Workout', 'Bike', 'Bootcamp', 'Circuit Training', 'Elliptical', 'Hike', 'Interval Workout',
                   'Martial Arts', 'Run', 'Spinning', 'Sport', 'Swim', 'Treadmill', 'Walk', 'Weights', 'Workout',
                   'Yoga/Pilates']

    labels = ['label_ttm_stage', 'label_breq_self_determination',
              'label_sema_negative_feelings', 'label_ipip_extraversion_category', 'label_ipip_agreeableness_category',
              'label_ipip_conscientiousness_category', 'label_ipip_stability_category', 'label_ipip_intellect_category',
              'label_stai_stress_category', 'label_panas_negative_affect']

    # Replace outliers with NaNs separately for each column in the dataframe
    columns = list(df.iloc[:, 2:].columns)  # excludes id and date
    # exclude labels
    for x in labels:
        columns.remove(x)
    # exclude categorical features
    for x in categorical:
        columns.remove(x)
    for col in columns:
        if (col == "startTime") or (col == "endTime"):
            continue
        df[col] = df[col].mask(df[col].sub(df[col].mean()).div(df[col].std()).abs().gt(3))

    # Replace NaN values with column's median for continuous features
    columns = list(df.iloc[:, 2:].columns)  # excludes id and date 
    # exclude labels
    for x in labels:
        columns.remove(x)
    # exclude categorical features
    for x in categorical:
        columns.remove(x)
    for col in columns:
        if (col == "startTime") or (col == "endTime"):
            continue
        df[col] = df[col].apply(pd.to_numeric, errors='coerce')
        df[col] = df[col].fillna(df[col].median())

    # Replace NaN values with column's more frequent occurrence for categorical features
    for col in categorical:
        df[col] = df[col].fillna(df[col].mode().iloc[0])

    return df


# --------------------------------------------------------------------------- #

# adds wear_day
def f(row):
    if row['steps'] < 500:
        val = 0
    else:
        val = 1
    return val


# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #

# Weekly frequency in fitbit dataframe regarding the corresponding survey

def weekly_fitbit_frequency(survey, fitbit, users):  # survey is stai or panas dataframe

    column_list = list(survey.columns)
    fitbit_columns = list(fitbit.columns)
    fitbit_columns.remove('id')
    fitbit_columns.remove('date')
    for x in fitbit_columns:
        column_list.append(x)
    fitbit_survey = pd.DataFrame(columns=column_list)
    for user in users:
        user_survey = survey.loc[survey['id'] == user]
        user_fitbit = fitbit.loc[fitbit['id'] == user]
        fitbit_survey = pd.concat([fitbit_survey, user_survey], ignore_index=True)
        for day in list(user_survey['date']):
            weekly_fitbit = user_fitbit.loc[user_fitbit['date'] < day]
            weekly_fitbit = weekly_fitbit.set_index(['date'])
            weekly_fitbit = weekly_fitbit.last('7D')
            cols = list(weekly_fitbit.columns)
            cols.remove('id')
            for column in cols:
                fitbit_survey.loc[(fitbit_survey.id == user) & (fitbit_survey.date == day), column] = statistics.median(
                    list(weekly_fitbit[column]))
    fitbit_survey["date"] = pd.to_datetime(pd.to_datetime(fitbit_survey["date"]).dt.date)

    return fitbit_survey


# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #

# Split train and test set in order each user to belong only in one of them

def train_test_split_per_user(data, train_size=0.7):
    users = list(set(data.id))
    users = sorted(users, reverse=True)  # fix randomness
    total_users = len(users)
    slice = int(train_size * total_users)
    users_train = users[:slice]
    users_test = users[slice:]
    return data[data.id.isin(users_train)], data[data.id.isin(users_test)]


# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #

# Label Engineering VO2Max Get VO2 max (cardio score) category based on age category and filteredDemographicVO2Max,
# according to this publication: https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0073182 as
# summarized here: https://www.healthline.com/health/vo2-max#increasing-vo%E2%82%82-max

def get_cardio_category(gender, age, vo2max):
    if pd.isna(gender):
        return np.nan
    if gender == "MALE":
        if age == "<30":
            if vo2max >= 51.1:
                return "Superior/Excellent"
            elif vo2max >= 41.7:
                return "Fair/Good"
            else:
                return "Poor"
        else:
            if vo2max >= 48.3:
                return "Superior/Excellent"
            elif vo2max >= 40.5:
                return "Fair/Good"
            else:
                return "Poor"
    else:
        if age == "<30":
            if vo2max >= 43.9:
                return "Superior/Excellent"
            elif vo2max >= 36.1:
                return "Fair/Good"
            else:
                return "Poor"
        else:
            if vo2max >= 42.4:
                return "Superior/Excellent"
            elif vo2max >= 34.4:
                return "Fair/Good"
            else:
                return "Poor"
