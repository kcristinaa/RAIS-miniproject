import statistics
import numpy as np
import pandas as pd
from datetime import datetime


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


def fitbit_one_hot_encoding(fitbit):

    # bmi encoding
    fitbit["bmi"] = fitbit["bmi"].apply(lambda x: 31.0 if x == '>=30' else x)
    fitbit["bmi"] = fitbit["bmi"].apply(lambda x: 18.0 if x == '<19' else x)
    fitbit["bmi"] = fitbit["bmi"].apply(lambda x: 26.0 if x == '>=25' else x)
    fitbit['bmi_category'] = fitbit.bmi.apply(lambda bmi: 'Underweight' if bmi < 18.5 else (
        'Normal' if bmi < 25 else ('Overweight' if bmi < 30 else 'Obese')))
    fitbit = fitbit.drop(columns=['bmi'])
    bmi_category = pd.get_dummies(fitbit['bmi_category'])
    fitbit = pd.concat([fitbit, bmi_category], axis=1)
    fitbit.drop(['bmi_category'], axis=1, inplace=True)

    # age encoding
    age = pd.get_dummies(fitbit['age'])
    fitbit = pd.concat([fitbit, age], axis=1)
    fitbit.drop(['age'], axis=1, inplace=True)
    fitbit = fitbit.rename(columns={"<30": "below_30s", ">=30": "above_30s"})

    # mindfulness session encoding
    mind = pd.get_dummies(fitbit['mindfulness_session'])
    fitbit = pd.concat([fitbit, mind], axis=1)
    fitbit.drop(['mindfulness_session'], axis=1, inplace=True)

    # gender encoding
    gender = pd.get_dummies(fitbit['gender'])
    fitbit = pd.concat([fitbit, gender], axis=1)
    fitbit.drop(['gender'], axis=1, inplace=True)

    # activity type encoding
    s = fitbit['activityType']
    dum = pd.get_dummies(s.apply(pd.Series).stack()).sum(level=0)
    df = pd.concat([s, dum], axis=1)
    fitbit = pd.concat([fitbit, df], axis=1)
    fitbit = fitbit.drop(columns='activityType')

    # badgeType encoding - actually is deletion because it has 92% missing values
    fitbit = fitbit.drop(columns='badgeType')

    return fitbit


# --------------------------------------------------------------------------- #

# Weekly frequency in fitbit dataframe regarding the corresponding survey

def weekly_fitbit_frequency(survey, fitbit, users):  # survey is stai or panas dataframe

    column_list = list(survey.columns)
    fitbit_columns = list(survey.columns)
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

    return fitbit_survey


# --------------------------------------------------------------------------- #

# Pre-processing actions after merging the fitbit dataframe with a sema/survey

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

    data = data.drop(columns=['date', 'year', 'month', 'weekday', 'week', 'day'])

    return data


def post_preprocessing(df, isSema):

    # Because of way too many missing values in spo2 (80%) and scl_avg (95%), I drop these 2 columns
    df = df.drop(columns=['spo2', 'scl_avg'])

    # Drop duplicates
    if not (isSema):
        df = df.loc[df.astype(str).drop_duplicates().index]

    # Remove id
    df = df.drop(columns=['id'])

    # Day-related feature extraction
    df = date_engineering(df)

    # Replace outliers
    df = df.mask(df.sub(df.mean()).div(df.std()).abs().gt(2))

    # Replace NaN values
    df = df.apply(lambda x: x.fillna(x.median()), axis=0)

    return df

# --------------------------------------------------------------------------- #
