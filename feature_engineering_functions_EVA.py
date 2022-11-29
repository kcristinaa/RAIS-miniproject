# Creates 2 columns that represent if a user has tracked at least once its spo2 or eda
def use_EDA_SpO2_ECG(df):
    df['spo2_tracking'] = ""
    df['EDA_tracking'] = ""
    df['ECG_tracking'] = ""
    users = list(df['id'].unique())

    for user in users:
        user_df = df.loc[df['id'] == user]
        #spo2
        if user_df['spo2'].isnull().sum() == len(user_df):
            df.loc[df['id'] == user, 'spo2_tracking'] = 0
        else:
            df.loc[df['id'] == user, 'spo2_tracking'] = 1
        #EDA
        if user_df['scl_avg'].isnull().sum() == len(user_df):
            df.loc[df['id'] == user, 'EDA_tracking'] = 0
        else:
            df.loc[df['id'] == user, 'EDA_tracking'] = 1
        #ECG
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
