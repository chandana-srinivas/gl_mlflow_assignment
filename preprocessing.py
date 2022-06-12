import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
DATA_DIR = './data'
DATASET_PATH = os.path.join(DATA_DIR,'Titanic+Data+Set.csv')
PROCESSED_DATASET_PATH = os.path.join(DATA_DIR,'Titanic+Data+Set+Preprocess.csv')
def preprocess(df):
    # Drop some unimportant columns
    df.drop(['PassengerId','Name','Ticket','Cabin'], axis=1, inplace=True)

    # Fill unknown 
    df['Embarked'] = df['Embarked'].fillna('Unknown')

    # Impute 

    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Encode
    for col in df.columns:
        if df[col].dtype == 'object':
            le = LabelEncoder()
            ohe = OneHotEncoder()

            le.fit(df[col])
            df[col] = le.transform(df[col])
            new_cols = ['{}_{}'.format(col,c) for c in le.classes_]
            ohe.fit(df[[col]])
            ohe_array = ohe.transform(df[[col]]).toarray().astype(int)
            # Add one hot encoded columns
            df[new_cols] = pd.DataFrame(ohe_array)
            # Remove original column
            df.pop(col)
            #encode data


            
if __name__ == '__main__':
    data = pd.read_csv(DATASET_PATH)

    print(data.describe())
    print(data.head())

    null_values = data.isnull().sum()
    print(null_values)
    
    ## Do preprocessing
    preprocess(data)

    # print(data.describe())
    # print(data.head())
    null_values = data.isnull().sum()
    print(null_values)

    data.to_csv(PROCESSED_DATASET_PATH,index=False)
    print("Preprocessing Done, Preprocessed Data is written to {}".format(PROCESSED_DATASET_PATH))


    data_df_attr = data.iloc[:, 0:10]

    axes = pd.plotting.scatter_matrix(data_df_attr)
    #plt.tight_layout()
    #plt.savefig('d:\greatlakes\mpg_pairpanel.png')

    sns.pairplot(data_df_attr, diag_kind='kde')   # to plot density curve instead of histogram

    sns.pairplot(data_df_attr)
    plt.show()
