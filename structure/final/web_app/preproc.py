import pandas as pd
import numpy as np
import utils as u
import utils2 as p
import streamlit as st
import datetime

# Scaler
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler
)
# Train-Test Split
from sklearn.model_selection import train_test_split
# Models
import models as mod
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier 


def preproc_(path):
    
    user_input = pd.read_csv(path, index_col='Claim Identifier')

    st.text("Processing your input data...")

    user_input['Age at Injury'] = 2024 - user_input['Birth Year']

    # List of columns to convert to datetime
    date_columns = ['Accident Date', 'Assembly Date', 'C-2 Date', 'C-3 Date', 'First Hearing Date']

    # Apply pd.to_datetime() to each column in the list for both df and user_input
    for col in date_columns:
        user_input[col] = pd.to_datetime(user_input[col], errors='coerce')

    mapping = {
    '5D. SPECIAL FUND - UNKNOWN': '5. SPECIAL FUND OR UNKNOWN',
    '5A. SPECIAL FUND - CONS. COMM. (SECT. 25-A)': '5. SPECIAL FUND OR UNKNOWN',
    '5C. SPECIAL FUND - POI CARRIER WCB MENANDS': '5. SPECIAL FUND OR UNKNOWN',
    'UNKNOWN': '5. SPECIAL FUND OR UNKNOWN'}  

    user_input['Carrier Type'] = user_input['Carrier Type'].replace(mapping)

    user_input['Gender Enc'] = user_input['Gender'].replace({'M': 0, 'F': 1, 'U/X': 2})


    for column in user_input.columns:
        # Check if the column is a datetime type
        if pd.api.types.is_datetime64_any_dtype(user_input[column]) and column not in ['C-3 Date', 'First Hearing Date']:
            user_input[f'{column} Year'] = user_input[column].dt.year
            user_input[f'{column} Month'] = user_input[column].dt.month
            user_input[f'{column} Day'] = user_input[column].dt.day
            user_input[f'{column} Day of Week'] = user_input[column].dt.weekday 

    user_input['Accident to Assembly Time'] = (user_input['Assembly Date'] - user_input['Accident Date']).dt.days

    user_input['Assembly to C-2 Time'] = (user_input['Assembly Date'] - user_input['C-2 Date']).dt.days

    user_input['Accident to C-2 Time'] = (user_input['C-2 Date'] - user_input['Accident Date']).dt.days

    user_input['WCIO Part Of Body Code'] = user_input['WCIO Part Of Body Code'].abs()

    columns_to_join = [
    'WCIO Cause of Injury Code',
    'WCIO Nature of Injury Code',
    'WCIO Part Of Body Code'
    ]

    user_input[columns_to_join] = user_input[columns_to_join].fillna(0).astype(int)

    user_input['WCIO Codes'] = user_input[columns_to_join].astype(str).agg(''.join, axis=1).astype(int)

    user_input['Insurance'] = user_input['Carrier Name'].str.contains('ins', case=False, na=False).astype(int)

    # Create a new column 'Zip Code Valid' to flag the validity of the 'Zip Code' field
    user_input['Zip Code Valid'] = user_input['Zip Code'].apply(
        lambda x: 2 if pd.isna(x)          
        else (1 if not str(x).isnumeric()   
            else 0)                       
    )

    user_input['Industry Sector'] = user_input['Industry Code Description'].apply(u.group_industry)

    bins = [-1, 17, 64, 117]
    labels = [0, 1, 2]

    user_input['Age Group'] = pd.cut(user_input['Age at Injury'], 
                                    bins=bins, labels=labels, right=True)
    
    drop = ['Accident Date', 'Assembly Date',
        'C-2 Date', 'Zip Code']
    

    user_input.drop(columns = drop, axis = 1, inplace = True)

    # Reading the train data
    df= pd.read_csv("/Users/anaca/Documents/GitHub/machine_learning/web_app/train_data_EDA.csv",
                    index_col='Claim Identifier')
    
    # Split the DataFrame into features (X) and target variable (y)
    X = df.drop('Claim Injury Type', axis=1) 
    y = df['Claim Injury Type']  

    # Split the dataset into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42,
                                                    stratify = y) 

    ## Feature Engineering
    #Alternative Dispute Resolution
    
    X_train['Alternative Dispute Resolution Enc'] = X_train['Alternative Dispute Resolution'].replace({'N': 0, 'Y': 1, 'U': 1})
    X_val['Alternative Dispute Resolution Enc'] = X_val['Alternative Dispute Resolution'].replace({'N': 0, 'Y': 1, 'U': 1}) 
    user_input['Alternative Dispute Resolution Enc'] = user_input['Alternative Dispute Resolution'].replace({'No': 0, 'Yes': 1, 'U': 1})
    
    #Attorney/Representative
    X_train['Attorney/Representative Enc'] = X_train['Attorney/Representative'].replace({'N': 0, 'Y': 1})
    X_val['Attorney/Representative Enc'] = X_val['Attorney/Representative'].replace({'N': 0, 'Y': 1})
    user_input['Attorney/Representative Enc'] = user_input['Attorney/Representative'].replace({'No': 0, 'Yes': 1})
    
    #Carrier Name
    train_carriers = set(X_train['Carrier Name'].unique())
    user_input_carriers = set(user_input['Carrier Name'].unique())

    common_categories = train_carriers.intersection(user_input_carriers)
    
    common_category_map = {category: idx + 1 for idx, 
                       category in enumerate(common_categories)}
    
    X_train['Carrier Name'] = X_train['Carrier Name'].map(common_category_map).fillna(0).astype(int)
    X_val['Carrier Name'] = X_val['Carrier Name'].map(common_category_map).fillna(0).astype(int)
    user_input['Carrier Name'] = user_input['Carrier Name'].map(common_category_map).fillna(0).astype(int)
    
    X_train, X_val, user_input = p.encode(X_train, X_val, user_input, 'Carrier Name', 'count')
    
    #Carrier Type
    X_train, X_val, user_input = p.encode(X_train, X_val, user_input, 'Carrier Type', 'count')
    #Count Enc
    X_train, X_val, user_input = p.encode(X_train, X_val, user_input, 'Carrier Type', 'count')
    #One-Hot-Enc
    X_train, X_val, user_input = p.encode(X_train, X_val, user_input, 'Carrier Type', 'OHE')
    
    #County of Injury
    X_train, X_val, user_input = p.encode(X_train, X_val, user_input, 'County of Injury', 'count')
    
    #Covid-19 Indicator
    X_train['COVID-19 Indicator Enc'] = X_train['COVID-19 Indicator'].replace({'N': 0, 'Y': 1})
    X_val['COVID-19 Indicator Enc'] = X_val['COVID-19 Indicator'].replace({'N': 0, 'Y': 1})
    user_input['COVID-19 Indicator Enc'] = user_input['COVID-19 Indicator'].replace({'No': 0, 'Yes': 1})
    
    #District Name 
    X_train, X_val, user_input = p.encode(X_train, X_val, user_input, 'District Name', 'count')
    
    #Gender 
    X_train, X_val, user_input = p.encode(X_train, X_val, user_input, 'Gender', 'OHE')
    
    #Medical Fee Region
    X_train, X_val, user_input = p.encode(X_train, X_val, user_input, 'Medical Fee Region', 'count')
    
    #Industry Sector 
    X_train, X_val, user_input = p.encode(X_train, X_val, user_input, 'Industry Sector', 'count')
    
    #Remove Enc Variables
    drop = ['Alternative Dispute Resolution', 'Attorney/Representative', 'Carrier Type', 'County of Injury',
        'COVID-19 Indicator', 'District Name', 'Gender', 'Carrier Name',
        'Medical Fee Region', 'Industry Sector']
    
    X_train.drop(columns = drop, axis = 1, inplace = True)
    X_val.drop(columns = drop, axis = 1, inplace = True)
    user_input.drop(columns = drop, axis = 1, inplace = True)
    
    ## Missing Values
    # C_3 Date 
    X_train['C-3 Date Binary'] = X_train['C-3 Date'].notna().astype(int)
    X_val['C-3 Date Binary'] = X_val['C-3 Date'].notna().astype(int)
    user_input['C-3 Date Binary'] = user_input['C-3 Date'].notna().astype(int)
    
    # First Hearing Date
    X_train['First Hearing Date Binary'] = X_train['First Hearing Date'].notna().astype(int)
    X_val['First Hearing Date Binary'] = X_val['First Hearing Date'].notna().astype(int)
    user_input['First Hearing Date Binary'] = user_input['First Hearing Date'].notna().astype(int)

    # Remove Transformed date
    drop = ['C-3 Date', 'First Hearing Date']

    X_train.drop(columns = drop, axis = 1, inplace = True)
    X_val.drop(columns = drop, axis = 1, inplace = True)
    user_input.drop(columns = drop, axis = 1, inplace = True)

    # Filling MV
    # IME-4 Count
    X_train['IME-4 Count'] = X_train['IME-4 Count'].fillna(0)
    X_val['IME-4 Count'] = X_val['IME-4 Count'].fillna(0)
    user_input['IME-4 Count'] = user_input['IME-4 Count'].fillna(0)

    # Industry Code
    X_train['Industry Code'] = X_train['Industry Code'].fillna(0)
    X_val['Industry Code'] = X_val['Industry Code'].fillna(0)
    user_input['Industry Code'] = user_input['Industry Code'].fillna(0)

    # Accident Date & C-2 Date
    p.fill_dates(X_train, [X_val, user_input], 'Accident Date')
    p.fill_dates(X_train, [X_val, user_input], 'C-2 Date')

    p.fill_dow([X_train, X_val, user_input], 'Accident Date')
    p.fill_dow([X_train, X_val, user_input], 'C-2 Date')
    
    # Time Between
    X_train = p.fill_missing_times(X_train, ['Accident to Assembly Time', 
                             'Assembly to C-2 Time',
                             'Accident to C-2 Time'])

    X_val = p.fill_missing_times(X_val, ['Accident to Assembly Time', 
                             'Assembly to C-2 Time',
                             'Accident to C-2 Time'])

    user_input = p.fill_missing_times(user_input, ['Accident to Assembly Time', 
                             'Assembly to C-2 Time',
                             'Accident to C-2 Time'])

    # Birth Year
    p.fill_birth_year([X_train, X_val, user_input])

    ## Scalling
    # Variable Type Split
    num = ['Age at Injury', 'Average Weekly Wage', 'Birth Year',
       'IME-4 Count', 'Number of Dependents', 'Accident Date Year',
       'Accident Date Month', 'Accident Date Day', 
       'Assembly Date Year', 'Assembly Date Month', 
       'Assembly Date Day', 'C-2 Date Year', 'C-2 Date Month',
       'C-2 Date Day', 'Accident to Assembly Time',
       'Assembly to C-2 Time', 'Accident to C-2 Time']
      # 'Wage to Age Ratio', 'Average Weekly Wage Sqrt',
      # 'IME-4 Count Log', 'IME-4 Count Double Log']

 
    categ = [var for var in X_train.columns if var not in num]

    categ_count_encoding = ['Carrier Name Enc', 'Carrier Type Enc',
                        'County of Injury Enc', 'District Name Enc',
                        'Medical Fee Region Enc', 'Industry Sector Enc']


    categ_label_bin = [var for var in X_train.columns if var
                   in categ and var not in categ_count_encoding]


    # Scaling
    num_count_enc = num + categ_count_encoding
    robust = RobustScaler()

    # Scaling the numerical features in the training set using RobustScaler
    X_train_num_count_enc_RS = robust.fit_transform(X_train[num_count_enc])
    X_train_num_count_enc_RS = pd.DataFrame(X_train_num_count_enc_RS, columns=num_count_enc, index=X_train.index)

    # Scaling the numerical features in the validation set using the fitted RobustScaler
    X_val_num_count_enc_RS = robust.transform(X_val[num_count_enc])
    X_val_num_count_enc_RS = pd.DataFrame(X_val_num_count_enc_RS, columns=num_count_enc, index=X_val.index)

    # Scaling the numerical features in the user_input set using the same fitted RobustScaler
    user_input_num_count_enc_RS = robust.transform(user_input[num_count_enc])
    user_input_num_count_enc_RS = pd.DataFrame(user_input_num_count_enc_RS, columns=num_count_enc, index=user_input.index)

    # Joining Features
    X_train_RS = pd.concat([X_train_num_count_enc_RS, 
                        X_train[categ_label_bin]], axis=1)
    X_val_RS = pd.concat([X_val_num_count_enc_RS, 
                      X_val[categ_label_bin]], axis=1)
    user_input_RS = pd.concat([user_input_num_count_enc_RS, 
                     user_input[categ_label_bin]], axis=1)

    # Countinuing MV

    #Average Weekly Wage
    p.ball_tree_impute([X_train_RS, X_val_RS], 
                   'Average Weekly Wage')
    
    ## Outliers 

    #Avg Weekly Wage
    # Square Root 
    X_train_RS['Average Weekly Wage Sqrt'] = np.sqrt(X_train_RS['Average Weekly Wage'])

    X_val_RS['Average Weekly Wage Sqrt'] = np.sqrt(X_val_RS['Average Weekly Wage'])

    user_input_RS['Average Weekly Wage Sqrt'] = np.sqrt(user_input_RS['Average Weekly Wage'])

    # Winsorization
    upper_limit = X_train['Average Weekly Wage'].quantile(0.99)
    lower_limit = X_train['Average Weekly Wage'].quantile(0.01)

    X_train_RS['Average Weekly Wage'] = X_train_RS['Average Weekly Wage'].clip(lower = lower_limit
                                                                  , upper=upper_limit)
    # Birth Year

    # IME-4 Count
    X_train_RS['IME-4 Count Log'] = np.log1p(X_train_RS['IME-4 Count'])
    X_train_RS['IME-4 Count Double Log'] = np.log1p(X_train_RS['IME-4 Count Log'])

    X_val_RS['IME-4 Count Log'] = np.log1p(X_val_RS['IME-4 Count'])
    X_val_RS['IME-4 Count Double Log'] = np.log1p(X_val_RS['IME-4 Count Log'])

    user_input_RS['IME-4 Count Log'] = np.log1p(user_input_RS['IME-4 Count'])
    user_input_RS['IME-4 Count Double Log'] = np.log1p(user_input_RS['IME-4 Count Log'])

    # Accident Date Year


    # C-2 Date Year
    

    #Alternative Dispute Resolution Enc --> it will prob be dropeed
    # Ensuring y_train as the same indices as Y_train
    y_train = y_train[X_train_RS.index]

    ## Modeling 
    st.write('The model is being trained...')
    st.write('This will take a few (3-4) minutes')

    ## Modeling
    model = XGBClassifier()
    model.fit(X_train_RS, y_train)
    train_pred = model.predict(X_train_RS)
    val_pred = model.predict(X_val_RS)

    ## Final Predictions

    test_filtered = user_input_RS

    test_filtered['Claim Injury Type'] = model.predict(test_filtered)

    #Map Predictions to Original Values

    label_mapping = {
        0: "1. CANCELLED",
        1: "2. NON-COMP",
        2: "3. MED ONLY",
        3: "4. TEMPORARY",
        4: "5. PPD SCH LOSS",
        5: "6. PPD NSL",
        6: "7. PTD",
        7: "8. DEATH"   
    }

    test_filtered['Claim Injury Type'] = test_filtered['Claim Injury Type'].replace(label_mapping)
    predicted_label = test_filtered['Claim Injury Type'].iloc[0]
    
    return predicted_label





