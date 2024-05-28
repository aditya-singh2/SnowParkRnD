#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###Python file for udf function


# In[ ]:


def run_experiment(sf_pass, dataset, target):
    import os
    from snowflake.snowpark import Session
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import OneHotEncoder
    from snowflake.ml.modeling.tree import DecisionTreeClassifier
    
    connection_parameters = {
    "account": "ug94937.us-east4.gcp",
    "user": "ADITYASINGH",
    "password": sf_pass,
    "role": "ADITYASINGH",  # optional
    "warehouse": "FOSFOR_INSIGHT_WH",  # optional
    "database": "FIRST_DB",  # optional
    "schema": "PUBLIC",  # optional
    }
    
    session = Session.builder.configs(connection_parameters).create()
    session.sql_simplifier_enabled = True
    
    data = session.table(dataset)
    data = data.to_pandas()
    
    # Data Preprocessing: Validating and encoding the data if required and imputing null values.
    data = data.fillna(method='pad')  # Filling null values with the previous ones
    data = data.fillna(method='bfill')  # Filling null value with the next ones
    
    def encoding(df, target_column):
        """
        Checking whether encoding required in target and feature datasets.
        If required, then encoding them with label and one hot encoding.
        :param:
        df: input dataframe
        target_column: target column
        :returns:
        df_target: target dataframe
        le_target: target label encoder object
        df_feature: feature dataframe
        le_dict_feature: dict of feature label encoder objects
        oh_enc_feature: feature one hot encoder object
        le_column_feature: list of feature label encoder columns
        oh_column_feature: list of feature one hot encoder columns
        """
        df_target = df[[target_column]]
        le_target = None
        # Target column validation and encoding
        if df.dtypes[target_column].name in ['object', 'bool']:
            print(f"target_column is of {df.dtypes[target_column].name} datatype, encoding required.")
            le_target = LabelEncoder()
            df_target[target_column] = pd.DataFrame(le_target.fit_transform(df_target[target_column].astype(str)))
            print(f"Target column label encoded {df_target[target_column]}, object: {le_target}")

        # Feature column validation and encoding
        df_feature = df.drop(target_column, axis=1)
        non_numeric_cols = df_feature.select_dtypes(include=['object', 'bool']).columns.tolist()
        le_dict_feature = {}
        le_column_feature = []
        oh_column_feature = []
        oh_enc_feature = None
        if len(non_numeric_cols) >= 1:
            print(f"{non_numeric_cols} columns are non numeric in feature dataset, encoding required.")
            for col in non_numeric_cols:
                if df_feature[col].nunique() >= 10:
                    le_column_feature.append(col)
                else:
                    oh_column_feature.append(col)

            print(f"Columns identified to be encoded with label encoder: {le_column_feature}\n"
                  f"Columns identified to be encoded with one hot encoder: {oh_column_feature}")

            # columns to be label encoded
            if len(le_column_feature) == 0:
                df_feature = df_feature
            else:
                for col in le_column_feature:
                    le_dict_feature[col] = LabelEncoder()
                    df_feature[col] = le_dict_feature[col].fit_transform(df_feature[col].astype(str))
                    print(f"{col} column label encoded {df_feature[col]}, object: {le_dict_feature[col]}")

            # columns to be one hot encoded
            if len(oh_column_feature) == 0:
                df_feature = df_feature
            else:
                unique_combinations = pd.get_dummies(df_feature[oh_column_feature])
                unique_combinations_list = unique_combinations.columns.tolist()
                oh_enc_feature = OneHotEncoder()
                oh_encoded_array = oh_enc_feature.fit_transform(df_feature[oh_column_feature]).toarray() if len(oh_column_feature) > 1 else oh_enc_feature.fit_transform(df_feature[oh_column_feature]).toarray()
                df_oh_enc = pd.DataFrame(oh_encoded_array, columns=unique_combinations_list)
                df_feature = df_feature.drop(columns=oh_column_feature)
                df_feature = df_feature.join(df_oh_enc)
#                 print(f"new one hot encoded df: {oh_encoded_array}\n"
#                       f"one hot encoder object: {oh_enc_feature}\n")
#             print(f"final feature df created: {df_feature}")
        return df_target, le_target, df_feature, le_dict_feature, oh_enc_feature, le_column_feature, oh_column_feature
    
    df_target, le_target, df_feature, le_dict_feature, oh_enc_feature, le_column_feature, oh_column_feature = encoding(data, target)
    
    features_pandas = pd.concat([df_feature, df_target], axis=1)
    features_pandas.columns = map(str.upper, features_pandas.columns)
    features_pandas.columns = features_pandas.columns.str.replace(' ', '_')
    features_df = session.create_dataframe(features_pandas)
    
    FEATURE_COLUMNS=list(features_df.columns)
    FEATURE_COLUMNS.remove(target)
    LABEL_COLUMNS = [target]
    OUTPUT_COLUMNS = ["PREDICTION"]
    
    model = DecisionTreeClassifier(
        input_cols=FEATURE_COLUMNS,
        label_cols=LABEL_COLUMNS,
        output_cols=OUTPUT_COLUMNS
    )
    model.fit(features_df)

    # Use the model to make predictions.
    predictions = model.predict(features_df)
    return predictions[OUTPUT_COLUMNS]

