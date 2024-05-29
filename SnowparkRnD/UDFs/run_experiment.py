#!/usr/bin/env python
# coding: utf-8

# In[34]:


def run_exp(algos, dataset, target):    
    import os, importlib
    from snowflake.snowpark import Session
    from snowflake.ml.modeling.pipeline import Pipeline
    from snowflake.ml.modeling.preprocessing import MinMaxScaler, OrdinalEncoder
    from snowflake.ml.modeling.metrics import mean_squared_error, mean_absolute_error, r2_score
    from snowflake.snowpark.types import StructType, StructField, IntegerType, StringType
    from snowflake.ml.modeling.compose import ColumnTransformer
    from snowflake.snowpark import Session, FileOperation
    
#     connection_parameters = {
#     "account": "ug94937.us-east4.gcp",
#     "user": "ADITYASINGH",
#     "password": os.environ.get('SF_Password'),
#     "role": "ADITYASINGH",  # optional
#     "warehouse": "FOSFOR_INSIGHT_WH",  # optional
#     "database": "FIRST_DB",  # optional
#     "schema": "PUBLIC",  # optional
#     }
#     session = Session.builder.configs(connection_parameters).create()
    
    session = get_active_session() 
    
    # Read dataset
    df_train, df_test = session.table(dataset).drop('ROW').random_split(weights=[0.9, 0.1], seed=0)
    print(df_train.show())
    features = df_train.columns
    features.remove(target)
    
    # generating feature names
    data_schema = session.sql(f"DESCRIBE TABLE {dataset}").collect()
    categorical_types = ['VARCHAR','CHAR','STRING','TEXT','BOOL']
    categorical_features = []
    for row in data_schema:
        for typ in categorical_types:
            if typ in row['type'] and row['name']!=target:
                categorical_features.append(row['name'])
                break
    numerical_features = list(set(features) - set(categorical_features))
    categorical_features_oe = list(map(lambda a: a+'_OE', categorical_features))
    print("numerical_features: ", numerical_features)
    print("categorical_features_oe: ", categorical_features_oe)
    
#     #Numerical pipeline
#     numeric_transform = Pipeline(steps=[
#         ("scaler", MinMaxScaler(output_cols=numerical_features))
#     ]
#     )
    
#     #Categorical pipeline
#     categoric_transform = Pipeline(steps=[
#         ("ord", OrdinalEncoder(output_cols=categorical_features_oe))
#     ]
#     )
    
#     #preprocessor
#     preprocessor = ColumnTransformer(
#         output_cols=categorical_features_oe+numerical_features+[target],
#         transformers=[
#             ('num', numeric_transform, numerical_features),
#             ('cat', categoric_transform, categorical_features)
#         ],
#         remainder='passthrough'
#     )
    
    
    #pipeline steps 
    categorical_pp = {
        'ord': OrdinalEncoder(input_cols=categorical_features, output_cols=categorical_features_oe) 
    }
    numerical_pp = {
        'scaler': MinMaxScaler(input_cols=numerical_features, output_cols=numerical_features)
    }
    steps = [(key, categorical_pp[key]) for key in categorical_pp if categorical_features!=[]] + \
    [(key, numerical_pp[key]) for key in numerical_pp if numerical_features!=[]]

    
    # Define a pipeline that does the preprocessing and training of 
    # dynamically generate list of selected algorithms for imports
    df_all_pred = None
    for algorithm in algos:
        algorithm = algorithm.rsplit('.', 1)
        module = importlib.import_module(algorithm[0])
        print(algorithm[1])
        attr = getattr(module, algorithm[1])
        
        pipe = Pipeline(steps=steps+[("algorithm", attr(input_cols=categorical_features_oe+numerical_features
                                              , label_cols=[target]
                                              , output_cols=[f'PREDICTIONS_{algorithm[1]}'.upper()]))]
               )

        # Fit the pipeline
        xgb_model = pipe.fit(df_train)
         
        # Test the model
        df_test_pred = xgb_model.predict(df_test)
        
        #combining predictions
        if df_all_pred is None:
            df_all_pred = df_test_pred.select(df_test_pred[f'PREDICTIONS_{algorithm[1]}'.upper()])
        else:
            df_all_pred = df_all_pred.join(df_test_pred.select(df_test_pred[f'PREDICTIONS_{algorithm[1]}'.upper()]))
            
        # metrices
        mse = mean_squared_error(df=df_test_pred, y_true_col_names=target, y_pred_col_names=f'PREDICTIONS_{algorithm[1]}'.upper())
        mae = mean_absolute_error(df=df_test_pred, y_true_col_names=target, y_pred_col_names=f'PREDICTIONS_{algorithm[1]}'.upper())
        r2 = r2_score(df=df_test_pred, y_true_col_name=target, y_pred_col_name=f'PREDICTIONS_{algorithm[1]}'.upper())
        print(f'{algorithm[1]} MSE: {mse}')
        print(f'{algorithm[1]} MAE: {mae}')
        print(f'{algorithm[1]} R2: {r2}')
        
    return df_all_pred

