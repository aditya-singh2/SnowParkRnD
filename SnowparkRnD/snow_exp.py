#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import logging, sys, os
from snowflake.snowpark.session import Session
from snowflake.ml.registry.registry import Registry

CONNECTION_PARAMETERS = {
    "account": "ug94937.us-east4.gcp",
    "user":"ADITYASINGH",
    "password": os.environ.get('SF_Password'),
    "role": "ADITYASINGH",
    "database": "FIRST_DB",
    "warehouse": "FOSFOR_INSIGHT_WH",
    "schema": "PUBLIC",
}

exp_data = os.getenv("EXPERIMENT_DETAILS")

def create_stage(session, stage_name="demo"):
    try:
        session.sql(f"create or replace stage {stage_name}").collect()
        return f"@{stage_name}"
    except Exception as ex:
        print("Error while creating snowflake session", ex)
        raise ex

def get_session():
    """
    Method creates snowflake session object.
    :return:
    """
    try:
        return Session.builder.configs(CONNECTION_PARAMETERS).create()
    except Exception as ex:
        print("Error while creating snowflake session", ex)
        raise ex


# Stored Procedure
def train_ml_models(session: Session, exp_data: str) -> list:
    from snowflake.ml.modeling.pipeline import Pipeline
    from snowflake.ml.modeling.preprocessing import MinMaxScaler, OrdinalEncoder
    from snowflake.ml.modeling.metrics import mean_squared_error, mean_absolute_error, r2_score
    from snowflake.ml.modeling.xgboost import XGBRegressor
    import importlib, os, json
    # from snowflake.snowpark import Session, FileOperation
    
    # Experiment details
    exp_details=json.loads(exp_data)
    
    # Read dataset, Random split
    df_train, df_test = session.table(exp_details.get("dataset")).drop('ROW').random_split(weights=[0.9, 0.1], seed=0)
    features = df_train.columns
    features.remove(exp_details.get("target_column"))
    
    # get features
    data_schema = session.sql(f"DESCRIBE TABLE {exp_details.get('dataset')}").collect()
    categorical_types = ['VARCHAR','CHAR','STRING','TEXT','BOOL']
    categorical_features = []
    for row in data_schema:
        for typ in categorical_types:
            if typ in row['type'] and row['name']!=exp_details.get("target_column"):
                categorical_features.append(row['name'])
                break
    numerical_features = list(set(features) - set(categorical_features))
    categorical_features_oe = list(map(lambda a: a+'_OE', categorical_features))
    print("numerical_features: ", numerical_features)
    print("categorical_features_oe: ", categorical_features_oe)
    
    
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
    # dynamically import selected algorithms
    for algorithm, hyperparam in exp_details.get("algo_details").items():
        algorithm = algorithm.rsplit('.', 1)
        module = importlib.import_module(algorithm[0])
        print(algorithm[1])
        attr = getattr(module, algorithm[1])
        
        pipe = Pipeline(steps=steps+[("algorithm", attr(input_cols=categorical_features_oe+numerical_features
                                              , label_cols=[exp_details.get("target_column")]
                                              , output_cols=[f'PREDICTIONS_{algorithm[1]}'.upper()]))]
               )

        # Fit the pipeline
        model = pipe.fit(df_train)
         
        # Test the model
        df_test_pred = model.predict(df_test)
        
        # metrices
        mse = mean_squared_error(df=df_test_pred, y_true_col_names=exp_details.get("target_column"), y_pred_col_names=f'PREDICTIONS_{algorithm[1]}'.upper())
        mae = mean_absolute_error(df=df_test_pred, y_true_col_names=exp_details.get("target_column"), y_pred_col_names=f'PREDICTIONS_{algorithm[1]}'.upper())
        r2 = r2_score(df=df_test_pred, y_true_col_name=exp_details.get("target_column"), y_pred_col_name=f'PREDICTIONS_{algorithm[1]}'.upper())
        print("Execution Completed")
        print(f'{algorithm[1]} MSE: {mse}')
        print(f'{algorithm[1]} MAE: {mae}')
        print(f'{algorithm[1]} R2: {r2}')
        

        # LOG MODEL INTO SNOWFLAKE REGISTRY
        from snowflake.ml.registry.registry import Registry
        reg = Registry(session=session)
        # Log the model
        model_name = f"expname_{algorithm}"
        try:
            mv = reg.log_model(model=model,
                               model_name=exp_details.get("name", "sample_experiment")+"_"+algorithm,
                               comment="test",
                               version_name="run1",
                               python_version="3.9.19",
                               conda_dependencies=["scikit-learn==1.3.2"],
                               metrics={"model_metrics": {"MSE": mse, "MAE": mae, "r2": r2}, "project_id": "0001", "type": "EXP"})
        except Exception as ex:
            return ex
    return [{"EXP_NAME":exp_details.get("name", "sample_experiment"),
             "Version":"Run1",
             "matrices":{"model_metrics": {"MSE": mse, "MAE": mae, "r2": r2}, "project_id": "0001", "type": "EXP"},
             "Alogirthm_Type":"Regression",
             "Alogithms": list(exp_details.get("algo_details").keys()),
             "RUN_STATUS":"SUCCESS",
             "registry_exp_name":""}]


# Initilization
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
print("Creating Snowflake Session object...")
session = get_session()
stage = create_stage(session)
print("Session has been created !")

print("Creating stored procedure...")
session.sproc.register(func=train_ml_models,
                       name="train_ml_models",
                       packages=["snowflake-snowpark-python", "snowflake-ml-python"],
                       isPermanant=False,
                       stage_location=stage,
                       replace=True)
print("Stored procedure has been created successfully!")

print("Executing Stored Procedure")
procedure_response = session.call("train_ml_models", exp_data)
print("Stored Procedure Executed Successfully !")
print(procedure_response)

#Log in mlflow
print("Logging in mlflow completed !")

