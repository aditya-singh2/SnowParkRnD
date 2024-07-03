import logging, sys, os
from snowflake.snowpark.session import Session
from snowflake.ml.registry.registry import Registry


def create_stage(session, stage_name="demo"):
    try:
        session.sql(f"create or replace stage {stage_name}").collect()
        return f"@{stage_name}"
    except Exception as ex:
        print("Error while creating snowflake session", ex)
        raise ex

def get_session(dataset, project_id):
    """
    Method creates snowflake session object.
    :return:
    """
    try:
        conn = get_conn_details_from_ds_name(dataset, project_id)
        region=conn["params"]["READER"]["region"] if conn["params"]["READER"]["cloudPlatform"] is None \
                    else conn["params"]["READER"]["region"]+"."+conn["params"]["READER"]["cloudPlatform"]
        account = conn['params']['READER']['accountId'] if region is None \
                    else conn['params']['READER']['accountId']+"."+region
        CONNECTION_PARAMETERS = {
            "account": account,
            "user":conn['params']['READER']['user'],
            "password": conn['params']['READER']['password'],
            "role": conn['params']['READER']['role'],
            "database": conn['params']['READER']['database'],
            "warehouse": conn['params']['READER']['wareHouse'],
            "schema": conn['params']['READER']['schema']
        }
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
    import logging
    
     # variable for holding logs
    logs = []
    
    # function for accumulating logs
    def log_message(level: str, message: str):
        logs.append(f"{level}: {message}")
    
    log_message("INFO","Starting Experiment Recipe Execution")
    
    # Experiment details
    exp_details=json.loads(exp_data)
    
    # Read dataset, Random split
    log_message("INFO","Identifing dataset features")
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
    log_message("INFO",f"numerical_features:  {numerical_features}")
    log_message("INFO",f"categorical_features_oe: {categorical_features_oe}")
    
    
    #pipeline steps 
    log_message("INFO","Setting up preprocessing pipeline based on dataset")
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
        log_message("INFO",f"Running Algorithm {algorithm[1]}")
        attr = getattr(module, algorithm[1])
        
        pipe = Pipeline(steps=steps+[("algorithm", attr(input_cols=categorical_features_oe+numerical_features
                                              , label_cols=[exp_details.get("target_column")]
                                              , output_cols=[f'PREDICTIONS_{algorithm[1]}'.upper()]))]
               )

        # Fit the pipeline
        log_message("INFO",f"Running model pipeline {algorithm[1]}")
        model = pipe.fit(df_train)
         
        # Test the model
        log_message("INFO","Running prediction on model with test dataset")
        df_test_pred = model.predict(df_test)
        
        # metrices
        log_message("INFO","Generating Metrices")
        accuracy = accuracy_score(df=df_test_pred, y_true_col_names=exp_details.get("target_column"), y_pred_col_names=f'PREDICTIONS_{algorithm[1]}'.upper())
        f1_score = f1_score(df=df_test_pred, y_true_col_names=exp_details.get("target_column"), y_pred_col_names=f'PREDICTIONS_{algorithm[1]}'.upper())
        recall_score = recall_score(df=df_test_pred, y_true_col_names=exp_details.get("target_column"), y_pred_col_names=f'PREDICTIONS_{algorithm[1]}'.upper())
        precision_score = precision_score(df=df_test_pred, y_true_col_names=exp_details.get("target_column"), y_pred_col_names=f'PREDICTIONS_{algorithm[1]}'.upper())
        roc_auc_score = roc_auc_score(df=df_test_pred, y_true_col_names=exp_details.get("target_column"), y_score_col_names=f'PREDICTIONS_{algorithm[1]}'.upper())
        log_message("INFO","Metrices generation completed!!!")
        log_message("INFO",f'{algorithm[1]} accuracy: {accuracy}')
        log_message("INFO",f'{algorithm[1]} f1_score: {f1_score}')
        log_message("INFO",f'{algorithm[1]} recall_score: {recall_score}')
        log_message("INFO",f'{algorithm[1]} precision_score: {precision_score}')
        log_message("INFO",f'{algorithm[1]} roc_auc_score: {roc_auc_score}')
        

        # LOG MODEL INTO SNOWFLAKE REGISTRY
        from snowflake.ml.registry.registry import Registry
        reg = Registry(session=session)
        # Log the model
        log_message("INFO","Started: Registering model on snowflake")
        try:
            mv = reg.log_model(model=model,
                               model_name=exp_details.get("name", "sample_experiment")+"_"+algorithm[1],
                               comment="test",
                               version_name="run1",
                               python_version="3.9.19",
                               conda_dependencies=["scikit-learn==1.3.2"],
                               metrics=[{"model_metrics": {"roc_auc_score": roc_auc_score, "precision_score": precision_score, "f1_score": f1_score, "recall_score": recall_score, "accuracy_score": accuracy}, "project_id": "0001", "type": "EXP"}])
            log_message("INFO","Registeration of model completed!!!")
        except Exception as ex:
            key = 'Processing aborted due to error 370001' 
            if key in str(ex):
                log_message("INFO","Registeration of model completed!!!")
                pass
            else:
                log_message("ERROR","Exception Occured while registering model")
                return str(ex).split('?')
    return [{"Execution Logs:": "\n".join(logs),
             "EXP_NAME":exp_details.get("name", "sample_experiment"),
             "Version":"Run1",
             "matrices":{"model_metrics": {"roc_auc_score": roc_auc_score, "precision_score": precision_score, "f1_score": f1_score, "recall_score": recall_score, "accuracy_score": accuracy}, "project_id": "0001", "type": "EXP"},
             "Alogirthm_Type":"Regression",
             "Alogithms": list(exp_details.get("algo_details").keys()),
             "RUN_STATUS":"SUCCESS",
             "registry_exp_name":""}]


# Initilization
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
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

