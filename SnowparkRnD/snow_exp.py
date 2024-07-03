import logging, sys, os
from snowflake.ml.registry.registry import Registry
from fosforio.manager import get_conn_details_from_ds_name
from snowflake.snowpark.session import Session
import json

CONNECTION_PARAMETERS = {
    "account": "ug94937.us-east4.gcp",
    "user":"ADITYASINGH",
    "password": os.environ.get('SF_Password'),
    "role": "ADITYASINGH",
    "database": "FIRST_DB",
    "warehouse": "FOSFOR_INSIGHT_WH",
    "schema": "PUBLIC",
}


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
#         conn = get_conn_details_from_ds_name(dataset, project_id)
#         print(conn)
#         region=conn["params"]["READER"]["region"] if conn["params"]["READER"]["cloudPlatform"] is None \
#                     else conn["params"]["READER"]["region"]+"."+conn["params"]["READER"]["cloudPlatform"]
#         account = conn['params']['READER']['accountId'] if region is None \
#                     else conn['params']['READER']['accountId']+"."+region
#         CONNECTION_PARAMETERS = {
#             "account": account,
#             "user":conn['params']['READER']['user'],
#             "password": conn['params']['READER']['password'],
#             "role": conn['params']['READER']['role'],
#             "database": conn['params']['READER']['database'],
#             "warehouse": conn['params']['READER']['wareHouse'],
#             "schema": conn['params']['READER']['schema']
#         }
        return Session.builder.configs(CONNECTION_PARAMETERS).create()
    except Exception as ex:
        print("Error while creating snowflake session", ex)
        raise ex


# Method to be registered as SF Stored Procedure 
def run_experiment(session: Session, exp_data: str) -> list:
    # Imports
    from snowflake.ml.modeling.pipeline import Pipeline
    from snowflake.ml.modeling.preprocessing import MinMaxScaler, LabelEncoder, OneHotEncoder
    from snowflake.ml.modeling.metrics import r2_score, accuracy_score, precision_score, roc_auc_score, \
        f1_score, recall_score, log_loss, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
    from snowflake.snowpark.functions import col, is_null, regexp_replace, when, lit
    from snowflake.snowpark.types import StringType
    from snowflake.snowpark.exceptions import SnowparkSQLException
    import importlib, sys, json, os, logging
    from snowflake.ml.registry.registry import Registry
    
    
    # Functions used in stored proc
    def apply_data_cleansing(df):
        """
        Method handles null values in snowpark dataframe.
        :param:
        df: input dataframe
        :returns:
        df_cleaned: dataframe after null handling
        """
        # fillna - Unknown and 0
        schema_fields = df.schema.fields
        fill_values = {field.name: "UNKNOWN" if isinstance(field.datatype, StringType) else 0 for field in schema_fields}
        df_cleaned = df.fillna(fill_values)
        return df_cleaned


    def get_feature_columns(df):
        """
        Identifies the numerical and categorical features in dataset.
        Identifies features for label encoding and one hot encoding
        :param:
        df: input dataframe
        :returns:
        categorical_features: list of non-numerical feature columns
        numerical_features: list of numerical feature columns
        le_column_features: list of feature label encoder columns
        oh_column_features: list of feature one hot encoder columns
        """
        schema_fields = df.schema.fields
        features = df.columns
        features.remove(exp_details.get("target_column"))
        df_schema = session.sql(f"DESCRIBE TABLE {exp_details.get('dataset')}").collect()
        categorical_types = ['VARCHAR','CHAR','STRING','TEXT','BOOL']
        categorical_features = []  
        for row in df_schema:
            for typ in categorical_types:
                if typ in row['type']:
                    categorical_features.append(row['name'])
                    break
        numerical_features = list(set(features) - set(categorical_features))
        print(f"numerical_features:  {numerical_features}")
        print(f"categorical_features: {categorical_features}")
        
        #identify columns for labelencoding and onehotencoding   
        le_column_features = categorical_features
        oh_column_features = []
        if len(categorical_features) >= 1:
            print(f"{categorical_features} columns are non numeric in feature dataset, encoding required.")
            for column in categorical_features:
                if df.select(df[column]).distinct().count() < 10:
                    oh_column_features.append(column)
            print(f"Columns identified to be encoded with label encoder: {le_column_features}")
            print(f"Columns identified to be encoded with one hot encoder: {oh_column_features}")
        return categorical_features, numerical_features, le_column_features, oh_column_features


    def create_and_run_preprocessing(df, categorical_features, numerical_features, le_column_features, oh_column_features):
        """
        Based on different features column input creates preprocessing steps and run it on input dataframe
        :param:
        df: input dataframe
        categorical_features: list of non-numerical feature columns
        numerical_features: list of numerical feature columns
        le_column_features: list of feature label encoder columns
        oh_column_features: list of feature one hot encoder columns
        :returns:
        df_train: preprocessed train dataset
        df_test: preprocessed test dataset
        """
        #pipeline steps 
        print("Setting up preprocessing pipeline based on dataset")
        categorical_pp = {f'le_{column}':LabelEncoder(input_cols=column, output_cols=column) for column in le_column_features}
        if len(oh_column_features)>0:
            categorical_pp['oh_enc'] = OneHotEncoder(input_cols=oh_column_features, output_cols=oh_column_features, handle_unknown='ignore')
        numerical_pp = {
            'scaler': MinMaxScaler(input_cols=numerical_features, output_cols=numerical_features)
        }
        steps = [(key, categorical_pp[key]) for key in categorical_pp if categorical_pp[key]!=[]] + \
        [(key, numerical_pp[key]) for key in numerical_pp if numerical_features!=[]]
        print(f"Selected preprocesing steps: \n{steps}")    
            
        # Run preprocessing pipeline steps 
        print("Running data preprocessing pipeline")
        df = Pipeline(steps=steps).fit(df).transform(df)
        print(f"Transformed dataset: \n {df.show()}")
        df_train, df_test = df.random_split(weights=[0.8, 0.2], seed=0)
        return df_train, df_test


    def run_estimator(df_train, df_test, input_cols):
        """
        trains on df_train and creates model object for given algorithm/estimator.
        runs prediction function of model object on test dataset
        :param:
        df_train: input training dataframe
        df_test: input test dataframe
        input_cols: list of input feature names
        :returns:
        df_pred: output predict dataframe
        """
        for algorithm, hyperparam in exp_details.get("algo_details").items():
            algorithm = algorithm.rsplit('.', 1)
            module = importlib.import_module(algorithm[0])
            print(f"----Running Algorithm {algorithm[1]}----")
            attr = getattr(module, algorithm[1])
            pipe = Pipeline(steps=[("algorithm", attr(input_cols=input_cols
                                                  , label_cols=[exp_details.get("target_column")]
                                                  , output_cols=['PREDICTIONS']))]
                   )
    
            # Fit the pipeline
            print(f"Running model pipeline {algorithm[1]}")
            model = pipe.fit(df_train)
     
            # Test the model
            print("Running prediction on model with test dataset")
            df_pred = model.predict(df_test)
            return model, df_pred

    
    def try_or(fn):
        try:
            out = fn()
            return out
        except:
            return None
    
        
    def eval_metrics(df_pred):
        print("Generating Metrices")
        if exp_details.get("algorithm_type") == 'classification':
            return {
            'accuracy_score': try_or(lambda: accuracy_score(df=df_pred, y_true_col_names=exp_details.get("target_column"), y_pred_col_names='PREDICTIONS')),
            'f1_score': try_or(lambda: f1_score(df=df_pred, y_true_col_names=exp_details.get("target_column"), y_pred_col_names='PREDICTIONS')),
            'recall_score': try_or(lambda: recall_score(df=df_pred, y_true_col_names=exp_details.get("target_column"), y_pred_col_names='PREDICTIONS')),
            'precision_score': try_or(lambda: precision_score(df=df_pred, y_true_col_names=exp_details.get("target_column"), y_pred_col_names='PREDICTIONS')),
            'roc_auc': try_or(lambda: roc_auc_score(df=df_pred, y_true_col_names=exp_details.get("target_column"), y_score_col_names='PREDICTIONS')),
            'log_loss': try_or(lambda: log_loss(df=df_pred, y_true_col_names=exp_details.get("target_column"), y_pred_col_names='PREDICTIONS'))
            }
        else:
            return {
            'mean_absolute_percentage_error': try_or(lambda: mean_absolute_percentage_error(df=df_pred, y_true_col_names=exp_details.get("target_column"), y_pred_col_names='PREDICTIONS')),
            'r2_score': try_or(lambda: r2_score(df=df_pred, y_true_col_name=exp_details.get("target_column"), y_pred_col_name='PREDICTIONS')),
            'mean_absolute_error': try_or(lambda: mean_absolute_error(df=df_pred, y_true_col_names=exp_details.get("target_column"), y_pred_col_names='PREDICTIONS')),
            'mean_squared_error': try_or(lambda: mean_squared_error(df=df_pred, y_true_col_names=exp_details.get("target_column"), y_pred_col_names='PREDICTIONS'))
            }


    def register_model(model, metrics_info):
        print("Started: Registering model on snowflake")
        reg = Registry(session=session)
        
        clean = lambda x : x.replace("-","_")
        project_id = clean(exp_details.get("project_id"))
        exp_id = clean(exp_details.get("id"))
        run_id = clean(exp_details.get("run_id"))
        model_name = exp_details.get("exp_name")+"_"+project_id+"_"+exp_id+"_"+run_id
        mv = reg.log_model(model=model,
                           model_name=model_name,
                           comment=exp_details.get("description"),
                           version_name="V1",
                           python_version="3.9.19",
                           conda_dependencies=["xgboost","scikit-learn"],
                           metrics=[{"model_metrics": metrics_info, 
                                     "project_id": exp_details.get("project_id"),
                                     "id": exp_details.get("id"),
                                     "run_id": exp_details.get("run_id"),
                                     "algorithm_type": exp_details.get("algorithm_type"),
                                     "algo_details": exp_details.get("algo_details"),
                                     "dataset": exp_details.get("dataset"),
                                     "target_column": exp_details.get("target_column"),
                                     "exp_name": exp_details.get("exp_name"),
                                     "source": "EXP"}])
        print("Registeration of model completed!!!")
        return model_name
            
            
    def create_tags(session, exp_details):
        for key in exp_details.keys():
            tag = session.sql(f"CREATE TAG IF NOT EXISTS {key}")
            tag.show()
        if exp_details.get('algorithm_type')=='classification':
            metric_names=["accuracy_score","precision_score","recall_score","f1_score","roc_auc","log_loss"]
        else:
            metric_names=["r2_score","mean_absolute_error","mean_squared_error","mean_absolute_percentage_error"]
        for name in metric_names:
            tag = session.sql(f"CREATE TAG IF NOT EXISTS {name}")
            tag.show()

        
    def set_tags(session, m_name, exp_details, metric_info):
        for key, value in exp_details.items():
            value = str(value).replace("'","\"")
            tag = session.sql(f"ALTER MODEL IF EXISTS {m_name} SET TAG {key}='{value}'")
            tag.show()
        for key, value in metric_info.items():
            value = str(value).replace("'","\"")
            tag = session.sql(f"ALTER MODEL IF EXISTS {m_name} SET TAG {key}='{value}'")
            tag.show()
    
#     try:
    # loading experiment details
    exp_details=json.loads(exp_data)
    
    # creating user tags if not exist
    create_tags(session, exp_details)
    
    # Reading dataset
    print("Reading dataset features")
    data = session.table(exp_details.get("dataset"))
    
    # fillna
    data = apply_data_cleansing(data)
    
    # Identify feature columns
    categorical_features, numerical_features, le_column_features, oh_column_features = get_feature_columns(data)
    
    # Based on feature, do preprocessing
    data_train, data_test = create_and_run_preprocessing(data, categorical_features, numerical_features, le_column_features, oh_column_features)
        
    # Run model training and prediction
    input_cols = data_train.columns
    input_cols.remove(exp_details.get("target_column"))    
    model, data_pred = run_estimator(data_train, data_test, input_cols)
    
    # Evaluate model metrices
    metrics_info = eval_metrics(data_pred)
    print(metrics_info)
    
    # Register model on snowflake registry
    model_name = register_model(model, metrics_info)
    print(model_name)
    
    # Set relevant tags to model object
    set_tags(session, model_name, exp_details, metrics_info)
    
    return [{"exp_name":exp_details.get("exp_name", "sample_experiment"),
             "version":"V1",
             "matrices":{"model_metrics": metrics_info, "project_id": exp_details.get("project_id"), "source": "EXP"},
             "algorithm_type":exp_details.get("algorithm_type"),
             "algorithm": list(exp_details.get("algo_details").keys()),
             "dataset": exp_details.get("dataset"),
             "target_column": exp_details.get("target_column"),
             "RUN_STATUS": "SUCCESS",
                 "registry_model_name":exp_details.get("exp_name")+"_"+exp_details.get("project_id")+"_"+exp_details.get("id")+"_"+exp_details.get("run_id")}]
#     except Exception as ex:
#         key = 'Processing aborted due to error 370001' 
#         if key in str(ex):
#             print("Registeration of model completed!!!")
#             return model_name
#         else:
#             print("Exception Occured in run experiment")
#             return str(ex).split('.')
        
        
def create_sproc(session, stage, func_name="run_experiment"):
    print("Creating stored procedure...")
    session.custom_package_usage_config['enabled'] = True
    session.sproc.register(func=run_experiment,
                           name=func_name,
                           packages=["snowflake-snowpark-python", "snowflake-ml-python","scikit-learn"],
                           isPermanant=False,
                           stage_location=stage,
                           replace=True)
    print("Stored procedure has been created successfully!")

    
def initiate_sproc_process(payload, sproc_name="run_experiment"):
    exp_details = json.loads(payload)
    print("Creating Snowflake Session object...")
    session = get_session(exp_details.get("dataset"),exp_details.get("project_id"))
    print("Session has been created !")
    stage = create_stage(session)
    print("Creating stored procedure...")
#     session.add_import("/notebooks/notebooks/SnowparkRnD/snow_exp.py", import_path="notebooks.notebooks.SnowparkRnD.snow_exp")
    create_sproc(session, stage)
    print("Stored procedure has been created successfully!")
    print("Executing Procedure")
    sproc_response = session.call(sproc_name, payload)
    print("Stored Procedure Executed Successfully !")
    print(sproc_response)
    return sproc_response

# Initilization
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import uuid
# project_id = str(uuid.uuid4())
run_id = str(uuid.uuid4())
print(run_id)
# payload-1 (classification Airline Delay dataset)
exp_data = '''{{
"source":"EXP", 
"project_id":"0e0fb803-22db-4d88-9f2f-f6f75b6abcf0", 
"id":"7bbb5061-54d4-4862-8d47-7fbee388a4d1", 
"run_id":"{0}", 
"exp_name": "Final_recipe", 
"algorithm_type":"classification", 
"algo_details": {{"snowflake.ml.modeling.xgboost.XGBClassifier": null}}, 
"dataset": "AIRLINE_DEP_DELAY_10K", 
"target_column": "DEP_DEL15"}}'''.format(run_id)


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

