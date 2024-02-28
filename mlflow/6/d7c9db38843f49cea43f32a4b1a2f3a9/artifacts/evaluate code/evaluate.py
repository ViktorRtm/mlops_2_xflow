import os
import json

import pandas as pd
import numpy as np

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error
from model_learning import fit
from loading_datasets import load_dataset

def calculate_score(val_dataset, submission):

    merged_Y = val_dataset.merge(
        submission, left_on=['animal_id_x', 'lactation'], right_on=['animal_id', 'lactation'], how='outer'
    )
    mean_squared_errors = []

    median_value = np.nanmedian(submission[[f'milk_yield_{i}' for i in range(3, 11)]].values)

    for index, row in merged_Y.iterrows():
        arr_real = (
            row[
                [f'milk_yield_{i}_x' for i in range(3, 11)]
            ].fillna(method='ffill')
             .fillna(method='bfill')
        )
        arr_predict = (
            row[
                [f'milk_yield_{i}' for i in range(3, 11)]
            ].fillna(method='ffill')
             .fillna(method='bfill')
             .fillna(value=median_value)
             .fillna(value=0)
        )

        mean_squared_errors.append(mean_squared_error(arr_real, arr_predict))

    rmse_score = np.sqrt(np.mean(mean_squared_errors))

    return rmse_score

def predict(models: list[HistGradientBoostingRegressor]) -> pd.DataFrame:
    """
    @param:
    @param:
    @returns:
    """

    in_val_x_path = '/home/viktor/project/data/stage4/val_x.csv'
    X_val = load_dataset(in_val_x_path)

    num_columns = X_val.select_dtypes(include='number').columns
    cat_columns = X_val.select_dtypes(include='category').columns

    cat_columns = cat_columns[1:]

    num_pipe = Pipeline([('scaler', MinMaxScaler())])
    cat_ohe_pipe = Pipeline([('oheencoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    preproc = ColumnTransformer(transformers=[
        ('num', num_pipe, num_columns),
        ('cat_ord', cat_ohe_pipe, cat_columns)
        ])

    X_test_preproc = preproc.fit_transform(X_val)
    
    list_prediction_ridge = [item.predict(X_test_preproc) for item in models]

    subm_df_ridge = X_val.loc[:,['animal_id_x', 'lactation']]
    subm_df_ridge.columns = ['animal_id', 'lactation']

    for idx in range(0, 8):
        subm_df_ridge[f'milk_yield_{idx+3}'] = list_prediction_ridge[idx]

    return subm_df_ridge

if __name__ == '__main__':
    
    _models = fit()
    _submission = predict(_models)

    in_val_x_path = '/home/viktor/project/data/stage4/val_x.csv'
    in_val_y_path = '/home/viktor/project/data/stage4/val_y.csv'
    X_val = load_dataset(in_val_x_path)
    y_val = load_dataset(in_val_y_path)

    val_dataset = X_val.merge(y_val, right_index=True, left_index=True)

    rmse = calculate_score(val_dataset, _submission)

    os.makedirs(os.path.join('evaluate'), exist_ok=True)
    
    with open('evaluate/score.json', 'w') as f:
        json.dump({'score': rmse}, f)
