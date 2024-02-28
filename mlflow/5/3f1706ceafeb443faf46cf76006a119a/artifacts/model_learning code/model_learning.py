import pickle
import yaml
import os

from typing import Any
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from data_preparation import load_dataset


def fit() -> list[HistGradientBoostingRegressor]:
    """
    Простая линейная модель
    @return: лист из обученных моделей
    """
    
    params = yaml.safe_load(open('/home/viktor/project/params.yaml'))['train']

    in_train_x_path = '/home/viktor/project/data/stage4/train_x.csv'
    in_train_y_path = '/home/viktor/project/data/stage4/train_y.csv'
    os.makedirs('/home/viktor/project/models', exist_ok=True)
    
    X_train = load_dataset(in_train_x_path)
    y_train = load_dataset(in_train_y_path)
        
    if os.path.exists('models/model.pickle'):
        with open('models/model.pickle', 'rb') as f:
            list_of_model_Ridge = pickle.load(f)
            f.close()
            return list_of_model_Ridge

    num_pipe = Pipeline([('scaler', MinMaxScaler())])
    cat_ohe_pipe = Pipeline([('oheencoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

    num_columns = X_train.select_dtypes(include='number').columns
    cat_columns = X_train.select_dtypes(include='category').columns

    cat_columns = cat_columns[1:]

    preproc = ColumnTransformer(transformers=[
        ('num', num_pipe, num_columns),
        ('cat_ord', cat_ohe_pipe, cat_columns)
        ])

    list_of_model_HistGradientBoostingRegressor = []

    for item in range(3, 11):
        print(f'Обучается {item-2} модель')
        pipe = None
        pipe = Pipeline([
            ('preproc', preproc),
            ('HistGradientBoostingRegressor', HistGradientBoostingRegressor(loss='poisson',
                                                                            learning_rate=params['learning_rate'],
                                                                            max_iter=180,
                                                                            max_leaf_nodes=100,
                                                                            max_depth=params['max_depth'],
                                                                            categorical_features=None,
                                                                            l2_regularization=0,
                                                                            scoring='neg_root_mean_squared_error',
                                                                            verbose=2,
                                                                            random_state=42
                                                                            ))
                        ])
        pipe.fit(X_train, y_train[f'milk_yield_{item}_x'])
        list_of_model_HistGradientBoostingRegressor.append(pipe['HistGradientBoostingRegressor'])

    with open('/home/viktor/project/models/model.pickle', 'wb') as f:
        pickle.dump(list_of_model_HistGradientBoostingRegressor, f)
    f.close()

    return list_of_model_HistGradientBoostingRegressor

if __name__ == '__main__':
    
    _models = fit()
