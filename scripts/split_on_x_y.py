import os
import mlflow

from mlflow.tracking import MlflowClient
from loading_datasets import load_dataset, upload_dataset

X_COLUMNS = ['animal_id_x','lactation','farm','milk_yield_1_x','milk_yield_2_x','day_of_life_befor_calving','milk_yield_1_y','milk_yield_2_y','milk_yield_3_y','milk_yield_4_y','milk_yield_5_y','milk_yield_6_y','milk_yield_7_y','milk_yield_8_y','milk_yield_9_y','milk_yield_10_y','milk_yeld_mean']
Y_COLUMNS = ['milk_yield_3_x','milk_yield_4_x','milk_yield_5_x','milk_yield_6_x','milk_yield_7_x','milk_yield_8_x','milk_yield_9_x','milk_yield_10_x']

in_train_path = '/home/viktor/project/data/stage2/train.csv'
out_train_x_path = '/home/viktor/project/data/stage3/train_x.csv'
out_train_y_path = '/home/viktor/project/data/stage3/train_y.csv'
os.makedirs('/home/viktor/project/data/stage3', exist_ok=True)

mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('split_on_x_y')

with mlflow.start_run():
	train_df = load_dataset(in_train_path)

	train_df_x, train_df_y = train_df.loc[:,X_COLUMNS], train_df.loc[:,Y_COLUMNS]

	upload_dataset(train_df_x, out_train_x_path)
	upload_dataset(train_df_y, out_train_y_path)
	mlflow.log_artifact(local_path='/home/viktor/mlflow/scripts/split_on_x_y.py', artifact_path='split_on_x_y code')
	mlflow.end_run()
