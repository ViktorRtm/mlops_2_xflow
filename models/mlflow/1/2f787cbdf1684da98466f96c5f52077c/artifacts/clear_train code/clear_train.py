import os
import mlflow

from mlflow.tracking import MlflowClient
from loading_datasets import load_dataset, upload_dataset

os.environ['MLFLOW_REGISTRY_URI'] = '/home/viktor/mlflow/'
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('clear_train')


in_train_path = '/home/viktor/project/data/train.csv'
in_pedigree_path = '/home/viktor/project/data/pedigree.csv'
out_train_path = '/home/viktor/project/data/stage1/train.csv'
os.makedirs('/home/viktor/project/data/stage1', exist_ok=True)

with mlflow.start_run():
	train_df = load_dataset(in_train_path)
	cow_pedigree = load_dataset(in_pedigree_path)

	for item in train_df.columns:
		train_df.loc[train_df[item].isna(), item] = 0

	train_df = train_df.merge(cow_pedigree[['animal_id', 'mother_id']], 
	                                        on='animal_id', 
	                                        how='left')

	upload_dataset(train_df, out_train_path)
	mlflow.log_artifact(local_path='//home/viktor/mlflow/scripts/clear_train.py', artifact_path='clear_train code')
	mlflow.end_run()
