import yaml
import os

from data_preparation import load_dataset, upload_dataset
from sklearn.model_selection import train_test_split


params = yaml.safe_load(open('/home/viktor/project/params.yaml'))['split']

in_x_path = '/home/viktor/project/data/stage3/train_x.csv'
in_y_path = '/home/viktor/project/data/stage3/train_y.csv'
out_train_x_path = '/home/viktor/project/data/stage4/train_x.csv'
out_train_y_path = '/home/viktor/project/data/stage4/train_y.csv'
out_val_x_path = '/home/viktor/project/data/stage4/val_x.csv'
out_val_y_path = '/home/viktor/project/data/stage4/val_y.csv'
os.makedirs('/home/viktor/project/data/stage4', exist_ok=True)

split_ratio = params['split_ratio']

X = load_dataset(in_x_path)
y = load_dataset(in_y_path)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=split_ratio)

upload_dataset(X_train, out_train_x_path)
upload_dataset(X_val, out_val_x_path)
upload_dataset(y_train, out_train_y_path)
upload_dataset(y_val, out_val_y_path)
