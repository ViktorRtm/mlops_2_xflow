import os
import pandas as pd
import numpy as np
import tqdm

from loading_datasets import load_dataset, upload_dataset


def data_preparation(X_df: pd.DataFrame, X_meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    @return:
    """

    X_df.calving_date = pd.to_datetime(X_df.calving_date)
    X_df.birth_date = pd.to_datetime(X_df.birth_date)
    X_df['day_of_life_befor_calving'] = (X_df.calving_date - X_df.birth_date).map(lambda x: x.days)

    X_df.lactation = X_df.lactation.astype('object')
    X_df.farmgroup = X_df.farmgroup.astype('object')
    
    X_df = X_df.merge(X_meta_df, 
                     left_on=['mother_id'],
                     right_on=['animal_id'], 
                     how='left')

    columns_for_drop = ['calving_date', 'farmgroup', 'birth_date', 'mother_id', 'animal_id_y']

    X_df = X_df.drop(columns=columns_for_drop)

    for item in X_df.columns:
        X_df.loc[X_df[item].isna(), item] = 0

    return X_df


in_train_path = '/home/viktor/project/data/stage1/train.csv'
in_test_path = '/home/viktor/project/data/X_test_private.csv'
out_train_path = '/home/viktor/project/data/stage2/train.csv'
os.makedirs('/home/viktor/project/data/stage2', exist_ok=True)

train_df = load_dataset(in_train_path)
test_df = load_dataset(in_test_path)

work_df = pd.concat([train_df, test_df])
group_milk_yeld = work_df.groupby(['animal_id'], as_index=False)[['milk_yield_1', 'milk_yield_2', 'milk_yield_3', 'milk_yield_4', 'milk_yield_5', 'milk_yield_6', 'milk_yield_7', 'milk_yield_8', 'milk_yield_9', 'milk_yield_10']].mean()

mean_list = []

for item in tqdm.tqdm(group_milk_yeld.iterrows()):
    mean_list.append(np.mean(item[1][1:]))

group_milk_yeld['milk_yeld_mean'] = mean_list

train_df = data_preparation(train_df, group_milk_yeld)

upload_dataset(train_df, out_train_path)
