from airflow import DAG
import airflow
from datetime import datetime, timedelta
from airflow.operators.bash import BashOperator

args = {
    'owner': 'ViktorRtm',
    'start_date':datetime(2024, 2, 26),
    'retries': 1,
    'retry_delay': timedelta(minutes=10),
    'depends_on_past': False,
    'provide_context':True
}

with DAG(
    dag_id='cow_milk_predict', 
    schedule=None,  
    catchup=False, 
    default_args=args
    ) as dag:
    clear_train = BashOperator(
        task_id='clear_train',
        bash_command='python3 /home/viktor/project/scripts/clear_train.py',
        dag=dag
    )
    data_preparation = BashOperator(
        task_id='data_preparation',
        bash_command='python3 /home/viktor/project/scripts/data_preparation.py',
        dag=dag
    )
    split_on_x_y = BashOperator(
        task_id='split_on_x_y',
        bash_command='python3 /home/viktor/project/scripts/split_on_x_y.py',
        dag=dag
    )
    data_split = BashOperator(
        task_id='data_split',
        bash_command='python3 /home/viktor/project/scripts/data_split.py',
        dag=dag
    )
    model_learning = BashOperator(
        task_id='model_learning',
        bash_command='python3 /home/viktor/project/scripts/model_learning.py',
        dag=dag
    )
    evaluate = BashOperator(
        task_id='evaluate',
        bash_command='python3 /home/viktor/project/scripts/evaluate.py',
        dag=dag
    )

    clear_train >> data_preparation >> split_on_x_y >> data_split >> model_learning >> evaluate
