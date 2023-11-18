import glob
import json
import dill
from datetime import datetime
import os
import pandas as pd

#path = 'C:/Users/user/airflow_hw'
path = os.environ.get('PROJECT_PATH', '.')


def predict():
    for model_path in glob.glob(f'{path}/data/models/*.pkl'):
        with open(model_path, 'rb') as file:
            model = dill.load(file)
    df_pred = pd.DataFrame(columns=['car_id', 'pred'])
    for filename in glob.glob(f'{path}/data/test/*.json'):
        with open(filename) as fin:
            form = json.load(fin)
            df = pd.DataFrame.from_dict([form])
            y = model.predict(df)
            X = {'car_id': df.id, 'pred': y}
            df1 = pd.DataFrame(X)
            df_pred = pd.concat([df_pred, df1], axis=0)
    df_pred.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)


if __name__ == '__main__':
    predict()

