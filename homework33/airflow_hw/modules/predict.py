import pandas as pd
import os
import dill
import json
from datetime import datetime


def predict():
    with open('../data/models/cars_pipe_202407311853.pkl', 'rb') as file:
        model = dill.load(file)

        df_pred = pd.DataFrame(columns=['id', 'predict'])
        files_test = os.listdir('../data/test')

        for filename in files_test:
            with open(f'../data/test/{filename}', 'r') as file:
                form = json.load(file)
            data = pd.DataFrame.from_dict([form])
            y = model.predict(data)

            dict_pred = {'id': data['id'].values[0], 'predict': y[0]}
            df = pd.DataFrame([dict_pred])
            df_pred = pd.concat([df, df_pred], ignore_index=True)

        # Получить текущее время и дату
        now = datetime.now().strftime('%Y%m%d%H%M')
        # Сохранить результаты в CSV файл с временной меткой
        df_pred.to_csv(f'../data/predictions/{now}.csv', index=False)


if __name__ == '__main__':
    predict()
