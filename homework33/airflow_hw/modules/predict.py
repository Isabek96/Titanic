import pandas as pd
import os
import dill
import json
from datetime import datetime

path = os.environ.get('PROJECT_PATH', '.')


def get_latest_model(model):
    files = os.listdir(model)
    if not files:
        print("В каталоге не найдены файлы моделей.")
    files.sort()
    latest_file = os.path.join(model, files[-1])
    return latest_file


def predict():
    with open(f'{path}./data/models/cars_pipe_202407311853.pkl', 'rb') as file:
        model = dill.load(file)

        df_pred = pd.DataFrame(columns=['id', 'predict'])
        files_test = os.listdir(f'{path}./data/test')

        for filename in files_test:
            with open(f'{path}./data/test/{filename}', 'r') as file:
                form = json.load(file)
            data = pd.DataFrame.from_dict([form])
            y = model.predict(data)

            dict_pred = {'id': data['id'].values[0], 'predict': y[0]}
            df = pd.DataFrame([dict_pred])
            df_pred = pd.concat([df, df_pred], ignore_index=True)

        # Получить текущее время и дату
        now = datetime.now().strftime('%Y%m%d')
        # Сохранить результаты в CSV файл с временной меткой
        df_pred.to_csv(f'{path}./data/predictions/predict_{now}.csv', index=False)


if __name__ == '__main__':
    predict()
