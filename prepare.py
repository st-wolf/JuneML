import csv
import numpy as np
import os


"""
Признаковое описание:

Пока предлагаю остановится на следущем подмножестве признаков:
pclass, sex, age, fare
"""
FEATURES = [
    'passenger_id', # идентификатор юзера
    'survived',     # спасли ли бедолагу?
    'pclass',       # класс билета пассажира (1 = 1st, 2 = 2nd, 3 = 3rd)
    'name',         # имя
    'sex',          # половая принадлежность (0 = male, 1 = female)
    'age',          # возраст
    'sibsp',        # количество братьев, сестер, супруг и т.д. на борту
    'parch',        # количество родителей, детей на борту 
    'ticket',       # номер билета
    'fare',         # стоимость билета  
    'cabin',        # место на корабле
    'embarked',     # порт посадки (C = Cherbourg, Q = Queenstown, S = Southampton)
]

def convert_row_to_dict(row):
    """
    Преобразует строку в словарь
    @param row параметры обучаемой модели
    @return dict 
    """
    feature_row = {}

    i = 0

    for feature in FEATURES:
        feature_row[feature] = row[i]
        i += 1

    return feature_row

def load_training_answers(file_name, path_dir = 'data', relative = True):
    """
    @todo Копипаста - плохо
    Читает ответы учителя из csv
    Возвращает вектор ответов
    @param :file_name: имя файла
    @param :path_dir: путь с файлом (по умолчанию берется из папки data)
    @param :relative: абсолютный ли относительный путь передан в path_dir?
    @return numpy.array
    """

    # Определяем абсолютный путь к файлу
    if relative:
        path = os.getcwd()
    else:
        path = ''

    file_path = os.path.join(path, path_dir, file_name)

    with open(file_path) as input_file:
        # csv.reader возвращает объект типа Reader
        # закрытие файла input_file приводит к уничтожению ридера
        raw_data = csv.reader(input_file, delimiter = ',')

        next(raw_data) # Опускаем заголовок

        data = []

        IS_SURVIVED = 1 # костыль

        for row in raw_data:
            data.append(row[IS_SURVIVED])

    return np.array(data)




def load_data_set(file_name, path_dir = 'data', relative = True):
    """
    Читает данные из csv
    Возвращает ndarray с разобранными данными
    @param :file_name: имя файла
    @param :path_dir: путь с файлом (по умолчанию берется из папки data)
    @param :relative: абсолютный ли относительный путь передан в path_dir?
    @return list
    """

    # Определяем абсолютный путь к файлу
    if relative:
        path = os.getcwd()
    else:
        path = ''

    file_path = os.path.join(path, path_dir, file_name)

    with open(file_path) as input_file:
        # csv.reader возвращает объект типа Reader
        # закрытие файла input_file приводит к уничтожению ридера
        raw_data = csv.reader(input_file, delimiter = ',')

        next(raw_data) # Опускаем заголовок

        data = []

        for row in raw_data:
            data.append(convert_row_to_dict(row))

    return data

def parse_data(data):
    """ Превращает человеко-понятные данные во feature matrix
    Пока работаем только с подмножеством (pclass, sex, age, fare)
    @param np_array data разобранные в load_training_set данные
    @return numpy.array 
    """
    parsed_data = []

    for row in data:
        # костыль для отсутствующего в датасете возраста
        if row['age'] == "":
            continue

        sex = 1 if row['sex'] == 'male' else 0

        parsed_data.append(
            [int(row['pclass']), 
            sex,                    # Прости, Господи
            float(row['age']), 
            float(row['fare'])
            ]
        )

    return np.array(parsed_data)


        
def normalize(x):
    # Нормализаця. Значения всех признаков переводятся в диапазон [-1,1]
    # добавляется столбец с единицами
    x_norm = x / np.amax(abs(x), axis = 0)
    features_len = x_norm.shape[0]
    x_norm = np.c_[ np.ones(features_len), x_norm  ]
    
    return x_norm


"""
Feature Matrix будем хранить в формате ndarray из библиотеки numpy.
Все массивы ndarray имеют тип. В нашем случае он числовой.
Значит нужно аккуратно распарсить csv и положить данные в ndarray.
Все числовые признаки пока примем как есть, остальным присвоим числовые значения.
"""


# data = load_data_set('train.csv')
# answers = load_training_answers('train.csv')
# f_matrix = parse_data(data)
# f_n_matrix = normalize(f_matrix)


# # Делаем вывод нормальным
# np.set_printoptions(threshold=np.nan) # вывод всей матрицы
# np.set_printoptions(suppress=True) # человеко-понятные цифры
# print(f_n_matrix)