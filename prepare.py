import csv
import numpy as np
import os

from collections import namedtuple

'''
Именованный кортеж с признаками
'''
Row = namedtuple(
    'Row', 
    [
        'passenger_id', # идентификатор юзера
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
)

'''
Используемые признаки
'''
FEATURES = [
    'pclass',
    'sex',
    'age', 
    'fare'
]

def load_data_set(data_file_name, answers_file_name, path_dir = 'data', relative = True):
    """
    Читает данные из csv
    Возвращает ndarray с разобранными данными
    @param :data_file_name: имя файла с данными
    @param :answers_file_name: имя файла с ответами
    @param :path_dir: путь с файлом (по умолчанию берется из папки data)
    @param :relative: абсолютный ли относительный путь передан в path_dir?
    @return (np.array, np.array) Матрицу параметров, Вектор ответов
    """

    # Определяем абсолютный путь к файлу
    if relative:
        path = os.getcwd()
    else:
        path = ''

    # Путь к файлу с данными
    data_file_path = os.path.join(path, path_dir, data_file_name)

    # Путь к файлу с ответами
    answers_file_path = os.path.join(path, path_dir, answers_file_name)

    try:
        with open(data_file_path, 'r') as data_file, open(answers_file_path, 'r') as answers_file:
            raw_data = csv.reader(data_file, delimiter = ',')
            raw_answers = csv.reader(answers_file, delimiter = ',')

            # Опускаем заголовки
            next(raw_data)
            next(raw_answers)

            data = []
            answers = []
            for row_dat, row_ans in zip(raw_data, raw_answers):
                dat = Row._make(row_dat) # Создаем объект namedtuple для удобной работы с полями
                if (check_data_consist(dat)):
                    vector_data = convert_data_to_vector(dat) # Превращаем данные в необходимого вида вектор
                    data.append(vector_data)
                    answers.append(int(row_ans[1])) # Параметр выживания

            data = np.array(data)
            answers = np.array(answers)

        return (data, answers)

    except IOError as e:
        print ('Operation failed: %s' % e.strerror)



def check_data_consist(data):
    '''
    Проверяем целостность входных данных
    @data namedtuple - данные для проверки
    @return Bool
    '''

    # Конвертируем в словарь для доступа к индексам
    data = data._asdict()
    for feature in FEATURES:
        if (feature in data) and (data[feature] == ''):
            return False
    return True


def convert_data_to_vector(data):
    """ Превращает человеко-понятные данные во вектор значений параметров
    Пока работаем только с подмножеством (pclass, sex, age, fare)
    @param np_array data разобранные в load_training_set данные
    @return numpy.array 
    """
    vector_data = []

    # Конвертируем в словарь для доступа к индексам
    data = data._asdict()
    for feature in FEATURES:
        value = data[feature]

        if feature == 'sex':
            value = 1 if (value == 'male') else 0

        vector_data.append(float(value))

    return vector_data


        
def normalize(data):
    '''Нормализаця. Значения всех признаков переводятся в диапазон [-1,1]
       добавляется столбец с единицами'''
    data_norm = data / np.amax(abs(data), axis = 0)
    features_len = data_norm.shape[0]
    data_norm = np.c_[ np.ones(features_len), data_norm  ]
    
    return data_norm


"""
Feature Matrix будем хранить в формате ndarray из библиотеки numpy.
Все массивы ndarray имеют тип. В нашем случае он числовой.
Значит нужно аккуратно распарсить csv и положить данные в ndarray.
Все числовые признаки пока примем как есть, остальным присвоим числовые значения.
"""