import csv
import numpy as np

# Достаем данные из csv файла

"""
Feature Matrix будем хранить в формате ndarray из библиотеки numpy.
Все массивы ndarray имеют тип. В нашем случае он числовой.
Значит нужно аккуратно распарсить csv и положить данные в ndarray.
Все числовые признаки пока примем как есть, остальным присвоим числовые значения.
"""

# Отбор признаков

"""
Признаковое описание:

pclass   : класс билета пассажира (1 = 1st, 2 = 2nd, 3 = 3rd)
name     : имя
sex      : половая принадлежность (0 = male, 1 = female)
age      : возраст
sibsp    : количество братьев, сестер, супруг и т.д. на борту
parch    : количество родителей, детей на борту 
ticket   : номер билета
fare     : стоимость билета
cabin    : место на корабле
embarked : порт посадки (C = Cherbourg, Q = Queenstown, S = Southampton)

Пока предлагаю остановится на следущем подмножестве признаков:
pclass, sex, age, fare
"""

def parse_row(row):
	# Преобразует строку в числа
	sex = 1 if row[4] == 'male' else 0
	parsed = [int(row[2]), sex, float(row[5]), float(row[9])]
	return parsed

def load(file_name):
	# Читает данные из csv, возвращает X и y в ndarray

	with open('./data/train.csv') as input_file:
		# csv.reader возвращает объект типа Reader
		# закрытие файла input_file приводит к уничтожению ридера
		raw_data = csv.reader(input_file, delimiter = ',')
		next(raw_data) # Опускаем заголовок

		data = []
		survived = []
		for row in raw_data:
			if all([row[i] != '' for i in [2, 4, 5, 9]]):
				data.append(parse_row(row))
				survived.append(int(row[1]))

	X = np.array(data)
	y = np.array(survived)

	return X, y

def normalize(X):
	# Нормализаця. Значения всех признаков переводятся в диапазон [-1,1]
	return X / abs(X).max(axis = 0)

# X, y = load('./data/train.csv')
