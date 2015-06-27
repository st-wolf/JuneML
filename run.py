import csv
import sys

from prepare import *
from view import *
from learn import *
from view import *

""" Процент данных, по которым обучаем"""
PERCENT = 70

""" Точность поиска """
ACCURACY = 1e-10

""" Ограничение по итерациям (пока не нужно) """
MAX_ITER = -1


# Joseph: 5000 итераций градиентного спуска:
# 	param = [4.98803499, -3.80921867, -2.51819642, -2.93658204, 0.27530952]
# Luci: хрен знает как:
# 	param = [4.88031677, -3.65326592, -2.51913162, -2.52432301, -1.80838042]

# На первых 2-х третях (по ним обучали): 80.46
# На последней трети (по которой не обучали): 79.83

data, answers = load_data_set('train_data.csv', 'train_answers.csv')
data = normalize(data)

n = answers.size
n_learn = round(PERCENT / 100 * n)
n_check = n - n_learn

param = teach(data[: n_learn, :], answers[: n_learn], ACCURACY, MAX_ITER)
print_count_error(param, data, answers, PERCENT)


# count_error()

# Костылищщи! 
# Не думай об этом, лучше я объясню тебе, в чем проблема 
def parse_row(row):
	# Преобразует строку в числа
	sex = 1 if row[3] == 'male' else 0

	f = lambda i: float(row[i]) if row[i] != '' else 0

	parsed = [f(1), sex, f(4), f(8)]
	return parsed


def process_test():
	# Ща будет еще костыль!
	# ID-шники никуда не выгружаются
	ids = []
	with open('data/test_data.csv', 'r') as data_file:
		raw_data = csv.reader(data_file, delimiter = ',')
		next(raw_data)

		data = []
		for row in raw_data:
			data.append(parse_row(row))
			ids.append(row[0])

		data = np.array(data)


	data = normalize(data)

	param = [4.66641156, -3.40293505, -2.54710512, -2.48068236, -1.57692318]
	classified = classify(param, data)

	with open('data/test_answers.txt', 'w') as answers_file:
		answers_writer = csv.writer(answers_file, delimiter = ',')
		answers_writer.writerow(["PassengerId", "Survived"])

		for pass_id, forcast in zip(ids, classified):
			answers_writer.writerow([pass_id, int(forcast)])

def vizualize():
	# 1 - pclass, 2 - sex, 3 - age, 4 - fare
	param = np.array([4.66641156, -3.40293505, -2.54710512, -2.48068236, -1.57692318])
	f_numbers = (1, 4)
	names = ["age", "fare"]
	x_range = (0,1)
	y_range = (-0.2, 1.2)

	show_classes(data[:, 3], data[:, 4], answers, names)
	# decision_boundary(param, f_numbers, x_range, y_range, names)


