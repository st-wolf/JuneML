import csv

from prepare import *
from view import *
from learn import *

# Разделим выборку на трети
# Обучим по 2/3, проверим по 1/3

data, answers = load_data_set('train_data.csv', 'train_answers.csv')
data = normalize(data)

n = answers.size
n_learn = 2 * (n // 3)
n_check = n - n_learn

def get_param():
	param = teach(data[:n_learn, :], answers[:n_learn])
	return param

# print(get_param())

# Joseph: 10000 итераций градиентного спуска:
# 	param = [4.66641156, -3.40293505, -2.54710512, -2.48068236, -1.57692318]
# Luci: хрен знает как:
# 	param = [4.88031677, -3.65326592, -2.51913162, -2.52432301, -1.80838042]

def count_error():
	# param = [4.66641156, -3.40293505, -2.54710512, -2.48068236, -1.57692318]
	param = get_param()
	classified = classify(param, data[:n_learn, :])

	n_error = 0
	for forcast, answer in zip(classified, answers[:n_learn]):
		if forcast != answer:
			n_error += 1

	# На первых 2-х тертях (по ним обучали): 80.46
	# На последней трети (по которой не обучали): 79.83
	print(1 - n_error / n_learn)

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

process_test()
