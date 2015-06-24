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
	param = [4.66641156, -3.40293505, -2.54710512, -2.48068236, -1.57692318]
	classified = classify(param, data[:n_learn, :])
	
	n_error = 0
	for forcast, answer in zip(classified, answers[:n_learn]):
		if forcast != answer:
			n_error += 1

	# На первых 2-х тертях (по ним обучали): 80.46
	# На последней трети (по которой не обучали): 79.83
	print(n_error / n_learn * 100)

count_error()

# classified_data = classify(theta, test_data)

# n = 0
# m = len(classified_data)
# for iter in range(0, m - 1):
# 	if bool(test_answers[iter]) == classified_data[iter]:
#		n += 1
# print('Процент попаданий - ', n / m * 100)
