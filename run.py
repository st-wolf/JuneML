from prepare import *
from view import *
from learn import *
import matplotlib.pyplot as plt

test_data, test_answers = load_data_set('train_data.csv', 'train_answers.csv')
test_data = normalize(test_data)

x0 = test_data[1, :]
optim_feature_m = teach(test_data, test_answers, x0)
theta = np.array(optim_feature_m)
print(optim_feature_m)

# theta = np.array([ 4.88031677, -3.65326592, -2.51913162, -2.52432301, -1.80838042])

classified_data = classify(theta, test_data)

n = 0
m = len(classified_data)
for iter in range(0, m - 1):
	if bool(test_answers[iter]) == classified_data[iter]:
		n += 1
print('Процент попаданий - ', n / m * 100)
