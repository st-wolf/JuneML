from prepare import *
from view import *
from learn import *
import matplotlib.pyplot as plt


# train_data = load_data_set('train.csv')
# train_answers = load_training_answers('train.csv')
# f_matrix = parse_data(data)
# f_n_matrix = normalize(f_matrix)

# # Делаем вывод нормальным
# np.set_printoptions(threshold = np.nan) # вывод всей матрицы
# np.set_printoptions(suppress = True) # человеко-понятные цифры

# x0 = f_n_matrix[1, :]
# optim_feature_m = teach(f_n_matrix, answers, x0)

# print(optim_feature_m)

# Лог работы по изначальным данным (старая версия)
# точность: 9.97064402533e-06, 2259 шагов
# тета: [ 4.98156804 -3.80341945 -2.51632704 -2.93216553  0.2794166 ]
# время работы:	1m10.601s


test_data, test_answers = load_data_set('test.csv', 'gendermodel.csv')
test_data = normalize(test_data)

# x0 = test_data[1, :]
# optim_feature_m = teach(test_data, test_answers, x0)
# print(optim_feature_m)


theta = np.array([ 4.98156804, -3.80341945, -2.51632704, -2.93216553,  0.2794166 ])
classified_data = classify(theta, test_data)

n = 0
m = len(classified_data)
for iter in range(0, m - 1):
	if bool(test_answers[iter]) == classified_data[iter]:
		n += 1
print('Процент попаданий - ', n / m * 100)