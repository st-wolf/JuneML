from prepare import *
from view import *
from learn import *
import matplotlib.pyplot as plt


data = load_data_set('train.csv')
answers = load_training_answers('train.csv')
f_matrix = parse_data(data)
f_n_matrix = normalize(f_matrix)

# Делаем вывод нормальным
np.set_printoptions(threshold = np.nan) # вывод всей матрицы
np.set_printoptions(suppress = True) # человеко-понятные цифры

x0 = f_n_matrix[1, :]

optim_feature_m = teach(f_n_matrix, answers, x0)

print(optim_feature_m)

# Лог работы
# точность: 9.97064402533e-06, 2259 шагов
# тета: [ 4.98156804 -3.80341945 -2.51632704 -2.93216553  0.2794166 ]
# время работы:	1m10.601s