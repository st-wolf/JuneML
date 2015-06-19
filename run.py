from prepare import *
from learn import *
import matplotlib.pyplot as plt


data = load_data_set('train.csv')
answers = load_training_answers('train.csv')
f_matrix = parse_data(data)
f_n_matrix = normalize(f_matrix)

# Делаем вывод нормальным
np.set_printoptions(threshold = np.nan) # вывод всей матрицы
np.set_printoptions(suppress = True) # человеко-понятные цифры

x0 = f_n_matrix.shape[1]
print(x0)
optim_feature_m = teach(f_n_matrix, answers, x0)


# print(f_matrix[:, 1])



# a = np.arange(-9, 9)
# b = sigmoid(a)


# print(f_n_matrix)



# print(optim_feature_m)

#----------------------
# theta = np.arange(3, 5)
# x = np.arange(-3, -1)
# y = np.ones(2)

# print(theta, x, y)

# x0 = np.zeros(x.shape[0] + 1)

# print(x0)

# print(teach(x, y, x0))

# plt.scatter(x, y, s = 40, c = 'yellow')

# plt.grid()
# plt.show()
#______________________