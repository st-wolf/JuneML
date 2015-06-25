import numpy as np
import matplotlib.pyplot as plt

from learn import cost, grad
from optimize import gradien_descent

def learning_curve(features, answers):
	"""
	Рисует кривую обучения (зависомость cost от итераций)
	"""

	rate = 0
	param_init = np.zeros(features.shape[1])

	_, cost_values = gradien_descent(
		lambda param: cost(param, features, answers, rate),
		lambda param: grad(param, features, answers, rate),
		param_init,
		maxiter = 500 )

	plt.plot(cost_values)
	plt.show()


def show_classes(x1, x2, y, names = []):
	"""
	Рисует классы на плоскости двух параметров
	"""

	# Цветовая схема от Andrew Ng
	plt.scatter(x1[y == 0], x2[y == 0], s = 40, c = 'yellow')
	plt.scatter(x1[y == 1], x2[y == 1], s = 100, c = 'black', marker = '+', linewidth = 2)
	plt.legend(['died', 'survived'])

	if len(names) == 2:
		plt.xlabel(names[0])
		plt.ylabel(names[1])

	plt.grid()
	plt.show()

# Прототип
# TODO: добавить полиномиальные фичи
# Нужно иметь возможность добавить их и модель обучения тоже!

# Стоит сделать функцию более общей, передавая в нее классификатор, значения классов и т.д.
def decision_boundary(param, f_numbers, x_range, y_range, names = []):
	"""
	Рисует проекцию разделяющей поверхности на плоскость двух параметров

	Args:
		param (ndarray):
		f_numbers (2d tupples): номера признаков, определяющих плоскость проекции
		x_range (2d tuple): диапазон знанчени по оси x
		y_range (2d tuple): диапазон значений по оси y

	Return:
		None
	"""

	n_x = n_y = 10;

	x = np.linspace(x_range[0], x_range[1], n_x)
	y = np.linspace(y_range[0], y_range[1], n_y)
	X, Y = np.meshgrid(x, y)

	# Векторизуем, сделав один вызов функции classify
	# Разворачиваем матрицы в строку
	x_unroll = X.reshape(n_x * n_y)
	y_unroll = Y.reshape(n_x * n_y)

	ix, iy = f_numbers
	f_matrix = np.zeros((n_x * n_y, param.size))

	f_matrix[:, 0] = np.ones(n_x * n_y)
	f_matrix[:, ix] = x_unroll
	f_matrix[:, iy] = y_unroll

	hypothesis = np.dot(f_matrix, param)

	Z = hypothesis.reshape((n_x, n_y))

	plt.contour(X, Y, Z, levels = [0])

	if len(names) == 2:
		plt.xlabel(names[0])
		plt.ylabel(names[1])

	plt.grid()
	plt.show()

"""
param = np.array([1, 1, 1])
f_numbers = (1, 2)
x_range = y_range = (0, 1)
names = ["feature_1", "feature_2"]
decision_boundary(param, f_numbers, x_range, y_range, names)
"""