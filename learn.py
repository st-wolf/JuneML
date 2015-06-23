import numpy as np

from optimize import *

"""
Матрица объекты - признаки хранится в ndarray и имеет вид:

X = 1 .. ..
    1 .. ..
    1 .. ..

"""

def sigmoid(x):
	"""
	Сигмоид-функция
	"""

	return 1 / (1 + np.exp(-x))

def hypothesis(theta, x):
	"""
	Функция гипотезы логистической регрессии
	"""
	return sigmoid(np.dot(theta, x))


def cost(theta, x, y, rate = 0):
	"""
	Вычисление функции стоимости

	Args:
		theta (ndarray): параметры обучаемой модели
		X (ndarray): матрица объекты - признаки
		y (ndarray): вектор ответов
		rate: степень регуляризации

	Return:
		float: значение функции стоимости
	"""

	m = y.shape[0]

	# Костыль
	cost = 0

	for iter in range(0, m - 1):
		if (y[iter] == 0) and (1 - hypothesis(theta, x[iter, :]) > 0):
			-np.log(1 - hypothesis(theta, x[iter, :]))
		else:
			cost += -np.log(hypothesis(theta, x[iter, :]))

		# cost += -1 * y[iter] * np.log(hypothesis(theta, x[iter, :])) - ( 1 - y[iter] ) * np.log(1 - hypothesis(theta, x[iter, :]))

	cost = cost / m

	return cost

def grad(theta, x, y, rate = 0):
	"""
	Градиент функции стоимости

	Args:
		theta (ndarray): параметры обучаемой модели
		x (ndarray): матрица объекты - признаки
		y (ndarray): вектор ответов
		rate: степень регуляризации

	Return:
		ndarray: вектор градиента функции стоимости
	"""

	m = y.shape[0]

	# Костыль
	grad = 0
	for iter in range(0, m - 1):
		grad += np.dot((hypothesis(theta, x[iter, :]) - y[iter]), x[iter, :])

	return  grad / m


def teach(x, y, x0):
	"""
	Обучение параметром модели

	Args:
		param (ndarray): параметры обучаемой модели
		x (ndarray): матрица объекты - признаки
		y (ndarray): вектор ответов

	Reuturns:
		ndarray: оптимальный вектор параметров модели
	"""

	# Степень регуляризации
	rate = 0

	return newtons_method (
		lambda theta: cost(theta, x, y, rate),
		lambda theta: grad(theta, x, y, rate),
		x0,
		1e-5
	)
	
	# return gradien_descent(
	# 	lambda theta: cost(theta, x, y, rate),
	# 	lambda theta: grad(theta, x, y, rate),
	# 	x0,
	# 	1e-5,
	# 	-1
	# )

def classify(param, x):
	"""
	Принятие решения о принадлежности
	
	Args:
		param: параметры обучаемой модели
		x (ndarray): матрица объекты - признаки

	Returns:
		Bool: принадлежность классу (True: 1, False: 0)
	"""

	return (np.dot(x, param) >= 0)

class LRclassifier:
	# TODO: перенести обучение в конструктор, добавить метод classify
	def __init__(self):
		pass