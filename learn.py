import numpy as np

from optimize import *

"""
Матрица объекты - признаки хранится в ndarray и имеет вид:

x = 1 .. ..
    1 .. ..
    1 .. ..

"""

def sigmoid(x):
	"""
	Сигмоид-функция
	"""

	return 1 / (1 + np.exp(-x))


def cost(param, x, y, rate = 0):
	"""
	Векторизованная функция стоимости
	"""

	m = y.size
	return (1 / m) * (
		np.dot(-y, np.log(sigmoid(np.dot(x, param)))) + 
		np.dot(-(1 - y), np.log(1 - sigmoid(np.dot(x, param)))) ) + (
		(rate / (2 * m)) * np.dot(param[1:], param[1:]) )

def grad(param, x, y, rate = 0):
	"""
	Векторизованный градиет функции стоимости
	"""

	m = y.size
	return (1 / m) * np.dot(sigmoid(np.dot(x, param)) - y, x) + (
		(rate / m) * np.hstack((0, param[1:])) )


def teach(x, y, accuracy, maxiter):
	"""
	Обучение параметров модели

	Args:f
		param (ndarray): параметры обучаемой модели
		x (ndarray): матрица объекты - признаки
		y (ndarray): вектор ответов

	Reuturns:
		ndarray: оптимальный вектор параметров модели
	"""

	# Степень регуляризации
	rate = 0
	
	param_init = np.zeros(x.shape[1])

	param, _ = gradien_descent(
		lambda param: cost(param, x, y, rate),
		lambda param: grad(param, x, y, rate),
		param_init,
		accuracy,
		maxiter
	)

	print("Optimal parameters: ", param);

	return param


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


def print_count_error(param, data, answers, percent):
	"""
	Подсчитывает процент правильного прогнозирования данных

	Args:
		param: параметры обучаемой модели
		data:  данные
		answers: ответы
		percent: процент данных, используемых в обучении

	"""

	n = answers.size
	n_learn = round(percent / 100 * n)
	n_check = n - n_learn

	classified = classify(param, data[n_learn:, :])

	n_error = 0
	for forecast, answer in zip(classified, answers[n_learn:]):
		if forecast != answer:
			n_error += 1

	print('Probability forecast: ', (1 - n_error / n_check) * 100)

class LRclassifier:
	# TODO: перенести обучение в конструктор, добавить метод classify
	def __init__(self):
		pass