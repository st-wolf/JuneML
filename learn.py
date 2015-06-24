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


def cost(param, X, y, rate = 0):
	"""
	Векторизованная функция стоимости
	"""

	m = y.size
	return (1 / m) * (
		np.dot(-y, np.log(sigmoid(np.dot(X, param)))) + 
		np.dot(-(1 - y), np.log(1 - sigmoid(np.dot(X, param)))) ) + (
		(rate / (2*m)) * np.dot(param[1:], param[1:]) )

def grad(param, X, y, rate = 0):
	"""
	Векторизованный градиет функции стоимости
	"""

	m = y.size
	return (1 / m) * np.dot(sigmoid(np.dot(X, param)), X) + (
		(rate / m) * np.hstack((0, param[1:])) )

def teach(X, y):
	"""
	Обучение параметров модели

	Args:
		param (ndarray): параметры обучаемой модели
		X (ndarray): матрица объекты - признаки
		y (ndarray): вектор ответов

	Reuturns:
		ndarray: оптимальный вектор параметров модели
	"""

	# Степень регуляризации
	rate = 0
	
	param_init = np.zeros(param.size)
	return gradien_descent(
		lambda param: cost(param, X, y, rate),
		lambda param: grad(param, X, y, rate),
		param_init,
		1e-5,
		-1 )

def classify(param, X):
	"""
	Принятие решения о принадлежности
	
	Args:
		param: параметры обучаемой модели
		X (ndarray): матрица объекты - признаки

	Returns:
		Bool: принадлежность классу (True: 1, False: 0)
	"""

	return (np.dot(X, param) >= 0)

class LRclassifier:
	# TODO: перенести обучение в конструктор, добавить метод classify
	def __init__(self):
		pass