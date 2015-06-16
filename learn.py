import numpy as np

from optimize import gradien_descent

"""
Матрица объекты - признаки хранится в ndarray и имеет вид:

X = 1 .. ..
    1 .. ..
    1 .. ..

"""

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def cost(param, X, y, rate = 0):
	"""
	Вычисление функции стоимости

	Args:
		param (ndarray): параметры обучаемой модели
		X (ndarray): матрица объекты - признаки
		y (ndarray): вектор ответов
		rate: степень регуляризации

	Return:
		float: значение функции стоимости
	"""

	m = len(y)
	return (1 / m) * (
		np.dot(-y, np.log(sigmoid(np.dot(X, param)))) + 
		np.dot(-(1 - y), np.log(1 - sigmoid(np.dot(X, param)))) ) + (
		(rate / (2*m)) * np.dot(param[1:], param[1:]) )

def grad(param, X, y, rate = 0):
	"""
	Градиет функции стоимости

	Args:
		param (ndarray): параметры обучаемой модели
		X (ndarray): матрица объекты - признаки
		y (ndarray): вектор ответов
		rate: степень регуляризации

	Return:
		ndarray: вектор градиента функции стоимости
	"""

	m = len(y)
	return (1 / m) * np.dot(sigmoid(np.dot(X, param)), X) + (
		(rate / m) * np.hstack((0, param[1:])) )

def teach(X, y):
	"""
	Обучение параметром модели

	Args:
		X (ndarray): матрица объекты - признаки
		y (ndarray): вектор ответов

	Reutrns:
		ndarray: оптимальынй вектор параметров модели
	"""

	# Степень регуляризации
	rate = 0
	return gradien_descent(
		lambda param: cost(param, X, y, rate),
		lambda param: grad(param, X, y, rate),
		np.zeros(X.shape[1])
		)

def classify(param, X):
	"""
	Принятие решения о принадлежности
	
	Args:
		param: параметры обучаемой модели
		X (ndarray): матрица объекты - признаки

	Reutrns:
		Bool: принадлежность классу (True: 1, False: 0)
	"""

	return (np.dot(X, param) >= 0)

class LRclassifier:
	# TODO: перенести обучение в конструктор, добавить метод classify
	def __init__(self):
		pass