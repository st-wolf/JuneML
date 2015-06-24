# Тут лежит куча алгоритмов минимизации
# from scipy.optimize import minimize

from numpy.linalg import norm

def newtons_method(func, grad, x0, xtol = 1e-10, maxiter = 200):
	"""
	Метод Ньютона

	Args:
		func: Функция одного аргумента типа ndarray
		grad: Градиент функции f
		x0 (ndarray): Начальные условия градиентного спуска
		xtol : Критерий останова по точности

	Returns:
		ndarray: результат ньютоновской оптимизации
	"""

	pass

# Осваиваю Google Code Style for Python

def gradien_descent(f, grad, x0, xtol = 1e-10, maxiter = 200):
	""" 
	Метод градиентного спуска
 
	Args:
		f: Функция одного аргумента типа ndarray
		grad: Градиент функции f
		x0 (ndarray): Начальные условия градиентного спуска
		xtol : Критерий останова по точности
		maxiter (int): Максимальное количество итераций метода градиентного спуска.

	Returns:
		ndarray: результат градиентного спуска
	"""

	alpha = 100
	xmin = x0
	fmin = f(xmin)
	n = 0 

	while n < maxiter:
		n += 1
		x = xmin - alpha * grad(xmin)

		dx = norm(x - xmin)
		if dx < xtol:
			return x
		elif f(x) >= fmin:
			alpha /= 2
		else:
			xmin = x
	
	print("Gradiend descent: dx = %.15f after %i iterations" % (dx, maxiter))
	return xmin




