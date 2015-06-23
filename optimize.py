# Тут лежит куча алгоритмов минимизации
# from scipy.optimize import minimize

from numpy.linalg import norm

def newtons_method(func, grad, x0, xtol = 1e-5):
	"""
	Метод Ньютона

	Args:
		func: Функция одного аргумента типа ndarray
		grad: Градиент функции f
		x0 (ndarray): Начальные условия градиентного спуска
		xtol : Критерий останова по точности

	Returns:
		ndarray: результат градиентного спуска
	"""

	x_new = x0

	while abs(func(x_new)) >= xtol:
		x_old = x_new
		x_new = x_old - func(x_old)/grad(x_old)

	return x_new


# Осваиваю Google Code Style for Python

def gradien_descent(f, grad, x0, xtol = 1e-5, maxiter = 100):
	""" 
	Метод градиентного спуска
 
	Args:
		f: Функция одного аргумента типа ndarray
		grad: Градиент функции f
		x0 (ndarray): Начальные условия градиентного спуска
		xtol : Критерий останова по точности
		maxiter (int): Максимальное количество итераций метода градиентного спуска.
			Значение -1 - без ограничений

	Returns:
		ndarray: результат градиентного спуска
	"""

	alpha = 10
	xmin = x0
	fmin = f(xmin)
	n = 0 

	while (n < maxiter) if (maxiter >= 0) else True:
		n += 1
		x = xmin - alpha * grad(xmin)
		print(norm(x - xmin), n)
		if norm(x - xmin) < xtol:
			return x
		elif f(x) >= fmin:
			alpha /= 2
		else:
			xmin = x
	

	print("Required accuracy is not achieved")
	return xmin




