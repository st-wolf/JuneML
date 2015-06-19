# Тут лежит куча алгоритмов минимизации
# from scipy.optimize import minimize

from numpy.linalg import norm

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

	alpha = 1
	xmin = x0
	fmin = f(xmin)
	n = 0 

	while (n < maxiter) if (maxiter >= 0) else True:

		n += 1
		x = xmin - alpha * grad(xmin)
		if norm(x - xmin) < xtol:
			return x
		elif f(x) >= fmin:
			alpha /= 2
		else:
			xmin = x
	

	# print("Required accuracy is not achieved: x - xmin = %f\n" % (x - xmin))
	return xmin

# Надо разбираться
# def stochastic_gradient(...):