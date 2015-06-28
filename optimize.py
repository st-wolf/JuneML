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

def gradient_descent(f, grad, x0, xtol = 1e-10, maxiter = 200):
	""" 
	Метод градиентного спуска
 
	Args:
		f: Функция одного аргумента типа ndarray
		grad: Градиент функции f
		x0 (ndarray): Начальные условия градиентного спуска
		xtol : Критерий останова по точности
		maxiter (int): Максимальное количество итераций метода градиентного спуска.

	Returns:
		(ndarray, ndarray): результат градиентного спуска, массив значений f на итерациях
	"""

	alpha = 100
	xmin = x0
	fmin = f(xmin)

	fvalues = [fmin]

	n = 0 

	while (n < maxiter) or (maxiter < 0):
		n += 1
		x = xmin - alpha * grad(xmin)

		cost_norm = norm(grad(x))
		print(cost_norm)
		dx = norm(x - xmin)

		if (cost_norm < xtol) or (dx == 0):
			break
		elif f(x) >= fmin:
			alpha /= 2
		else:
			xmin = x
			fmin = f(x)
			fvalues.append(fmin)

	print("Gradient descent: cost_norm = %.4g after %i iterations" % (cost_norm, n))
	
	return (x, fvalues)




