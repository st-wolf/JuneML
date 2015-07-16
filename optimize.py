# Тут лежит куча алгоритмов минимизации
# from scipy.optimize import minimize

import numpy as np

from numpy.linalg import norm


# Prototype: optimize functon calls, think about merging of zoom and line_search
def line_search(func, diff, t0, tmax, maxiter = 100):
	"""
	Алгоритм линейной оптимизации по книге Nocedal, Wright - Numerical optimization.
	Поиск точки, удовлетворяющей строгим уловиям Вольфе с константами (1e-4, -0.9).
	Происходит минимизация вдоль некоторого ветора h.
	Функция: f(t) = f(x + t*h) и производная f'(t) = df / dt определяются снаружи.
	Для этого создается lambda - функция одного оргумента t.

	Args:
		func: Функция одного аргумента t - длины шага линейного поиска
		diff: Производная функции f
		t0: Первое приближение к оптимальному шагу
		tmax: Максимальный шаг
		maxiter: Максимальное число итераций

	Returns:
		Длина шага, удовлетворяющая строгим условиям Вольфе
	"""

	# Параметры строгих условий Вольфе
	c = (1e-4, 0.1)

	# Последовательность приближений к решению задачи
	t = [0, t0]

	# Счетчик количества итераций
	i = 1

	while True:
		if (func(t[i]) > func(0) + c[0] * t[i] * diff(0)) or \
			(func(t[i]) >= func(t[i-1])):

			return zoom(func, diff, t[i-1], t[i])

		if abs(diff(t[i])) <= c[1] * abs(diff(0)):
			return t[i]

		if diff(t[i]) >= 0:
			# Можно поменять t[i] и t[i-1] местами.
			# В книге реализация выглядит так, чтобы поддерживать выполнение
			# условия diff(t_low) * (t_hight - t_low) < 0
			# На мой взгляд, оно исключительно косметическое
			return zoom(func, diff, t[i], t[i-1])

		tnew = 2 * t[i]
		if tnew > tmax:
			tnew = tmax

		t.append(tnew)

		if i == maxiter:
			# Строгие условия Волфе не выполнены, однако условие убывания выполняется
			# Вообще, это условие нужно убрать, и так все должно работать.
			print("Line search: diff = %.15f after %i iterations" % (diff(t[i]), i))
			return t[i]

		i += 1


def zoom(func, diff, t_low, t_hight, maxiter = 100):
	"""
	Вспомогательная функция к алгоритму линейной оптимизации

	Args:
		func: Функция одного аргумента t - длины шага линейного поиска
		diff: Производная функции f
		t_low, t_hight: Значения по t между которыми заключена оптимальная длина шага
		maxiter: Максимальное число итераций

	Returns:
		Длина шага, удовлетворяющая строгим условиям Вольфе
	"""

	# Параметры строгих условий Вольфе
	c = (1e-4, 0.1)

	# Счетчик количества итераций
	i = 1

	while True:
		# Пока пользуемся бисекцией, в дальнейшем заменим на интерполяцию
		# кубическим многочленом
		t = (t_low + t_hight) / 2

		# print(t_low, t_hight)

		if (func(t) > func(0) + c[0] * t * diff(0)) or (func(t) > func(t_low)):
			t_hight = t
		else:
			if abs(diff(t)) <= -c[1] * diff(0):
				return t
			
			# От этого условия можно избавиться
			if diff(t) * (t_hight - t_low) >= 0:
				t_hight = t_low

			t_low = t

		if i == maxiter:
			print("Zoom: diff = %.15f after %i iterations" % (diff(t), i))
			return t

		i += 1


# Prototype: optimize function calls, use spetial formule for t0
def conjugate_gradients(func, grad, x0, gtol, maxiter = 200):
	"""
	Метод сопряженных градиентов с линейным поиском по правилу Вольфе

	Args:
		func: Функция одного аргумента типа ndarray
		grad: Градиент функции func
		x0 (ndarray): Начальные условия градиентного спуска
		gtol: Критерий останова по малости градиента
		maxiter (int): Максимальное количество итераций метода

	Returns:
		(ndarray, ndarray): результат градиентного спуска, массив значений func
			на итерациях
	"""

	# Размерность задачи - количество сопряженных направлений
	n = x0.size

	# Последовательность приближений к решению задачи
	x = [x0]

	# Счетчик количества итераций
	i = 1

	direction = -grad(x0)
	offset = 0

	while True:
		# Линейный поиск вдоль направления direction
		# В качетсве первого приближения к оптимальному шагу берем 1.
		# Это нормально для ньютоновских методов, однако авторитетная литература говорит,
		# что для сопряженных градиентов такое приближение является плохим.
		# Для сравнения можно поробовать использовать формулу:

		# t0 = step * np.dot(grad(x[i-1]), direction[i-1]) / np.dot(grad(x[i]), direction[i])

		step = line_search(
			func = lambda t: func(x[i-1] + t * direction),
			diff = lambda t: np.dot(grad(x[i-1] + t * direction), direction),
			t0 = 1,
			tmax = 1e5
			)

		x.append(x[i-1] + step * direction)

		#print('---', direction, x[i])

		if norm(grad(x[i])) < gtol:
			return x[i]

		if i == maxiter:
			print("Conjugent gradients: norm(grad) = %.15f after %i iterations" %
				(norm(grad(x[i])), i))
			return x[i]

		# Процедура обновления направления спуска
		if i % n == 0:
			direction = -grad(x[i])
		else:
			offset = np.dot(grad(x[i]), grad(x[i]) - grad(x[i-1])) / norm(grad(x[i-1])) ** 2
			direction = -grad(x[i]) + offset * direction

		i += 1


def gradient_descent(f, grad, x0, xtol = 1e-10, maxiter = 200):
	""" 
	Метод градиентного спуска
 
	Args:
		f: Функция одного аргумента типа ndarray
		grad: Градиент функции f
		x0 (ndarray): Начальные условия градиентного спуска
		xtol : Критерий останова по точности
		maxiter (int): Максимальное количество итераций метода градиентного спуска

	Returns:
		(ndarray, ndarray): результат градиентного спуска, массив значений f
			на итерациях
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




