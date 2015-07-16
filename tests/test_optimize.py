import numpy as np
import random
# Может быть сделать все импорты как from ... import item1, item2 ?

from unittest import TestCase, skip
from optimize import gradient_descent, line_search, conjugate_gradients
from numpy.linalg import norm


# TODO: использовать ООП, Люк.
def set_direction(f, x0, direction):
	"""
	Возвращает функцию, для которой будет выполнен линейный поиск
	"""
	return {
		"func": lambda t: f["func"](x0 + t * direction),
		"diff": lambda t: np.dot(f["grad"](x0 + t * direction), direction),
		"init": x0,
		"direction": direction
	}


# TODO: создать класс Counter, который примет в конструктор фукцию и будет считать количесвто вызовов
# перегрузить оператор (), передавать вместо функций и посмотреть количество вызовов
class Counter:
	pass


class TestOptimize(TestCase):
	def setUp(self):
		# Определяем набор тестовых функций для многомерной минимизации
		f_quad = {
			"func": lambda x: x ** 2,
			"grad": lambda x: 2*x,
			"init": np.array(1),
			"xmin": np.array(0),
			"xtol": 1e-6
		}

		f_3d_quad = {
			"func": lambda x: (x[0] ** 2) + (x[1] ** 2) + (x[2] ** 2) + 
				x[0]*x[1] + x[2] + 10,
			"grad": lambda x: np.array([2*x[0] + x[1], 2*x[1] + x[0], 2*x[2] + 1]),
			"init": np.array([10, 10, 10]),
			"xmin": np.array([0, 0, -0.5]),
			"xtol": 1e-6
		}

		f_tan_of_squared = {
			"func": lambda x: np.tan(x ** 2) ** 2,
			"grad": lambda x: 4*x * np.tan(x ** 2) / (np.cos(x ** 2) ** 2),
			"init": np.array(np.pi / 8),
			"xmin": 0,
			"xtol": 1e-3
		}

		f_cos_of_sum = {
			"func": lambda x: np.cos(x[0] + x[1]),
			"grad": lambda x: np.hstack([-np.sin(x[0] + x[1]), -np.sin(x[0] + x[1])]),
			"init": np.array([0, 0]),
			# Нужно использовать fmin вместо xmin
			# Нужно искользовать ftol вместо xtol
		}

		# Определяем набор направлений для алгоритма линейного поиска
		directions = [np.array(1), np.array([1, 1]), np.array([1, 2])]

		# Определяем набор стартовых позиций для алгоритма линейного поиска
		initials = [np.array(0), np.array([0.1, 0.1]), np.array([10, 10])]

		# Определяем набор тестовых функций для алгоритма линейного поиска
		f_cos_times_quad = {
			"func": lambda x: (x ** 2) * np.cos(x),
			"diff": lambda x: 2 * x * np.cos(x) + (x ** 2) * np.sin(x),
			"init": 5
		}

		f_directed_cos_of_sum = set_direction(f_cos_of_sum, initials[1], directions[2])

		self.linear_functions = [f_directed_cos_of_sum]
		self.functions = [f_quad, f_3d_quad, f_tan_of_squared]

	@skip("Not tested")
	def test_gradient_descent(self):
		for f in self.functions:
			xmin = gradient_descent(f["func"], f["grad"], f["init"], maxiter = 200)
			print(norm(xmin - f["xmin"]))
			self.assertTrue(norm(xmin - f["xmin"]) < f["xtol"])

	@skip("Not tested")
	def test_line_search(self):
		# Пока нет четкого критерия корректности
		# Проверяем, как хорошо достигаются условия Вольфе
		# Параметры метода (1e-4, 0.1) забиты наглухо - не надо их менять
		for f in self.linear_functions:
			# Первое приближение к искомой длине шага
			t0 = random.uniform(0, 10)
			trule = line_search(f["func"], f["diff"], t0, 1000)
			print("\n")
			print("t optimal: ", trule)
			print("Start: ", "Func: ", f["func"](0), "Diff: ", f["diff"](0))
			print("End: ", "Func: ", f["func"](trule), "Diff: ", f["diff"](trule))
			print("Point: ", f["init"] + trule * f["direction"])


	def test_conjugent_gradients(self):
		for f in self.functions:
			xmin = conjugate_gradients(f["func"], f["grad"], f["init"], gtol = 1e-9)
			print(norm(xmin - f["xmin"]))
			self.assertTrue(norm(xmin - f["xmin"]) < f["xtol"])