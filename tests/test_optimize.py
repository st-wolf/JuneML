import numpy as np

# Может быть сделать все импорты как from ... import item1, item2 ?
from unittest import TestCase
from optimize import gradien_descent
from numpy.linalg import norm

class TestOptimize(TestCase):
	def setUp(self):
		# Определяем набор тестовых функций
		f_quad = {
			"func": lambda x: x ** 2,
			"grad": lambda x: 2*x,
			"init": 1,
			"xmin": 0,
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
			"func": lambda x: np.tan(x ** 2),
			"grad": lambda x: 2*x / (np.cos(x ** 2) ** 2),
			"init": np.pi / 4,
			"xmin": 0,
			"xtol": 1e-6
		}

		self.functions = [f_quad, f_3d_quad, f_tan_of_squared]

	def test_gradient_descent(self):
		for f in self.functions:
			xmin = gradien_descent(f["func"], f["grad"], f["init"], f["xtol"])
			print(norm(xmin - f["xmin"]))
			self.assertTrue(norm(xmin - f["xmin"]) < f["xtol"])