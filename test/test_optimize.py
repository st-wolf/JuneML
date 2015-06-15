from unittest import TestCase
from optimize import gradien_descent

class TestOptimize(TestCase):
	def setUp(self):
		# Определяем набор тестовых функций
		f_quad = {
			"func": lambda x: x ** 2,
			"grad": lambda x: 2*x,
			"init": 1,
			"min": 0,
			"xtol": 1e-6
		}
		self.functions = [f_quad]

	def test_gradient_descent(self):
		for f in self.functions:
			xmin = gradien_descent(f["func"], f["grad"], f["init"])
			print(abs(xmin - f["min"]))
			self.assertFalse(abs(xmin - f["min"]) > f["xtol"])