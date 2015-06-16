import numpy as np

from unittest import TestCase
from view import show_classes

# TODO: подумать, как писать тесты на view
class TestView(TestCase):
	def test_show_classes(self):
		x1 = np.array([1, 2, 3, 4])
		x2 = np.array([2, 3, 1, 4])
		y  = np.array([1, 0, 0, 1])
		names = ['feature_1', 'feature_2']

		# Если внутри будет выброшено исключение, оно отобразится в консоле
		self.assertIsNone(show_classes(x1, x2, y, names = names))
