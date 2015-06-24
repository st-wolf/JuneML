import numpy as np

from unittest import TestCase
from learn import cost, grad
from numpy.linalg import norm

class TestLearn(TestCase):
	def setUp(self):
		self.features = np.array(
			[ [1, 0, 0],
			  [1, 0, 1],
			  [1, 1, 0] ])
		self.answers = np.array([1, 0, 0])
		self.param = np.array([1, 2, 3])

	def test_cost(self):
		result = cost(self.param, self.features, self.answers)
		self.assertTrue(abs(result - 2.4599996556) < 1e-6)

	def test_grad(self):
		result = grad(self.param, self.features, self.answers)
		self.assertTrue(norm(result - np.array([0.5552155, 0.31752471, 0.32733793])))