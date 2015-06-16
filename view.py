import matplotlib.pyplot as plt

def decision_boundary(param, *args):
	# Рисует проекцию разделяющей поверхности на плоскость двух параметров
	pass

def show_classes(x1, x2, y, names = []):
	# Рисует классы на плоскости двух параметров

	# Цветовая схема от Andrew Ng
	plt.scatter(x1[y == 0], x2[y == 0], s = 40, c = 'yellow')
	plt.scatter(x1[y == 1], x2[y == 1], s = 100, c = 'black', marker = '+', linewidth = 2)
	plt.legend(['died', 'survived'])

	if len(names) == 2:
		plt.xlabel(names[0])
		plt.ylabel(names[1])

	plt.grid()
	plt.show()