import matplotlib.pyplot as plt
import numpy             as np
import sys

def load(f):
	with open(f, 'r') as file:
		raw = file.read()

	lines = [i for i in raw.split('\n') if i != '' and not i.isspace()]

	x = []
	y = []
	for line in lines:
		xi, yi = line.split(' ')
		x.append(int(xi))
		y.append(float(yi))

	return x, y

if __name__ == '__main__':
	data = []
	for file in sys.argv[1:]:
		data.append(load(f))

	fig, ax = plt.subplots(1, len(data))

	for axi in range(len(data)):
		ax[axi].scatter(*data[axi], s=4)

	plt.show()