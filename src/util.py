# Author: Adam Robinson
import os

# Returns the dimensions of the current terminal window or a good guess if 
# something goes wrong.
def terminal_dims():
	try:
		rows, columns = os.popen('stty size', 'r').read().split()
		return int(rows), int(columns)
	except:
		return 90, 100