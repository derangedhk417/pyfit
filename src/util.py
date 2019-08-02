# Authors: Adam Robinson, James Hickman
import os
import sys
import numpy as np
import time
from datetime import datetime


import atexit

log_instances = []

class Log:
	def __init__(self, file_name, max_col=80, tab_size=4):
		self.file     = open(file_name, 'w', 1024*10)
		self.max_col  = max_col
		self._indent  = 0
		self.tab_size = 4

		log_instances.append(self)

	def log(self, text):
		# Split the logged information on spaces and add appropriate newline
		# characters and spaces to emulate tabs.
		words = text.split(' ')

		lines = []
		idx   = 0
		while idx < len(words):
			current_line = ' '*(self._indent * self.tab_size)

			while len(current_line) < self.max_col and idx < len(words):
				if len(current_line) + len(words[idx]) + 1 < self.max_col:
					current_line += words[idx] + ' '
					idx          += 1
				else:
					break

			lines.append(current_line)

		self.file.write('\n'.join(lines) + '\n')

	def indent(self):
		self._indent += 1

	def unindent(self):
		self._indent -= 1

	def __del__(self=None):
		try:
			self.file.close()
		except:
			print("Failed to close log file. It may be incomplete.")

@atexit.register
def cleanup():
	for log in log_instances:
		del log

# Returns the dimensions of the current terminal window or a good guess if 
# something goes wrong.
def terminal_dims():
	try:
		rows, columns = os.popen('stty size', 'r').read().split()
		return int(rows), int(columns)
	except:
		return 90, 100

class ProgressBar:
	def __init__(self, prefix, prefix_width, total, update_every = 10):
		self.prefix        = prefix
		self.prefix_width  = prefix_width
		self.update_every  = update_every
		self.total         = total
		self.current       = 0.0
		self.last          = 0.0
		self.remaining     = 0
		self.width         = terminal_dims()[1] - 55
		self.estimate      = True
		self.start_time    = datetime.now()
		self.update_count  = 0
		self.times         = [] # Timing of chunks of work.
		self.sizes         = [] # Amount of work in each chunk.
		self.display()

		self.last_time = time.time()


	def update(self, value):
		self.current      =  value
		self.update_count += 1

		if self.update_count % self.update_every == 0 or self.update_count == 1:
			work = int(self.current - self.last)
			should_estimate = (datetime.now() - self.start_time).seconds > 15
			if work != 0:
				self.last      = self.current
				timenow        = time.time()
				timing         = timenow - self.last_time
				self.last_time = timenow

				self.times.append(timing)
				self.sizes.append(work)

				# Convert to numpy arrays and calculate the average
				# time per unit work.
				times = np.array(self.times)
				works = np.array(self.sizes)

				avg = (times / works).mean()

				# Figure out how much work is left.
				self.remaining = self.total - self.current
				self.remaining = self.remaining * avg

			self.display(est=should_estimate)

	def finish(self):
		self.current = self.total
		total_time   = (datetime.now() - self.start_time).seconds
		seconds      = int(total_time % 60)
		minutes      = int(np.floor(total_time / 60))
		time         = ' (%02i:%02i elapsed)'%(minutes, seconds)
		self.display(_end='')
		print(time, end='')
		print('\n', end='')
		self.ttc = total_time

	# This function returns a tuple with the first member being the
	# percentage to display and the second number being the number
	# of ticks to draw in the progress bar.
	def get_display(self):
		percentage = (self.current / self.total) * 100
		ticks      = int(np.floor((self.current / self.total) * self.width))
		return (ticks, percentage)

	def display(self, est=False, _end='\r'):
		ticks, percentage = self.get_display()
		fill   = '='  * ticks
		space  = ' ' * (self.width - ticks)
		disp   = '%' + '%05.2f'%(percentage)

		rem_seconds = int(self.remaining)
		rem_minutes = rem_seconds // 60
		rem_seconds = rem_seconds % 60
		rem_hours   = rem_minutes // 60
		rem_minutes = rem_minutes % 60
		rem         = '%02i:%02i:%02i rem.'%(
			rem_hours, 
			rem_minutes, 
			rem_seconds
		)

		if self.current == self.total or not self.estimate or not est:
			rem = ''

		prefix = self.prefix + (' ' * (self.prefix_width - len(self.prefix)))


		# This is the only consistent way to clear the current line.
		# Using a \r character at the end of the line only works for
		# some environments. Odds are that this will not work on windows
		# but who cares.
		sys.stdout.write("\033[K")
		print(prefix + '[' + fill + space + ']' + ' ' + disp + ' ' + rem, end=_end)