# Authors: Adam Robinson, James Hickman
import os
import sys
import numpy as np
import time
from datetime import datetime

import code

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
		self.start_time    = datetime.now()
		self.update_count  = 0
		self.times         = [] # Timing of chunks of work.
		self.sizes         = [] # Amount of work in each chunk.
		self.display()

		self.last_time = time.time_ns()


	def update(self, value):
		self.current      =  value
		self.update_count += 1

		work           = int(self.current - self.last)
		if work != 0:
			self.last      = self.current
			timenow        = time.time_ns()
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

		if self.update_count % self.update_every == 0 or self.update_count == 1:
			self.display()

	def finish(self):
		self.current = self.total
		total_time   = (datetime.now() - self.start_time).seconds
		seconds      = int(total_time % 60)
		minutes      = int(np.floor(total_time / 60))
		time         = ' (%02i:%02i elapsed)'%(minutes, seconds)
		self.display(_end='')
		print(time, end='')
		print('\n', end='')

	# This function returns a tuple with the first member being the
	# percentage to display and the second number being the number
	# of ticks to draw in the progress bar.
	def get_display(self):
		percentage = (self.current / self.total) * 100
		ticks      = int(np.floor((self.current / self.total) * 35))
		return (ticks, percentage)

	def display(self, _end='\r'):
		ticks, percentage = self.get_display()
		fill   = '='  * ticks
		space  = ' ' * (35 - ticks)
		disp   = '%' + '%05.2f'%(percentage)

		rem_seconds = int(self.remaining // int(1e9))
		rem_minutes = rem_seconds // 60
		rem_seconds = rem_seconds % 60
		rem_hours   = rem_minutes // 60
		rem_minutes = rem_minutes % 60
		rem         = '%02i:%02i:%02i rem.'%(
			rem_hours, 
			rem_minutes, 
			rem_seconds
		)

		if self.current == self.total:
			rem = ''

		prefix = self.prefix + (' ' * (self.prefix_width - len(self.prefix)))


		# This is the only consistent way to clear the current line.
		# Using a \r character at the end of the line only works for
		# some environments. Odds are that this will not work on windows
		# but who cares.
		sys.stdout.write("\033[K")
		print(prefix + '[' + fill + space + ']' + ' ' + disp + ' ' + rem, end=_end)