import collections
import json
import pathlib
import os
import sys
import threading
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np

EPS = 1e-7


class AtomicInteger:
	def __init__(self):
		self._value = 0
		self._lock = threading.Lock()

	def inc(self):
		with self._lock:
			self._value += 1
			return self._value

	def dec(self):
		with self._lock:
			self._value -= 1
			return self._value

	def reset(self):
		self._value = 0

	@property
	def value(self):
		return self._value


def get_running_avg(x, dist=100):
	x = np.array(x)
	n = len(x)-dist
	running_avg = np.empty(n)
	for t in range(n):
		b = t - dist
		f = t + 1
		if b < 0:
			f -= b
			b = 0
		running_avg[t] = x[b:f].mean()
	return running_avg


def plot_running_avg(x, title='Running Average'):
	running_avg = get_running_avg(x)
	plt.plot(running_avg)
	plt.title(title)
	plt.show()


def plot_running_avgs(xrl, xopt):
	plt.plot(get_running_avg(xrl), label='RL')
	plt.plot(get_running_avg(xopt), label='Optimum')
	plt.xlabel('episodes')
	plt.ylabel('distance moving average')
	plt.title('Distance Evolution Throughout Episodes')
	plt.legend()
	file = 'saved_models/' + 'dist_evolution' + '.eps'
	plt.savefig(file, format='eps', dpi=1000)
	plt.show()


def plot(x, title=''):
	plt.plot(x)
	plt.title(title)
	plt.show()


def plot_xy(x, y, title=''):
	plt.plot(x, y)
	plt.title(title)
	plt.show()


def plot_x_and_avg(x, title='X and Running Average'):
	plt.plot(x)
	plt.plot(get_running_avg(x))
	plt.title(title)
	plt.show()


def reverse_subarray(v, i, j):
	while i < j:
		v[i], v[j] = v[j], v[i]
		i += 1
		j -= 1


def count_breakpoints_reduced(v, i, j):
	n = len(v)
	prev = v[i - 1] if i > 0 else -1
	next = v[j + 1] if j < n - 1 else n
	breakpoints_reduced = 0
	if abs(v[i] - prev) != 1:
		breakpoints_reduced += 1
	if abs(v[j] - next) != 1:
		breakpoints_reduced += 1
	if abs(v[j] - prev) != 1:
		breakpoints_reduced -= 1
	if abs(v[i] - next) != 1:
		breakpoints_reduced -= 1
	return breakpoints_reduced


def count_breakpoints_reduced_transposition(v, i, j, k):
	n = len(v)
	prev = v[i - 1] if i > 0 else -1
	next = v[k + 1] if k < n - 1 else n
	breakpoints_reduced = 0
	if abs(v[i] - prev) != 1:
		breakpoints_reduced += 1
	if abs(v[k] - next) != 1:
		breakpoints_reduced += 1
	if abs(v[j + 1] - v[j]) != 1:
		breakpoints_reduced += 1
	if abs(v[j + 1] - prev) != 1:
		breakpoints_reduced -= 1
	if abs(v[j] - next) != 1:
		breakpoints_reduced -= 1
	if abs(v[i] - v[k]) != 1:
		breakpoints_reduced -= 1
	return breakpoints_reduced

def get_action_greedy(p):
	n = len(p)
	for i in range(n - 1):
		for j in range(i + 1, n):
			score = score_rev(v, i, j)
			if max_score is None or score > max_score:
				max_score = score
				x, y = i, j

def score_rev(v, i, j):
	n = len(v)
	breakpoints_reduced = count_breakpoints_reduced(v, i, j)
	for r in (range(1, i), range(j + 1, n)):
		for idx in r:
			if v[idx] == v[idx - 1] - 1:
				return breakpoints_reduced, 1
	for idx in range(i + 1, j + 1):
		if v[idx] == v[idx - 1] + 1:
			return breakpoints_reduced, 1
	return breakpoints_reduced, 1 if (i > 0 and v[j] < v[i - 1]) or (j < n - 1 and v[i] > v[j + 1]) else 0


def breakpoints(v):
	n = len(v)
	ans = 0
	for i in range(1, n):
		if abs(v[i] - v[i - 1]) != 1:
			ans += 1
	if v[0] != 0:
		ans += 1
	if v[n - 1] != n - 1:
		ans += 1
	return ans


def greedy_reversal_sort(v, trace=None):
	v = v.copy()
	if trace is not None:
		trace.append(v.copy())
	n = len(v)
	b = breakpoints(v)
	ans = 0
	while b > 0:
		max_score = None
		x, y = None, None
		for i in range(n - 1):
			for j in range(i + 1, n):
				score = score_rev(v, i, j)
				if max_score is None or score > max_score:
					max_score = score
					x, y = i, j
		b -= max_score[0]
		reverse_subarray(v, x, y)
		if trace is not None:
			trace.append(v.copy())
		ans += 1
	return ans


def v_upperbound(state, gamma):
	steps = greedy_reversal_sort(state)
	ans = (np.float_power(gamma, steps) - 1) / (gamma - 1)
	return -ans


def v_upperbound_breakpoints(state, i, j, k=-1, gamma=0.99):
	if k == -1:
		breakpoints_reduced = 1 - count_breakpoints_reduced(state, i, j)
	else:
		breakpoints_reduced = 1 - count_breakpoints_reduced_transposition(state, i, j, k)
	return -breakpoints_reduced


class Eps1:
	def __init__(self):
		self._eps = 1
		self._min = 0.005
		self._decay = 0.99

	def eps(self, i):
		if self._eps > self._min:
			self._eps *= self._decay
		return self._eps


def eps2(i):
	return 1.5 / np.sqrt(i + 1)


def eps3(i):
	return 1 * (0.9 ** i)


class PermutationExactSolver:
	def __init__(self, n):
		path = 'exact_%d' % n
		self._ans = from_file(path)
		if self._ans is None:
			cur = np.arange(n)
			self._ans = {self._to_string(cur): 0}
			q = deque()
			q.append(cur)
			while len(q) > 0:
				cur = q.popleft()
				d = self._ans[self._to_string(cur)]
				for i in range(n - 1):
					for j in range(i + 1, n):
						reverse_subarray(cur, i, j)
						cur_str = self._to_string(cur)
						if cur_str not in self._ans:
							self._ans[cur_str] = d + 1
							q.append(cur.copy())
						reverse_subarray(cur, i, j)
			ensure_saved_models_dir()
			to_file(path, self._ans)

	@staticmethod
	def _to_string(a):
		return np.array2string(a)

	def solve(self, perm):
		return self._ans[self._to_string(perm)]


def default(o):
	if isinstance(o, np.int64):
		return int(o)
	raise TypeError


def to_file(path, entry):
	path = 'saved_models/' + path + '.json'
	with open(path, 'w') as f:
		f.write(json.dumps(entry, default=default))


def from_file(path):
	path = 'saved_models/' + path + '.json'
	try:
		with open(path, 'r') as f:
			return json.loads(f.read())
	except IOError:
		return None


def ensure_saved_models_dir():
	if sys.version_info[0] < 3:
		try:
			os.makedirs('saved_models')
			os.makedirs('train_data')
		except OSError:
			if not os.path.isdir('saved_models') or not os.path.isdir('train_data'):
				raise
	else:
		pathlib.Path('saved_models').mkdir(exist_ok=True)
		pathlib.Path('train_data').mkdir(exist_ok=True)


# Plot related methods for the report


def plot_dashed(xs, ys, labels=None, xlabel='', ylabel='', file='chart'):
	if not isinstance(xs, collections.Iterable):
		xs = [xs]
		ys = [ys]
	if labels is None:
		labels = [None] * len(xs)
	for x, y, label in zip(xs, ys, labels):
		plt.plot(x, get_running_avg(y, dist=60), label=label)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend()
	file = 'saved_models/' + file + '.eps'
	plt.savefig(file, format='eps', dpi=1000)
	plt.show()


def compare_agent_to_solver(agent, solver, its=100, solve_its=100, plot_result=True, exploit_greedy_trace=False):
	scores = np.empty(its)
	for i in range(its):
		permutation = agent.env.observation_space.sample()
		base_result = solver(permutation)
		rl = agent.solve(permutation, exploit_greedy_trace=exploit_greedy_trace, its=solve_its)
		scores[i] = rl / (base_result + EPS)
		print('It:', i, ' Ratio: %.3f' % scores[i])
	if plot_result:
		plot(scores)
		plot_running_avg(scores)
	scores_mean = scores.mean()
	print('Mean = %.3f' % scores_mean)
	return scores_mean


def generate_fixed(m, n=10):
	fixed = []
	for i in range(m):
		fixed.append(list(np.random.permutation(n)))
	to_file('fixed', fixed)


def save_ans(solve, label):
	fixed = from_file('fixed')
	# fixed = fixed[:100]
	ans = []
	for i, perm in enumerate(fixed):
		perm = np.array(perm)
		if i % 10 == 0:
			print('%.2f %%' % (i / len(fixed) * 100))
		ans.append(solve(perm))
	to_file('%s_ans' % label, ans)


def compare(labels):
	fixed = from_file('fixed')
	# fixed = fixed[:100]
	xs = [range(len(fixed))] * len(labels)
	ys = []
	for label in labels:
		y = from_file(label + '_ans')
		ys.append(y)
	plot_dashed(xs, ys, labels)


def plot_type1():
	ygreedy = np.array(from_file('greedy' + '_ans'))
	yrl = np.array(from_file('rl' + '_ans'))
	yexact = np.array(from_file('exact' + '_ans'))
	ygreedy = ygreedy / yexact
	yrl = yrl / yexact

	print(ygreedy.mean())
	print(yrl.mean())

	xs = [range(len(yrl))] * 2
	plot_dashed(
		xs, (ygreedy, yrl), ('Kececioglu and Sankoff', 'RL'),
		xlabel='simulations', ylabel='performance ratio', file='test')


def plot_type2():
	flavio = np.array(from_file('flaviostate' + '_ans'))
	onehot = np.array(from_file('onehot' + '_ans'))
	maxstate = np.array(from_file('maxstate' + '_ans'))
	exact = np.array(from_file('exact' + '_ans'))

	flavio = flavio / exact
	onehot = onehot / exact
	maxstate = maxstate / exact

	xs = [range(len(flavio))] * 3
	plot_dashed(
		xs, (flavio, onehot, maxstate), ('Permutation Characterization', 'One-Hot Encoding', 'Min-Max Normalization'),
		xlabel='episodes', ylabel='performance ratio', file='states_comp')


def plot_type3():
	ylambda = np.array(from_file('tdlambda' + '_ans'))
	ydnr = np.array(from_file('dnr' + '_ans'))
	yexact = np.array(from_file('exact' + '_ans'))

	for i in range(len(yexact)):
		if abs(yexact[i]) < EPS:
			ylambda[i] = 1
			ydnr[i] = 1
		else:
			ylambda[i] = ylambda[i] / yexact[i]
			ydnr[i] = ydnr[i] / yexact[i]

	xs = [range(len(yexact))] * 2
	plot_dashed(
		xs, (ylambda, ydnr), ('TD-Lambda', 'DDQN'),
		xlabel='episodes', ylabel='performance ratio', file='lambda_ddqn')


def plot_type4():
	divs = 30
	div_sz = 0.01

	ygreedy = np.array(from_file('greedy' + '_ans'))
	yrl = np.array(from_file('rl' + '_ans'))
	yexact = np.array(from_file('exact' + '_ans'))
	ygreedy = ygreedy / yexact
	yrl = yrl / yexact

	max_ = max(ygreedy.max(), yrl.max())
	min_ = min(ygreedy.min(), yrl.min())
	sz = (max_ - min_) / divs

	ans = {}
	n = len(yrl)
	for label, y in (('rl', yrl), ('greedy', ygreedy)):
		vals = [0] * divs
		for x in y:
			idx = min(int((x - 1) / div_sz), divs - 1)
			vals[idx] += 1

		ans[label] = {'[%.2f, %.2f%%)' % (i * div_sz * 100, (i + 1) * 100 * div_sz): vals[i] for i in range(divs)}

	print(ans)


def plot_type5():
	ygreedy = np.array(from_file('greedy' + '_ans'))
	yrl = np.array(from_file('rl' + '_ans'))
	yexact = np.array(from_file('exact' + '_ans'))
	n = len(yrl)

	faster_than_greedy = np.sum(yrl < ygreedy) / n * 100
	equal_greedy = np.sum(yrl == ygreedy) / n * 100
	slower_than_greedy = np.sum(yrl > ygreedy) / n * 100
	avg_ratio = (yrl / yexact).mean()
	print('faster: %.2f' % faster_than_greedy)
	print('equal: %.2f' % equal_greedy)
	print('slower: %.2f' % slower_than_greedy)
	print('avg: %.4f' % avg_ratio)


class OrnsteinUhlenbeckActionNoise:
	def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
		self.theta = theta
		self.mu = mu
		self.sigma = sigma
		self.dt = dt
		self.x0 = x0
		self.reset()

	def __call__(self):
		x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
			self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
		self.x_prev = x
		return x

	def reset(self):
		self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

	def __repr__(self):
		return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
