import gym
import numpy as np
from gym import spaces

from util import reverse_subarray, count_breakpoints_reduced, count_breakpoints_reduced_transposition, breakpoints

REV_OP = 'rev'
TRANS_OP = 'trans'


class PermutationSpace(gym.Space):
	def contains(self, x):
		raise Exception('Env does not support contains call')

	def __init__(self, n):
		self.n = n
		gym.Space.__init__(self, (n,), np.int)

	def sample(self):
		return np.random.permutation(self.n)


class PermutationSorting(gym.Env):
	def __init__(
			self, n, reversals=True, transpositions=False):

		actions = []

		if reversals:
			for i in range(n - 1):
				for j in range(i + 1, n):
					actions.append([REV_OP, i, j, -1])

		if transpositions:
			for i in range(n - 1):
				for k in range(i + 1, n):
					for j in range(i, k):
						actions.append([TRANS_OP, i, j, k])

		self._identity = np.arange(n)
		self.actions = actions
		self.observation_space = PermutationSpace(n)
		self.action_space = spaces.Discrete(len(actions))
		self.state_space = n * n
		self._state = np.random.permutation(n)
		self._render = False
		self._n = n
		self._breakpoints = 0
		self._reversals = reversals
		self._transpositions = transpositions

	def reset(self, forced=None):
		self._state = np.array(forced) if forced is not None else np.random.permutation(self._n)

		self._breakpoints = breakpoints(self._state)
		if self._render:
			print(self._state, self._breakpoints)
			#self._render = False

		return np.array(self._state)

	def render(self, mode='human'):
		self._render = True

	def step(self, action):
		type_, i, j, k = self.actions[action]
		state = self._state

		if type_ == REV_OP:
			reward = count_breakpoints_reduced(state, i, j)
			self._breakpoints -= reward
			reverse_subarray(state, i, j)
		else:
			reward = count_breakpoints_reduced_transposition(state, i, j, k)
			self._breakpoints -= reward
			state = np.concatenate((state[:i], state[j + 1: k + 1], state[i:j + 1], state[k + 1:]))
		# The reward for a step is the number of breakpoints removed, minus 1.
		reward = -1
		is_final = self._breakpoints == 0
		done = is_final

		if self._render:
			print(self._state,"action-idx:", action, type_, "i:", i, " j:", j, " k:", k, "===>", state, "  breakpoints:", self._breakpoints)
			#self._render = False

		self._state = state
		return np.array(state), reward, done, {}
