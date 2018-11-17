import random
from collections import deque
from multiprocessing.dummy import Pool as ThreadPool
import time

import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

from permutation_sorting import PermutationSorting
from state_transformers import OneHotStateTransformer
from util import v_upperbound, v_upperbound_breakpoints, reverse_subarray, AtomicInteger, plot_running_avg, plot, \
	greedy_reversal_sort, ensure_saved_models_dir

PRETRAIN_WEIGHTS_PATH = './saved_models/ddqn_pretrain_weights_15.h5'
FINAL_WEIGHTS_PATH = './saved_models/ddqn_final_weights_15.h5'


# Double DQN Agent
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DDQNAgent:
	def __init__(self, env, state_transformer, batch_size=64):
		self.render = False
		self.env = env
		self.state_size = state_transformer.dimensions
		self.action_size = self.env.action_space.n
		self.state_transformer = state_transformer

		# these is hyper parameters for the Double DQN
		self.discount_factor = 0.99
		self.learning_rate = float(1e-3)
		self.epsilon = 0.2
		self.epsilon_decay = 0.993
		self.epsilon_min = 0.1
		self.batch_size = batch_size
		self.train_start = 60000
		# create replay memory using deque
		self.memory = deque(maxlen=int(1e6))

		# create main model and target model
		self.model = self.build_model()
		self.target_model = self.build_model()

		# initialize target model
		self.update_target_model()

		print("State size =", self.state_size, " Action size =", self.action_size)

	# approximate Q function using Neural Network
	# state is input and Q Value of each action is output of network
	def build_model(self):
		model = Sequential()
		model.add(Dense(400, input_dim=self.state_size, activation='relu', kernel_initializer='he_uniform'))
		model.add(Dense(300, activation='relu', kernel_initializer='he_uniform'))
		model.add(Dense(self.action_size, activation='linear', kernel_initializer='he_uniform'))
		model.summary()
		model.compile(loss='logcosh', optimizer=Adam(lr=self.learning_rate))
		return model

	# after some time interval update the target model to be same with model
	def update_target_model(self):
		self.target_model.set_weights(self.model.get_weights())

	# get action from model using epsilon-greedy policy
	def get_action(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		else:
			q_value = self.model.predict(state)
			return np.argmax(q_value[0])

	# save sample <s,a,r,s'> to the replay memory
	def append_sample(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	# pick samples randomly from replay memory (with batch_size)
	def train_model(self):
		if len(self.memory) < self.train_start:
			return
		batch_size = min(self.batch_size, len(self.memory))
		mini_batch = random.sample(self.memory, batch_size)

		update_input = np.zeros((batch_size, self.state_size))
		update_target = np.zeros((batch_size, self.state_size))
		action, reward, done = [], [], []

		for i in range(batch_size):
			update_input[i] = mini_batch[i][0]
			action.append(mini_batch[i][1])
			reward.append(mini_batch[i][2])
			update_target[i] = mini_batch[i][3]
			done.append(mini_batch[i][4])

		target = self.model.predict(update_input)
		target_next = self.model.predict(update_target)
		target_val = self.target_model.predict(update_target)

		for i in range(self.batch_size):
			# like Q Learning, get maximum Q value at s'
			# But from target model
			if done[i]:
				target[i][action[i]] = reward[i]
			else:
				# the key point of Double DQN
				# selection of action is from model
				# update is from target model
				a = np.argmax(target_next[i])
				target[i][action[i]] = reward[i] + self.discount_factor * (
					target_val[i][a])

		# make minibatch which includes target q value and predicted q value
		# and do the model fit!
		self.model.fit(update_input, target, batch_size=self.batch_size, epochs=1, verbose=0)

	# Get the score of every possible action
	def fill_target(self, state, target):
		for idx, (_, i, j, k) in enumerate(self.env.actions):
			# state_aux = state.copy()
			# reverse_subarray(state_aux, i, j)
			# target[idx] = -1 + self.discount_factor * v_upperbound(state_aux, self.discount_factor)
			target[idx] = v_upperbound_breakpoints(state, i, j, k, self.discount_factor)

	def serial_pretrain(self, rows=5000, epochs=100):
		targets = np.empty((rows, self.action_size))
		states = np.empty((rows, self.state_size))
		start = time.time()
		for i in range(rows):
			p = self.env.observation_space.sample()
			states[i] = self.state_transformer.transform(p)
			self.fill_target(p, targets[i])
			if i % 100 == 0:
				print("-- %.1f %%" % (i / rows * 100), time.time() - start)
		self.model.fit(states, targets, batch_size=self.batch_size, epochs=epochs, verbose=1, validation_split=0.1)
		self.update_target_model()
		ensure_saved_models_dir()
		self.model.save_weights(PRETRAIN_WEIGHTS_PATH)

	def load_pretrain_weights(self, path=PRETRAIN_WEIGHTS_PATH):
		self.model.load_weights(path)
		self.update_target_model()

	def load_final_weights(self, path=FINAL_WEIGHTS_PATH):
		self.model.load_weights(path)
		self.update_target_model()

	@staticmethod
	def _is_identity(p):
		for i in range(len(p)):
			if p[i] != i:
				return False
		return True

	def run_episode(self, max_steps, forced=None, update_eps=True, update_model=True):
		done = False
		score = 0
		state = self.env.reset(forced=forced)

		if self._is_identity(state):
			return 0, 0

		state = self.state_transformer.transform(state)
		steps = 0
		while not done and steps < max_steps:
			if self.render:
				self.env.render()

			steps += 1

			# get action for the current state and go one step in environment
			action = self.get_action(state)

			next_state, reward, done, info = self.env.step(action)
			next_state = self.state_transformer.transform(next_state)

			if update_model:
				# save the sample <s, a, r, s'> to the replay memory
				self.append_sample(state, action, reward, next_state, done)
				# every time step do the training
				self.train_model()
			score += reward
			state = next_state

		if update_model:
			# every episode update the target model to be same with model
			self.update_target_model()

		if update_eps and self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay

		return score, steps

	def fill_memory(self, steps_total=None):
		print("Filling queue memory with greedy actions.")

		start = time.time()
		s, e = 0, 0
		if steps_total is None:
			steps_total = self.train_start
		while s < steps_total:
			loss, step, score = 0, 0, 0
			done = False
			p = self.env.reset()
			while self._is_identity(p):
				p = self.env.reset()
			state = self.state_transformer.transform(p)
			while not done and step < 1000:
				# get greedy action or random based on epsilon
				if np.random.rand() <= self.epsilon:
					action = random.randrange(self.action_size)
				else:
					targets = np.empty(self.action_size)
					self.fill_target(p, targets)
					action = np.argmax(targets)

				# get the next state
				p, reward, done, info = self.env.step(action)
				next_state = self.state_transformer.transform(p)
				# save the results on memory
				self.append_sample(state, action, reward, next_state, done)
				state = next_state
				score += reward
				step += 1
				s += 1
			e += 1

	def train(self, episodes=1000, max_steps=1000, plot_rewards=True):
		scores, steps = np.empty(episodes), np.empty(episodes)
		start = time.time()
		for e in range(episodes):
			score, step = self.run_episode(max_steps)
			scores[e], steps[e] = score, step
			print("Episode:", e, "  steps:", step, "  score:", score, "  epsilon:", self.epsilon, "  time:", time.time() - start)
			'''if e%100 == 0:
				ensure_saved_models_dir()
				self.model.save_weights(FINAL_WEIGHTS_PATH)
				print("Weights Saved")'''
		ensure_saved_models_dir()
		self.model.save_weights(FINAL_WEIGHTS_PATH)

		if plot_rewards:
			t_time = time.time() - start
			print("Mean score:", np.mean(scores), " Total steps:", np.sum(steps), " total time:", t_time)
			plot(scores)
			plot_running_avg(scores)
			np.save("./train_data/ddqn_" + str(self.state_size) + "_scores", scores)
			np.save("./train_data/ddqn_" + str(self.state_size) + "_time", t_time)
			np.save("./train_data/ddqn_" + str(self.state_size) + "_steps", steps)

	def solve(self, permutation, its=100, max_steps=100, exploit_greedy_trace=False, update_eps=False,
			  update_model=False):
		ans = None
		if exploit_greedy_trace:
			trace = []
			greedy_reversal_sort(permutation, trace)
		for _ in range(its):
			if exploit_greedy_trace:
				last_ans = None
				# noinspection PyUnboundLocalVariable
				for p in trace[::-1]:
					last_ans = self.run_episode(
						max_steps=max_steps, forced=p, update_model=update_model, update_eps=update_eps)
				if ans is None or last_ans > ans:
					ans = last_ans
			else:
				pans = self.run_episode(
					max_steps=max_steps, forced=permutation, update_model=update_model, update_eps=update_eps)
				if ans is None or pans > ans:
					ans = pans
		return -ans


def main():
	np.random.seed(12345678)
	n = 15
	env = PermutationSorting(n, transpositions=True)
	state_transformer = OneHotStateTransformer(n)
	agent = DDQNAgent(env, state_transformer, batch_size=126)
	#agent.serial_pretrain()
	# agent.load_pretrain_weights()
	# agent.load_final_weights()
	agent.fill_memory()
	agent.train()


if __name__ == '__main__':
	main()

#Mean score: -0.007  Total steps: 4070.0  total time: 22.056942462921143