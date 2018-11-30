import random
from collections import deque
import time
import sys
import gc
import math

import numpy as np
import tensorflow as tf

from permutation_sorting import PermutationSorting
from state_transformers import OneHotStateTransformer
from util import v_upperbound_breakpoints, plot_running_avg, plot, greedy_reversal_sort, ensure_saved_models_dir

# Double DQN Agent
# it uses Neural Network to approximate q function
# and replay memory & target q network
class DDQNAgent:
	def __init__(self, env, state_transformer, prev_params, hidden_layer_sizes=(400,300), batch_size=64, train_start=1000,
				 render=False, fill_mem=False, epsilon=0.6,
				 pretrain_path='./saved_models/ddqn_tf_pretrain_weights_100.h5',
				 train_path='./saved_models/ddqn_tf_final_weights_100.h5'):
		self.render = render
		self.env = env
		self.state_size = state_transformer.dimensions
		self.action_size = self.env.action_space.n
		self.state_transformer = state_transformer

		# these is hyper parameters for the Double DQN
		self.discount_factor = 0.99
		self.learning_rate = float(1e-3)
		self.epsilon = epsilon
		self.epsilon_decay = 0.9995
		self.epsilon_min = 0.1
		self.batch_size = batch_size
		self.train_start = train_start
		self.pretrain_path = pretrain_path
		self.train_path = train_path
		self.fill_mem = fill_mem
		# create replay memory using deque
		self.maxlen=int(2e6)
		self.memory = deque(maxlen=self.maxlen)

		self.state, self.actions, self.G, self.predict_op, self.train_op, self.cost = self.model(hidden_layer_sizes)
		self.network_params = tf.trainable_variables()[prev_params:]

		self.t_state, self.t_actions, self.t_G, self.t_predict_op, self.t_train_op, self.t_cost =\
			self.model(hidden_layer_sizes)
		self.t_network_params = tf.trainable_variables()[len(self.network_params)+prev_params:]
		self.update_target_network_params = \
			[self.t_network_params[i].assign(self.network_params[i]) for i in range(len(self.t_network_params))]

		print("State size =", self.state_size, " Action size =", self.action_size)
		
	def model (self, hidden_layer_sizes):
		# inputs and targets
		state = tf.placeholder(tf.float32, shape=(None, self.state_size), name='state')
		G = tf.placeholder(tf.float32, shape=(None, self.action_size), name='G')
		actions = tf.placeholder(tf.int32, shape=(None,), name='actions')

		Z = state
		for M in hidden_layer_sizes:
			Z = tf.layers.dense(Z, M, activation=tf.nn.relu, kernel_initializer=tf.initializers.random_uniform(-0.003, 0.003))
			Z = tf.layers.batch_normalization(Z)

		# final output layer
		predict_op = tf.layers.dense(Z, self.action_size, activation=None, kernel_initializer=tf.initializers.random_uniform(-0.003, 0.003))
		predict_op = tf.layers.batch_normalization(predict_op)

		#selected_action_values = predict_op + tf.one_hot(actions, self.action_size)
		#selected_action_values = tf.squeeze(selected_action_values)
		#cost = tf.square(G - selected_action_values)
		cost = tf.keras.losses.logcosh(G, predict_op)
		train_op = tf.train.AdamOptimizer(self.learning_rate, epsilon=1e-4).minimize(cost)

		return state, actions, G, predict_op, train_op, cost

	def set_session(self, session):
		self.session = session

	def update_target_model(self):
		self.session.run(self.update_target_network_params)

	def predict(self, states):
		return self.session.run(self.predict_op, feed_dict={self.state: states})
	
	def target_predict(self, states):
		return self.session.run(self.t_predict_op, feed_dict={self.t_state: states})

	def update(self, states, actions, targets):
		cost, _ = self.session.run([self.cost, self.train_op], feed_dict={self.state: states, self.actions: actions, self.G: targets})
		return np.mean(cost)

	# get action from model using epsilon-greedy policy
	def get_action(self, state):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		else:
			return np.argmax(self.predict(state)[0])

	# save sample <s,a,r,s'> to the replay memory
	def append_sample(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	# pick samples randomly from replay memory (with batch_size)
	def train_model(self):
		if len(self.memory) < self.train_start:
			return 0
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

		target = self.predict(update_input)
		target_next = np.argmax(self.predict(update_target), axis=1)
		target_val = self.target_predict(update_target)
		for i in range(self.batch_size):
			if done[i]:
				target[i][action[i]] = reward[i]
			else:
				a = target_next[i]
				target[i][action[i]] = reward[i] + self.discount_factor * (
					target_val[i][a])
		loss = self.update(update_input, np.array(action), target)
		return loss

	@staticmethod
	def _is_identity(p):
		for i in range(len(p)):
			if p[i] != i:
				return False
		return True

	def run_episode(self, max_steps, forced=None, update_eps=True, update_model=True):
		done = False
		score, loss, steps = 0, 0, 0
		state = self.env.reset(forced=forced)
		if self._is_identity(state):
			return 0, 0, 0
		state = self.state_transformer.transform(state)
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
				loss += self.train_model()
			score += reward
			state = next_state
		if update_model:
			self.update_target_model()
		if update_eps and self.epsilon > self.epsilon_min:
			self.epsilon *= self.epsilon_decay
		loss = loss/steps
		return score, steps, loss

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

	def train(self, episodes=1000, max_steps=800, plot_rewards=True):
		scores, steps, losses = np.zeros(episodes), np.zeros(episodes), np.zeros(episodes)
		start = time.time()
		saver = tf.train.Saver()
		break_flag = 0
		if self.fill_mem:
			self.fill_memory()
		for e in range(episodes):
			score, step, loss = self.run_episode(max_steps)
			scores[e], steps[e], losses[e] = score, step, loss
			print("Episode:", e, "  steps:", step, "  score: %.1f" % score,"  loss:", loss, "  epsilon:", self.epsilon, "  time:", time.time() - start)
			if e%100 == 0 and break_flag==e:
				self.fill_memory()
			break_flag = break_flag+1 if step == max_steps else 0
			#if break_flag > 1000 and e >= episodes/2: break
			if math.isnan(loss): break
		ensure_saved_models_dir()
		saver.save(self.session, self.train_path)

		if plot_rewards:
			t_time = time.time() - start
			print("Mean score:", np.mean(scores), " Total steps:", np.sum(steps), " total time:", t_time)
			np.save("./train_data/ddqn_tf_" + str(self.state_size) + "_scores", scores)
			np.save("./train_data/ddqn_tf_" + str(self.state_size) + "_time", t_time)
			np.save("./train_data/ddqn_tf_" + str(self.state_size) + "_steps", steps)
			plot(steps)
			plot_running_avg(steps)
			plot_running_avg(losses, title="Losses")

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
				score, steps, loss = self.run_episode(
					max_steps=max_steps, forced=permutation, update_model=update_model, update_eps=update_eps)
				if ans is None or steps > ans:
					ans = steps
		return -ans

	# Get the score of some actions
	def fill_target(self, state, target):
		for idx, (_, i, j, k) in enumerate(self.env.actions):
			target[idx] = v_upperbound_breakpoints(state, i, j, k, self.discount_factor)

	def serial_pretrain(self, rows=100000, batch_size=64, epochs=10):
		start = time.time()
		r = int(rows / batch_size)
		for j in range(r):
			targets = np.empty((batch_size, self.action_size))
			states = np.empty((batch_size, self.state_size))
			for i in range(batch_size):
				p = self.env.observation_space.sample()
				states[i] = self.state_transformer.transform(p)
				self.fill_target(p, targets[i])
				if i % 100 == 0:
					print(j, "-- %.6f %%" % ((j*batch_size + i)/rows * 100), time.time() - start)
			actions = np.argmax(targets, axis=1)
			print(j, "-- %.6f %%" % ((j * batch_size + i) / rows * 100), time.time() - start, "-- UPDATE")
			for i in range(epochs):
				self.update(states, actions, targets)
			targets, states = None, None
			gc.collect()
		ensure_saved_models_dir()
		saver = tf.train.Saver()
		saver.save(self.session, self.pretrain_path)
		self.update_target_model()

	def load_pretrain_weights(self):
		saver = tf.train.Saver()
		saver.restore(self.session, self.pretrain_path)
		self.update_target_model()

	def load_weights(self):
		saver = tf.train.Saver()
		saver.restore(self.session, self.train_path)
		self.update_target_model()



def main(argv):
	np.random.seed(12345678)
	n = int(argv[1])
	pretrain = './saved_models/ddqn_tf_pretrain_weights_' + str(n) + '.h5'
	train = './saved_models/ddqn_tf_final_weights_' + str(n) + '.h5'
	env = PermutationSorting(n, transpositions=True)
	state_transformer = OneHotStateTransformer(n)
	hidden_layer_sizes = (400, 300)
	agent = DDQNAgent(env, state_transformer, 0, hidden_layer_sizes=hidden_layer_sizes, batch_size=256, train_start=60000,
					  epsilon=0.7, fill_mem=True, render=False, pretrain_path=pretrain, train_path=train)
	with tf.Session() as sess:
		agent.set_session(sess)
		sess.run(tf.global_variables_initializer())
		#agent.serial_pretrain()
		#agent.load_pretrain_weights()
		#agent.load_weights()
		agent.train(episodes=5000)
		'''
		apt1 = 0
		for _ in range(100):
			p = np.random.permutation(n)
			rl_ans = agent.solve(p)
			ap1 = float(rl_ans)
			apt1 = (apt1 + ap1) / 2
			print(p, '-', 'RL:', rl_ans, ' Approx:', ap1)
		print(apt1)'''


if __name__ == '__main__':
	if len(sys.argv) < 2:
		sys.exit()
	main(sys.argv)
