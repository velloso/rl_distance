import random
import sys
from collections import deque
import time

import math
import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

from permutation_sorting import PermutationSorting
from state_transformers import OneHotStateTransformer
from util import plot_running_avg, plot, ensure_saved_models_dir, v_upperbound_breakpoints, \
	OrnsteinUhlenbeckActionNoise


class Actor:
	def __init__(self, state_size, actor_output, hidden_layer_sizes, prev_params=0,
				 discount_factor=0.99,
				 learning_rate=0.0001, batch_size=32, train_start=60000,
				 tau=0.001):
		self.actor_output = actor_output
		self.state_size = state_size

		# these is hyper parameters for the Actor Network
		self.discount_factor = discount_factor
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.train_start = train_start
		self.tau = tau

		# create main model and target model
		self.state, self.predict_op = self.build_model(hidden_layer_sizes)
		self.network_params = tf.trainable_variables()[prev_params:]

		self.t_state, self.t_predict_op = self.build_model(hidden_layer_sizes)
		self.t_network_params = tf.trainable_variables()[len(self.network_params) + prev_params:]

		# Op for periodically updating target network with online network
		# weights
		self.update_target_network_params = \
			[self.t_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
											 tf.multiply(self.t_network_params[i], 1. - self.tau))
			 for i in range(len(self.t_network_params))]

		# This gradient will be provided by the critic network
		self.action_gradient = tf.placeholder(tf.float32, [None, self.actor_output])

		# Combine the gradients here
		self.actor_gradients = tf.gradients(self.predict_op, self.network_params, -self.action_gradient)
		# Optimization Op
		self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
			apply_gradients(zip(self.actor_gradients, self.network_params))

		self.num_trainable_vars = len(self.network_params) + len(self.t_network_params)

	# Builds the network
	def build_model(self, hidden_layer_sizes):
		state = tf.placeholder(tf.float32, shape=(None, self.state_size), name='state_actor')

		Z = state
		for M in hidden_layer_sizes:
			Z = tf.layers.dense(Z, M, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_uniform())
		predict_op = tf.layers.dense(Z, self.actor_output, activation=tf.nn.tanh,
									 kernel_initializer=tf.initializers.random_uniform(-0.003, 0.003))
		return state, predict_op

	def set_session(self, session):
		self.session = session

	# Update the target model with the the function
	# target_weights = tau * model_weights + (1 - tau)*target_weights
	def update_target_model(self, copy=False):
		self.session.run(self.update_target_network_params)

	# Predict action based on state
	def get_action(self, state):
		return self.session.run(self.predict_op, feed_dict={
			self.state: state
		})

	# Predict action based on state, in target model
	def get_action_target(self, state):
		return self.session.run(self.t_predict_op, feed_dict={
			self.t_state: state
		})

	# Updates the weights of the main model
	def train_model(self, state, action_grad):
		self.session.run(self.optimize, feed_dict={
			self.state: state,
			self.action_gradient: action_grad
		})


class Critic:
	def __init__(self, state_size, n_neighbors, actor_output, action_size, hidden_layer_sizes, prev_params,
				 discount_factor=0.99, learning_rate=0.01, batch_size=32, train_start=1000, tau=0.001):
		self.n_neighbors = n_neighbors
		self.actor_output = actor_output
		self.state_size = state_size
		self.action_size = action_size

		# these is hyper parameters for the Critic Network
		self.discount_factor = discount_factor
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.train_start = train_start
		self.tau = tau

		# create main model and target model
		self.state, self.action, self.out = self.build_model(hidden_layer_sizes)
		self.network_params = tf.trainable_variables()[prev_params:]

		self.t_state, self.t_action, self.t_out = self.build_model(hidden_layer_sizes)
		self.t_network_params = tf.trainable_variables()[len(self.network_params) + prev_params:]

		self.update_target_network_params = \
			[self.t_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
											 + tf.multiply(self.t_network_params[i], 1. - self.tau))
			 for i in range(len(self.t_network_params))]

		# Network target (y_i)
		self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

		# Define loss and optimization Op
		self.loss = tf.losses.mean_squared_error(self.out, self.predicted_q_value)
		self.optimize = tf.train.AdamOptimizer(
			self.learning_rate).minimize(self.loss)

		# Get the gradient of the net w.r.t. the action.
		# For each action in the minibatch (i.e., for each x in xs),
		# this will sum up the gradients of each critic output in the minibatch
		# w.r.t. that action. Each output is independent of all
		# actions except for one.
		self.action_grads = tf.gradients(self.out, self.action)

	def build_model(self, layer_size):
		# inputs and targets
		state = tf.placeholder(tf.float32, shape=(None, self.state_size), name='state_critic')
		actions = tf.placeholder(tf.float32, shape=(None, self.actor_output), name='actions_critic')

		net = tf.concat([state, actions], axis=-1)
		for M in layer_size:
			net = tf.layers.dense(net, M, activation=tf.nn.relu, kernel_initializer=tf.keras.initializers.he_uniform())
		out = tf.layers.dense(net, 1, activation=None,
							  kernel_initializer=tf.initializers.random_uniform(-0.0003, 0.0003))
		return state, actions, out

	def set_session(self, session):
		self.session = session

	# Update the target model with the the function
	# target_weights = tau * model_weights + (1 - tau)*target_weights
	def update_target_model(self, copy=False):
		self.session.run(self.update_target_network_params)

	# Predict action based on state
	def get_q_value(self, state, action):
		return self.session.run(self.out, feed_dict={
			self.state: state,
			self.action: action
		})

	# Predict action based on state, in target model
	def get_q_target(self, state, action):
		return self.session.run(self.t_out, feed_dict={
			self.t_state: state,
			self.t_action: action
		})

	def action_gradients(self, state, action):
		return self.session.run(self.action_grads, feed_dict={
			self.state: state,
			self.action: action
		})

	# Updates the weights of the main model
	def train_model(self, state, action, q_value):
		return self.session.run([self.out, self.optimize], feed_dict={
			self.state: state,
			self.action: action,
			self.predicted_q_value: q_value
		})


class DDPGAgent:
	def __init__(self, env, state_transformer, layer_actor, layer_critic, actor_learning_rate=0.0001,
				 critic_learning_rate=0.001, train_start=1000, maxlen=1e5, fill_mem=False,
				 batch_size=64, discount_factor=0.99, tau=0.001, neighbors_percent=0.01, render=False,
				 pretrain_path='./saved_models/ddpg_tf_pretrain_weights_100.h5',
				 train_path='./saved_models/ddpg_tf_pretrain_weights_100.h5'):
		self.env = env
		self.state_transformer = state_transformer
		self.action_size = self.env.action_space.n
		self.enc = OneHotEncoder(handle_unknown='ignore')
		self.enc_actions = self.enc.fit_transform(self.env.actions)
		self.actor_output = self.enc_actions.shape[1]
		self.n_neighbors = math.ceil(neighbors_percent*self.action_size)
		self.neighbors = NearestNeighbors(n_neighbors=self.n_neighbors)
		self.neighbors.fit(self.enc_actions)
		#self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.action_size))

		self.state_size = state_transformer.dimensions
		self.discount_factor = discount_factor
		self.batch_size = batch_size
		self.train_start = train_start
		self.pretrain_path = pretrain_path
		self.train_path = train_path
		self.fill_mem = fill_mem

		# create replay memory using deque
		self.maxlen = int(maxlen)
		self.memory = deque(maxlen=self.maxlen)
		self.render = render

		self.actor = Actor(self.state_size, self.actor_output, layer_actor,
						   learning_rate=actor_learning_rate, batch_size=batch_size, tau=tau)

		self.critic = Critic(self.state_size, self.n_neighbors, self.actor_output, self.action_size,
							 layer_critic, prev_params=self.actor.num_trainable_vars,
							 learning_rate=critic_learning_rate, batch_size=batch_size, tau=tau)

		print("Action size =", self.action_size, " N Neighbours =", self.n_neighbors, " State size =", self.state_size)

	def set_session(self, session):
		self.session = session
		self.actor.set_session(self.session)
		self.critic.set_session(self.session)

	# save sample <s,a,r,s'> to the replay memory
	def append_sample(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	@staticmethod
	def _is_identity(p):
		for i in range(len(p)):
			if p[i] != i:
				return False
		return True

	# Predict action based on state, in target model
	def get_q_target_aux(self, state, actions):
		result = self.critic.get_q_target(np.repeat(state, self.n_neighbors, axis=0), actions.reshape(-1, self.actor_output))
		result = result.reshape(self.batch_size, self.n_neighbors, -1)
		return np.amax(result, axis=1)

	# Predict action based on state
	def get_q_value_aux(self, state, action):
		return self.critic.get_q_value(np.repeat(state, self.n_neighbors, axis=0), action[0])

	# Get the values of the near actions with idx = idx_list
	def get_near_actions(self, idx_list, batch=True):
		if batch:
			n_actions = np.empty((self.batch_size, self.n_neighbors, self.actor_output))
		else:
			n_actions = np.empty((1, self.n_neighbors, self.actor_output))
		for i, idx in enumerate(idx_list):
			n_actions[i] = (self.enc_actions[idx].todense())
		return n_actions

	# Applies the Wolpertinger Policy
	def get_action(self, state):
		# First gets an approximation action from the Actor network
		apx_action = self.actor.get_action(state)# + self.actor_noise()

		# Then gets the k nearest action of the approximated values
		act_neighbors_idx = self.neighbors.kneighbors(apx_action, return_distance=False)
		near_actions = self.get_near_actions(act_neighbors_idx, batch=False)

		# Then get the best action from the Critic network
		best_actions = self.get_q_value_aux(state, near_actions)
		best_actions_idx = np.argmax(best_actions)
		action_idx = act_neighbors_idx[0][best_actions_idx]
		return action_idx

	# Applies the Wolpertinger Policy to the target network
	def get_target_action(self, state):
		# First gets an approximation action from the Actor network
		apx_action = self.actor.get_action_target(state)# + self.actor_noise()

		# Then gets the k nearest action of the approximated values
		act_neighbors_idx = self.neighbors.kneighbors(apx_action, return_distance=False)
		near_actions = self.get_near_actions(act_neighbors_idx, batch=True)

		# Then get the best action from the Critic network
		best_actions = self.get_q_target_aux(state, near_actions)
		# action_idx = act_neighbors_idx[0][best_actions_idx]
		return best_actions

	def train_model(self):
		if len(self.memory) < self.train_start:
			return
		# Sample a random minibatch from the replay buffer
		batch_size = min(self.batch_size, len(self.memory))
		mini_batch = random.sample(self.memory, batch_size)

		state = np.zeros((batch_size, self.state_size))
		next_state = np.zeros((batch_size, self.state_size))
		action = np.zeros((batch_size, self.actor_output))
		reward, done = [], []
		# Get the values from the minibatch (st, at, rt, st+1, done_flag)
		for i in range(batch_size):
			state[i] = mini_batch[i][0]
			action[i] = mini_batch[i][1]
			reward.append(mini_batch[i][2])
			next_state[i] = mini_batch[i][3]
			done.append(mini_batch[i][4])

		# Update Critic network
		target_action = self.get_target_action(next_state)
		yi = np.zeros((self.batch_size, 1))
		for i in range(self.batch_size):
			if done[i]:
				yi[i] = reward[i]
			else:
				yi[i] = reward[i] + self.discount_factor * target_action[i]
		self.critic.train_model(state, action, np.array(yi))

		# Update Actor network
		action_out = self.actor.get_action(state)
		act_grads = self.critic.action_gradients(state, action_out)
		self.actor.train_model(state, act_grads[0])

		# Updates target networks
		self.actor.update_target_model()
		self.critic.update_target_model()

	def run_episode(self, max_steps, forced=None, update_model=True):
		if self.render:
			self.env.render()
		done = False
		score = 0
		steps = 0
		state = self.env.reset(forced=forced)

		if self._is_identity(state):
			return 0, steps

		state = self.state_transformer.transform(state)
		rem_steps = max_steps
		while not done and rem_steps > 0:
			if self.render:
				self.env.render()

			rem_steps -= 1
			steps += 1

			# get action for the current state and go one step in environment
			action_idx = self.get_action(state)
			next_state, reward, done, info = self.env.step(action_idx)
			next_state = self.state_transformer.transform(next_state)

			if update_model:
				# save the sample <s, a, r, s'> to the replay memory
				self.append_sample(state, self.enc_actions[action_idx].todense(), reward, next_state, done)
				# every time step do the training
				self.train_model()

			score += reward
			state = next_state

		return score, steps

	def train(self, episodes=1000, max_steps=800, plot_rewards=True):
		# Initialize target network weights
		self.actor.update_target_model(copy=True)
		self.critic.update_target_model(copy=True)
		scores, steps = np.empty(episodes), np.empty(episodes)
		start = time.time()
		break_flag = 0
		for e in range(episodes):
			score, step = self.run_episode(max_steps)
			scores[e], steps[e] = score, step
			print("Episode:", e, "  steps:", step, "  score:", score, "  time:", time.time() - start)
			if e%50 == 0 and step==max_steps and self.fill_mem:
				self.fill_memory()
			break_flag = break_flag+1 if step == max_steps else 0
			if break_flag > 50 and e >= episodes/2: break
		ensure_saved_models_dir()
		saver = tf.train.Saver()
		saver.save(self.session, self.train_path)

		if plot_rewards:
			t_time = time.time() - start
			print("Mean score:", np.mean(scores), " Total steps:", np.sum(steps), " total time:", t_time)
			np.save("./train_data/ddpg_enc_actions" + str(self.state_size) + str(self.n_neighbors) + "_scores", scores)
			np.save("./train_data/ddpg_enc_actions" + str(self.state_size) + str(self.n_neighbors) + "_time", t_time)
			np.save("./train_data/ddpg_enc_actions" + str(self.state_size) + str(self.n_neighbors) + "_steps", steps)
			plot(steps)
			plot_running_avg(steps)

	def solve(self, permutation, its=100, max_steps=100, update_model=False):
		ans = None
		for _ in range(its):
			pans = self.run_episode(max_steps=max_steps, forced=permutation, update_model=update_model)
			if ans is None or pans > ans:
				ans = pans
		return -ans

	def fill_target(self, state, target):
		for idx, (_, i, j, k) in enumerate(self.env.actions):
			target[idx] = v_upperbound_breakpoints(state, i, j, k, self.discount_factor)

	def fill_memory(self, steps_total=None):
		print("Filling queue memory with greedy actions.")
		s, e = 0, 0
		if steps_total is None:
			steps_total = self.train_start
		while s < steps_total:
			step = 0
			done = False
			p = self.env.reset()
			while self._is_identity(p):
				p = self.env.reset()
			state = self.state_transformer.transform(p)
			while not done and step < 1000:
				# get greedy action or random based on epsilon
				if np.random.rand() <= 0.7:
					action = random.randrange(self.action_size)
				else:
					targets = np.empty(self.action_size)
					self.fill_target(p, targets)
					action = np.argmax(targets)

				# get the next state
				p, reward, done, info = self.env.step(action)
				next_state = self.state_transformer.transform(p)
				# save the results on memory
				self.append_sample(state, self.enc_actions[action].todense(), reward, next_state, done)
				state = next_state
				step += 1
				s += 1
			e += 1

def main(argv):
	np.random.seed(12345678)
	n = int(argv[1])
	n_neighbors = 0.1
	pretrain = './saved_models/ddpg_tf_pretrain_weights_' + str(n) + str(n_neighbors) + '.h5'
	train = './saved_models/ddpg_tf_final_weights_' + str(n) + str(n_neighbors) + '.h5'
	env = PermutationSorting(n, transpositions=True)
	state_transformer = OneHotStateTransformer(n)
	actor_layer_sizes = [400, 300]
	critic_layer_sizes = [400, 300]
	agent = DDPGAgent(env, state_transformer, actor_layer_sizes, critic_layer_sizes, batch_size=32,
					  train_start=60000, maxlen=1e6, neighbors_percent=n_neighbors, render=False,
					  fill_mem=True, pretrain_path=pretrain, train_path=train)
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		agent.set_session(sess)
		sess.run(tf.global_variables_initializer())
		agent.train(episodes=1000)

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("Missing Params")
		sys.exit()
	main(sys.argv)
