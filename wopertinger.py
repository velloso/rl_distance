import random
import sys
from collections import deque
import time
import gc

import math
import numpy as np
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

from permutation_sorting import PermutationSorting
from state_transformers import OneHotStateTransformer
from util import plot_running_avg, plot, ensure_saved_models_dir, v_upperbound_breakpoints, breakpoints, \
	OrnsteinUhlenbeckActionNoise


# Implementation of the Wolpertinger Policy
class DDPGAgent:
	def __init__(self, env, state_transformer, layer_actor, layer_critic, actor_learning_rate=0.0001,
				 critic_learning_rate=0.001, train_start=1000, maxlen=1e5, batch_size=64,
				 discount_factor=0.99, tau=0.001, neighbors_percent=0.01, render=False,
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
		self.actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(self.actor_output))

		self.state_size = state_transformer.dimensions
		self.discount_factor = discount_factor
		self.batch_size = batch_size
		self.train_start = train_start
		self.pretrain_path = pretrain_path
		self.train_path = train_path

		# create replay memory using deque
		self.maxlen = int(maxlen)
		self.memory = deque(maxlen=self.maxlen)
		self.render = render

		# Inputs
		self.state = tf.placeholder(tf.float32, shape=(None, self.state_size), name='state')
		self.state_next = tf.placeholder(tf.float32, shape=(None, self.state_size), name='state_next')
		self.actions = tf.placeholder(tf.float32, shape=(None, self.actor_output), name='actions')
		self.q_val = tf.placeholder(tf.float32, [None, 1], name='q_val')

		# Main networks
		with tf.variable_scope('main/actor'):
			self.actor_op = self.build_model(self.state, layer_actor+[self.actor_output],
											 output_activation=tf.nn.tanh)
		with tf.variable_scope('main/critic'):
			self.critic_op = self.build_model(tf.concat([self.state, self.actions], axis=-1), layer_critic+[1])
		with tf.variable_scope('main/critic', reuse=True):
			self.critic_actor_op = self.build_model(tf.concat([self.state, self.actor_op], axis=-1),
													layer_critic+[1])

		# Target networks
		with tf.variable_scope('target/actor'):
			self.t_actor_op = self.build_model(self.state_next, layer_actor+[self.actor_output],
											   output_activation=tf.nn.tanh)
		with tf.variable_scope('target/critic'):
			self.t_critic_op = self.build_model(tf.concat([self.state_next, self.actions], axis=-1), layer_critic+[1])
		with tf.variable_scope('target/critic', reuse=True):
			self.t_critic_actor_op = self.build_model(tf.concat([self.state, self.t_actor_op], axis=-1),
													  layer_critic+[1])

		# DDPG losses
		self.actor_loss = -tf.reduce_mean(self.critic_actor_op)
		self.critic_loss = tf.reduce_mean((self.critic_op - self.q_val) ** 2)
		#self.critic_loss = tf.losses.mean_squared_error(self.actor_op, self.predicted_q_value)


		# Train functions
		self.train_actor = tf.train.AdamOptimizer(actor_learning_rate).\
			minimize(self.actor_loss, var_list=self.get_vars('main/actor'))
		self.train_critic = tf.train.AdamOptimizer(critic_learning_rate).\
			minimize(self.critic_loss, var_list=self.get_vars('main/critic'))

		# Target update
		self.target_update = tf.group([tf.assign(v_targ, tau * v_targ + (1 - tau) * v_main)
								  for v_main, v_targ in zip(self.get_vars('main'), self.get_vars('target'))])

		# Initializing targets to match main variables
		self.target_init = tf.group([tf.assign(v_targ, v_main)
								for v_main, v_targ in zip(self.get_vars('main'), self.get_vars('target'))])


		print("Action size =", self.action_size, " N Neighbours =", self.n_neighbors, " State size =", self.state_size)

	# Builds the network
	def build_model(self, input_layer, layer_sizes, activation=tf.nn.relu, output_activation=None):
		Z = input_layer
		for M in layer_sizes[:-1]:
			Z = tf.layers.dense(Z, M, activation=activation, kernel_initializer=tf.keras.initializers.he_uniform())
		predict_op = tf.layers.dense(Z, layer_sizes[-1], activation=output_activation,
									 kernel_initializer=tf.initializers.random_uniform(-0.003, 0.003))
		return predict_op

	@staticmethod
	def get_vars(scope):
		return [x for x in tf.global_variables() if scope in x.name]

	def set_session(self, session):
		self.session = session

	# save sample <s,a,r,s'> to the replay memory
	def append_sample(self, state, action, reward, next_state, done):
		self.memory.append((state, action, reward, next_state, done))

	@staticmethod
	def _is_identity(p):
		for i in range(len(p)):
			if p[i] != i:
				return False
		return True

	# Predict action based on state
	def actor_get_action(self, state, target=False):
		if target:
			return self.session.run(self.t_actor_op, feed_dict={
				self.state_next: state
			})
		else:
			return self.session.run(self.actor_op, feed_dict={
				self.state: state
			})

	# Updates the actor weights
	def actor_train_model(self, state):
		return self.session.run([self.actor_loss, self.train_actor], feed_dict={
			self.state: state
		})

	# Get Q value of actions
	def critic_get_q_value(self, state, action, target=False):
		if target:
			return self.session.run(self.t_critic_op, feed_dict={
				self.state_next: state,
				self.actions: action
			})
		else:
			return self.session.run(self.critic_op, feed_dict={
				self.state: state,
				self.actions: action
			})

	# Updates the critic weights
	def critic_train_model(self, state, action, q_value):
		return self.session.run([self.critic_loss, self.critic_op, self.train_critic], feed_dict={
			self.state: state,
			self.actions: action,
			self.q_val: q_value
		})

	# Update the target model with the the function
	# target_weights = tau * model_weights + (1 - tau)*target_weights
	def update_target_model(self):
		self.session.run(self.target_update)

	# Predict action based on state, in target model
	def get_q_target_aux(self, state, actions):
		result = self.critic_get_q_value(
			np.repeat(state, self.n_neighbors, axis=0), actions.reshape(-1, self.actor_output), target=True)
		result = result.reshape(self.batch_size, self.n_neighbors, -1)

		return np.amax(result, axis=1)

	# Predict action based on state
	def get_q_value_aux(self, state, action):
		return self.critic_get_q_value(np.repeat(state, self.n_neighbors, axis=0), action[0])

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
		apx_action = self.actor_get_action(state) + self.actor_noise()

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
		apx_action = self.actor_get_action(state, target=True) + self.actor_noise()

		# Then gets the k nearest action of the approximated values
		act_neighbors_idx = self.neighbors.kneighbors(apx_action, return_distance=False)
		near_actions = self.get_near_actions(act_neighbors_idx, batch=True)

		# Then get the best action from the Critic network
		best_actions = self.get_q_target_aux(state, near_actions)
		# action_idx = act_neighbors_idx[0][best_actions_idx]
		return best_actions

	# Train the model using values from the experience replay memory
	def train_model(self):
		if len(self.memory) < self.train_start:
			return 0
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
		loss = self.critic_train_model(state, action, np.array(yi))

		# Update Actor network
		self.actor_train_model(state)

		# Updates target networks
		self.update_target_model()

		return loss[0]

	# Run one episode
	def run_episode(self, max_steps, forced=None, update_model=True):
		if self.render:
			self.env.render()
		done = False
		score, steps, loss = 0, 0, 0
		state = self.env.reset(forced=forced)

		if self._is_identity(state):
			return 0, 0, 0

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
				loss += self.train_model()

			score += reward
			state = next_state

		loss = loss / steps
		return score, steps, loss

	# Train the agent
	def train(self, episodes=1000, max_steps=1000, plot_rewards=True):
		# Initialize target network weights
		scores, steps = np.empty(episodes), np.empty(episodes)
		start = time.time()
		break_flag = 0
		for e in range(episodes):
			score, step, loss = self.run_episode(max_steps)
			scores[e], steps[e] = score, step
			print("Episode:", e, "  steps:", step, "  score:", score, "  loss:", loss, "  time:", time.time() - start)
			#break_flag = break_flag+1 if step == max_steps else 0
			#if break_flag > 60: break
		saver = tf.train.Saver()
		saver.save(self.session, self.train_path)

		if plot_rewards:
			t_time = time.time() - start
			print("Mean step:", np.mean(steps), " Total steps:", np.sum(steps), " total time:", t_time)
			np.save("./train_data/ddpg_enc_actions" + str(self.state_size) + '_' + str(self.n_neighbors) + "_scores", scores)
			np.save("./train_data/ddpg_enc_actions" + str(self.state_size) + '_' + str(self.n_neighbors) + "_time", t_time)
			np.save("./train_data/ddpg_enc_actions" + str(self.state_size) + '_' + str(self.n_neighbors) + "_steps", steps)
			plot(steps)
			plot_running_avg(steps)

	# Solve a permutation
	def solve(self, permutation, its=100, max_steps=100, update_model=False):
		ans = None
		for _ in range(its):
			pans, _, _ = self.run_episode(max_steps=max_steps, forced=permutation, update_model=update_model)
			if ans is None or pans > ans:
				ans = pans
		return -ans

	# Fill the targets for the memory fill function
	def fill_target(self, state, target):
		for idx, (_, i, j, k) in enumerate(self.env.actions):
			target[idx] = v_upperbound_breakpoints(state, i, j, k)

	# Adds good data to the replay memory
	def fill_memory(self, steps_total=None, max_steps=800, epsilon=0.7):
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
			while not done and step < max_steps:
				# get greedy action or random based on epsilon
				if np.random.rand() <= epsilon:
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

	# Pre trains the agent
	def serial_pretrain(self, rows=16000000, batch_size=32, act_size=10, epochs=3):
		start = time.time()
		r = int(rows / batch_size)
		for j in range(r):
			states = np.empty((batch_size, self.state_size))
			actions = np.random.randint(self.action_size, size=(batch_size, act_size))
			targets = np.empty((batch_size, act_size))
			for i in range(batch_size):
				p = self.env.observation_space.sample()
				states[i] = self.state_transformer.transform(p)
				for k in range(act_size):
					_, ia, ja, ka = self.env.actions[actions[i][k]]
					targets[i][k] = v_upperbound_breakpoints(p, ia, ja, ka)
				if i % 100 == 0:
					print(j, "-- %.6f %%" % ((j*batch_size + i)/rows * 100), time.time() - start)
			print(j, "-- %.6f %%" % ((j * batch_size + i) / rows * 100), time.time() - start, "-- UPDATE")
			for i in range(epochs):
				self.critic_train_model(np.repeat(states, act_size, axis=0),
										self.enc_actions[actions.reshape(-1,)].todense(),
										targets.reshape(-1,1))
				self.actor_train_model(states)
			targets, states, actions = None, None, None
			gc.collect()
			if j % 100000 == 0:
				ensure_saved_models_dir()
				saver = tf.train.Saver()
				saver.save(self.session, self.pretrain_path)
				print("Pretrain weights saved")
		ensure_saved_models_dir()
		saver = tf.train.Saver()
		saver.save(self.session, self.pretrain_path)
		print("Pretrain weights saved")
		self.session.run(self.target_init)

	def load_pretrain_weights(self):
		saver = tf.train.Saver()
		saver.restore(self.session, self.pretrain_path)
		self.session.run(self.target_init)

	def load_weights(self):
		saver = tf.train.Saver()
		saver.restore(self.session, self.train_path)
		self.session.run(self.target_init)

# Trains the agent
def main(argv):
	np.random.seed(12345678)
	n = int(argv[1])
	n_neighbors = 0.8
	pretrain = './saved_models/ddpg_tf_pretrain_weights_' + str(n) + '.h5'
	train = './saved_models/ddpg_tf_final_weights_' + str(n) + '_' + str(n_neighbors) + '.h5'
	env = PermutationSorting(n, transpositions=True)
	state_transformer = OneHotStateTransformer(n)
	actor_layer_sizes = [400, 300]
	critic_layer_sizes = [400, 300]
	agent = DDPGAgent(env, state_transformer, actor_layer_sizes, critic_layer_sizes, batch_size=32,
					  actor_learning_rate=0.0001, critic_learning_rate=0.001,
					  train_start=1000, maxlen=2000, neighbors_percent=n_neighbors, render=False,
					  pretrain_path=pretrain, train_path=train)
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	with tf.Session(config=config) as sess:
		agent.set_session(sess)
		sess.run(tf.global_variables_initializer())
		sess.run(agent.target_init)
		#agent.serial_pretrain()
		agent.load_pretrain_weights()
		agent.fill_memory(steps_total=1000, max_steps=300, epsilon=0.3)
		agent.train(episodes=10000, max_steps=800)

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("Missing Params, input the permutation size")
		sys.exit()
	main(sys.argv)


# Parameters used for n = 15
# n_neighbors = 0.8
# actor_layer_sizes = [400, 300]
# critic_layer_sizes = [400, 300]
# agent = DDPGAgent(env, state_transformer, actor_layer_sizes, critic_layer_sizes, batch_size=32,
#					  actor_learning_rate=0.0001, critic_learning_rate=0.001,
#					  train_start=16000, maxlen=20000, neighbors_percent=n_neighbors, render=False,
#					  pretrain_path=pretrain, train_path=train)
# agent.serial_pretrain()
# agent.fill_memory(steps_total=16000, max_steps=300, epsilon=0.2)
# agent.train(episodes=10000, max_steps=800)
