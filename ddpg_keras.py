import random
import sys
from collections import deque
import time

import math
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, Add, BatchNormalization, Activation
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.initializers import RandomUniform
import keras.backend as keras_backend
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder

from permutation_sorting import PermutationSorting
from state_transformers import OneHotStateTransformer
from util import plot_running_avg, plot, ensure_saved_models_dir, v_upperbound_breakpoints

PRETRAIN_WEIGHTS_PATH = './saved_models/ddpg_pretrain_weights_test_100.h5'
FINAL_WEIGHTS_PATH = './saved_models/ddpg_final_weights_test_100.h5'


class Actor:
	def __init__(self, session, state_size, actor_output, discount_factor=0.99,
					learning_rate=0.0001, batch_size=32, train_start=60000,
					tau=0.001):

		self.session = session
		self.actor_output = actor_output
		self.state_size = state_size

		# these is hyper parameters for the Actor Network
		self.discount_factor = discount_factor
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.train_start = train_start
		self.tau = tau
		# create replay memory using deque
		self.maxlen = int(1e6)
		self.memory = deque(maxlen=self.maxlen)

		# Let tensorflow and keras work together
		keras_backend.set_session(session)

		# create main model and target model
		self.model = self.build_model()
		self.target_model = self.build_model()
		
		model_weights = self.model.get_weights()
		target_weights = self.target_model.get_weights()
		self.new_target_weights = [self.tau*model_weights_i + (1 - self.tau)*target_weights_i
							  for model_weights_i, target_weights_i in zip(model_weights, target_weights)]

		self.gradients = keras_backend.gradients(self.model.output, target_weights)

		# initialize target model
		self.update_target_model()

	#
	#
	def build_model(self):
		model = Sequential()
		model.add(Dense(400, input_dim=self.state_size, kernel_initializer='he_uniform'))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		model.add(Dense(300, kernel_initializer='he_uniform'))
		model.add(BatchNormalization())
		model.add(Activation('relu'))
		w_init = RandomUniform(minval=-0.003, maxval=0.003)
		model.add(Dense(self.actor_output, activation='tanh', kernel_initializer=w_init))
		#model.summary()
		model.compile(loss='logcosh', optimizer=Adam(lr=self.learning_rate, amsgrad=True))
		return model

	# Update the target model with the the function
	# target_weights = tau * model_weights + (1 - tau)*target_weights
	def update_target_model(self, copy=False):
		if copy:
			return self.target_model.set_weights(self.model.get_weights())
		self.target_model.set_weights(self.new_target_weights)

	# Predict action based on state
	def get_action(self, state):
		return self.model.predict(state)

	# Predict action based on state, in target model
	def get_action_target(self, state):
		return self.target_model.predict(state)

	# Updates the weights of the main model
	def train_model(self, state, action_grad):
		self.model.fit(state, action_grad, batch_size=self.batch_size, epochs=1, verbose=0)

	def load_pretrain_weights(self, path=PRETRAIN_WEIGHTS_PATH):
		self.model.load_weights(path)
		self.update_target_model()

	def load_final_weights(self, path=FINAL_WEIGHTS_PATH):
		self.model.load_weights(path)
		self.update_target_model()


class Critic:
	def __init__(self, session, state_size, n_neighbors, actor_output, action_size, discount_factor=0.99,
					learning_rate=0.01, batch_size=32, train_start=1000,
					tau=0.001):

		self.session = session
		self.n_neighbors = n_neighbors
		self.actor_output = actor_output
		self.action_size = action_size
		self.state_size = state_size

		# these is hyper parameters for the Actor Network
		self.discount_factor = discount_factor
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.train_start = train_start
		self.tau = tau
		# create replay memory using deque
		self.memory = deque(maxlen=2000)

		# Let tensorflow and keras work together
		keras_backend.set_session(session)

		# create main model and target model
		self.model, self.state_input, self.action_input = self.build_model()
		self.target_model, self.target_state_input, self.target_action_input = self.build_model()
		
		model_weights = self.model.get_weights()
		target_weights = self.target_model.get_weights()
		self.new_target_weights = [self.tau*model_weights_i + (1 - self.tau)*target_weights_i
							  for model_weights_i, target_weights_i in zip(model_weights, target_weights)]

		# initialize target model
		self.update_target_model()

		# Get the gradient of the net w.r.t. the action.
		# For each action in the minibatch (i.e., for each x in xs),
		# this will sum up the gradients of each critic output in the minibatch
		# w.r.t. that action. Each output is independent of all
		# actions except for one.
		self.action_grads = tf.gradients(self.model.output, self.action_input)

	#
	#
	def build_model(self):
		state_input_layer = Input(shape=[self.state_size])
		action_input_layer = Input(shape=[self.actor_output])
		state_layer = Dense(400, kernel_initializer='he_uniform')(state_input_layer)
		state_layer = BatchNormalization()(state_layer)
		state_layer = Activation('relu')(state_layer)
		action_layer = Dense(300, kernel_initializer='he_uniform')(action_input_layer)
		hidden = Dense(300, kernel_initializer='he_uniform')(state_layer)
		hidden = Add()([hidden, action_layer])
		hidden = Activation('relu')(hidden)
		w_init = RandomUniform(minval=-0.0003, maxval=0.0003)
		output_layer = Dense(1, activation='linear', kernel_initializer=w_init)(hidden)
		model = Model(inputs=[state_input_layer, action_input_layer], outputs=output_layer)
		#model.summary()
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
		return model, state_input_layer, action_input_layer


	# Update the target model with the the function
	# target_weights = tau * model_weights + (1 - tau)*target_weights
	def update_target_model(self, copy=False):
		if copy:
			return self.target_model.set_weights(self.model.get_weights())
		self.target_model.set_weights(self.new_target_weights)

	# Predict action based on state
	def get_q_value(self, state, action):
		return self.model.predict([state, action])

	# Predict action based on state, in target model
	def get_q_target(self, state, action):
		return self.target_model.predict([state, action])

	def action_gradients(self, inputs, actions):
		return self.session.run(self.action_grads, feed_dict={
			self.state_input: inputs,
			self.action_input: actions
		})

	# Updates the weights of the main model
	def train_model(self, state, action, q_value):
		return self.model.fit(x=[state, action], y=q_value, batch_size=self.batch_size, epochs=1, verbose=0)


class DDPGAgent:
	def __init__(self, env, state_transformer, actor_learning_rate=0.0001, critic_learning_rate=0.001,
					batch_size=64, discount_factor=0.99, tau=0.001, neighbors_percent=0.01, render=False):

		self.env = env
		self.state_transformer = state_transformer
		self.action_size = self.env.action_space.n
		self.enc = OneHotEncoder(handle_unknown='ignore')
		self.enc_actions = self.enc.fit_transform(self.env.actions)
		self.actor_output = self.enc_actions.shape[1]
		self.n_neighbors = math.ceil(neighbors_percent*self.action_size)
		self.neighbors = NearestNeighbors(n_neighbors=self.n_neighbors)
		self.neighbors.fit(self.enc_actions)

		self.state_size = state_transformer.dimensions
		self.discount_factor = discount_factor
		self.batch_size = batch_size
		self.train_start = 1000
		self.memory = deque(maxlen=2000)
		self.render = render

		self.session = self.generate_tensorflow_session()
		self.actor = Actor(self.session, self.state_size, self.actor_output,
						   learning_rate=actor_learning_rate, batch_size=batch_size, tau=tau)

		self.critic = Critic(self.session, self.state_size, self.n_neighbors, self.actor_output,
							 self.action_size, learning_rate=critic_learning_rate, batch_size=batch_size, tau=tau)

		print("Action size =", self.action_size, " N Neighbours =", self.n_neighbors, " State size =", self.state_size)

	def generate_tensorflow_session(self):
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		return tf.Session(config=config)

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
		apx_action = self.actor.get_action(state)

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
		apx_action = self.actor.get_action_target(state)

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
		self.actor.train_model(state, act_grads)

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
		for e in range(episodes):
			score, step = self.run_episode(max_steps)
			scores[e], steps[e] = score, step
			print("Episode:", e, "  steps:", step, "  score:", score, "  time:", time.time() - start)

		ensure_saved_models_dir()

		if plot_rewards:
			t_time = time.time() - start
			print("Mean score:", np.mean(scores), " Total steps:", np.sum(steps), " total time:", t_time)
			plot(scores)
			plot_running_avg(scores)
			np.save("./train_data/ddpg_enc_actions" + str(self.state_size) + str(self.n_neighbors) + "_scores", scores)
			np.save("./train_data/ddpg_enc_actions" + str(self.state_size) + str(self.n_neighbors) + "_time", t_time)
			np.save("./train_data/ddpg_enc_actions" + str(self.state_size) + str(self.n_neighbors) + "_steps", steps)

	def solve(self, permutation, its=100, max_steps=100, update_model=False):
		ans = None
		for _ in range(its):
			pans = self.run_episode(max_steps=max_steps, forced=permutation, update_model=update_model)
			if ans is None or pans > ans:
				ans = pans
		return -ans

	# Finds the breakpoint changes of each action for a state
	def fill_target(self, state, target):
		for idx, (_, i, j, k) in enumerate(self.env.actions):
			target[idx] = v_upperbound_breakpoints(state, i, j, k, self.discount_factor)

	# Function to add "good" steps into the replay memory
	# Helps the model learn some good actions early on
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
	env = PermutationSorting(n, transpositions=True)
	state_transformer = OneHotStateTransformer(n)
	agent = DDPGAgent(env, state_transformer, batch_size=64, neighbors_percent=0.1, render=False)
	agent.fill_memory()
	agent.train()

if __name__ == '__main__':
	if len(sys.argv) != 2:
		print("Missing Params")
		sys.exit()
	main(sys.argv)
