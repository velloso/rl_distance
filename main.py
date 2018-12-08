import numpy as np

import tensorflow as tf
import time
import re
from ddqn_tf import DDQNAgent
from wopertinger import DDPGAgent
from permutation_sorting import PermutationSorting
from state_transformers import OneHotStateTransformer
from util import random_line

# Compute and compare the DDQN for permutations size 10
# Compare with the exact distance
def ddqn_10():
	np.random.seed(12345678)
	n = 10
	env = PermutationSorting(n, transpositions=True)
	state_transformer = OneHotStateTransformer(n)
	ddqn = DDQNAgent(env, state_transformer, 0, hidden_layer_sizes=(400, 300), batch_size=256, train_start=60000,
					 epsilon=0.09, render=False,
					 train_path='./saved_models/ddqn_tf_final_weights_' + str(n) + '.h5')
	path = "./data/10_exact/all10urt"
	path_res = "./data/10_res/ddqn_exact"
	with tf.Session() as sess:
		ddqn.set_session(sess)
		ddqn.load_weights()
		n_better = 0
		n_worse = 0
		n_equal = 0
		t_dif = 0
		start = time.time()
		with open(path_res, 'w') as fp:
			for i in range(10000):
				f = open(path, 'r')
				line = random_line(f)
				p = np.fromstring(line, dtype=int, sep=',')
				p -= 1
				res = re.search(r'\s+\d+', line)
				exact_ans = int(res.group())
				f.close()
				ddqn_ans = ddqn.solve(p)
				dif = float(ddqn_ans)/float(exact_ans)
				t_dif = dif if t_dif == 0 else (dif + t_dif) / 2
				if ddqn_ans < exact_ans: n_better += 1
				elif ddqn_ans > exact_ans: n_worse += 1
				else: n_equal += 1
				string = str(i) + ' - ' + str(p) + ' - ' + ' DDQN: ' + str(ddqn_ans) + '  Exact: ' +\
						 str(exact_ans) + '  Difference: ' + str(dif)
				print(time.time() - start, string)
				fp.write(string + '\n')
			string = "Total difference: " + str(t_dif) + "  Times better: " + str(n_better) + "  Times worse: " +\
					 str(n_worse) + "  Times equal: " + str(n_equal)
			print(string)
			fp.write(string + '\n')

# Compute the  distance of WP for permutations size 10
def ddpg_10():
	n = 10
	neighbors = 0.8
	env = PermutationSorting(n, transpositions=True)
	state_transformer = OneHotStateTransformer(n)
	actor_layer_sizes = [400, 300]
	critic_layer_sizes = [400, 300]
	ddpg = DDPGAgent(env, state_transformer, actor_layer_sizes, critic_layer_sizes, batch_size=32,
					  actor_learning_rate=0.0001, critic_learning_rate=0.001,
					  train_start=60000, maxlen=1e6, neighbors_percent=neighbors, render=False,
					  train_path='./final_weights/ddpg_tf_final_weights_'+str(n)+'_'+str(neighbors)+'.h5')
	with tf.Session() as sess:
		ddpg.set_session(sess)
		ddpg.load_weights()
		path_db = "./data/10_res/10_db"
		path_res = "./data/10_res/wp.dist"
		start = time.time()
		i = 0
		with open(path_db, 'r') as fp, open(path_res, 'w') as res:
			line = fp.readline()
			while line:
				p = np.fromstring(line, dtype=int, sep=' ')
				ddpg_ans = ddpg.solve(p)
				print(time.time()-start, i, ddpg_ans)
				res.write(str(ddpg_ans)+'\n')
				i += 1
				line = fp.readline()

# Compute the  distance of WP for permutations size 15
def ddpg_15():
	n = 15
	neighbors = 0.8
	length = 1000
	env = PermutationSorting(n, transpositions=True)
	state_transformer = OneHotStateTransformer(n)
	rt = np.arange(1, 16)
	actor_layer_sizes = [400, 300]
	critic_layer_sizes = [400, 300]
	ddpg = DDPGAgent(env, state_transformer, actor_layer_sizes, critic_layer_sizes, batch_size=32,
					  actor_learning_rate=0.0001, critic_learning_rate=0.001,
					  train_start=60000, maxlen=1e6, neighbors_percent=neighbors, render=False,
					  train_path='./final_weights/ddpg_tf_final_weights_'+str(n)+'_'+str(neighbors)+'.h5')
	with tf.Session() as sess:
		ddpg.set_session(sess)
		ddpg.load_weights()
		for revt in rt:
			path_db = "./data/db15/perms-10k-r"+str(revt)+"-t"+str(revt)
			path_res = "./data/15_wp/wp-1k-r"+str(revt)+"-t"+str(revt)+".dist"
			start = time.time()
			with open(path_db, 'r') as fp, open(path_res, 'w') as res:
				line = fp.readline()
				i = 0
				while line and i < length:
					p = np.fromstring(line, dtype=int, sep=',')
					p = p-1
					ddpg_ans = ddpg.solve(p)
					print(time.time()-start, i, ddpg_ans)
					res.write(str(ddpg_ans)+'\n')
					i += 1
					line = fp.readline()

# Compare the results found with the WP to any other result
def compare(p_comp="./data/15_rahman/rahman-"):
	p_wp = "./data/15_wp/wp-1k-"
	rt = np.arange(1, 16)
	n_better = 0
	n_worse = 0
	n_equal = 0
	t_dif = 0
	for i in (rt):
		path_wp = p_wp+"r"+str(i)+"-t"+str(i)+".dist"
		path_comp = p_comp + "r" + str(i) + "-t" + str(i) + ".dist"
		with open(path_wp, 'r') as fp, open(path_comp, 'r') as fc:
			line = fp.readline()
			comp = fc.readline()
			while line and comp:
				comp_ans = int(comp)
				wp_ans = int(line)
				if wp_ans != 100:
					dif = wp_ans/comp_ans
					t_dif += dif
					if wp_ans < comp_ans: n_better += 1
					elif wp_ans > comp_ans: n_worse += 1
					else: n_equal += 1
				line = fp.readline()
				comp = fc.readline()
	total = (n_better+n_worse+n_equal)
	t_dif = t_dif/total
	string = "Total: "+str(total)+"  Total difference: "+str(t_dif)+"  Times better: "+\
			 str(n_better)+"("+str(n_better/total)+")  Times worse: "+str(n_worse)+"("+str(n_worse/total)+")  Times equal: "+str(n_equal)+"("+str(n_equal/total)+")"
	print(string)

if __name__ == '__main__':
	compare(p_comp="./data/15_walter/walter-")
	compare(p_comp="./data/15_dod/dod2015-")
	compare(p_comp="./data/15_breakpoint/bp-")
	compare(p_comp="./data/15_rahman/rahman-")
