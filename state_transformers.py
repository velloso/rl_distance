import copy
import logging
import math

import networkx as nx
import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


# encontra todas as cliques de um grafo
def find_cliques(G):
	# Cache nbrs and find first pivot (highest degree)
	maxconn = -1
	nnbrs = {}
	pivotnbrs = set()  # handle empty graph

	for n, nbrs in G.adjacency():

		nbrs = set(nbrs)
		nbrs.discard(n)
		conn = len(nbrs)
		if conn > maxconn:
			nnbrs[n] = pivotnbrs = nbrs
			maxconn = conn
		else:
			nnbrs[n] = nbrs
	# Initial setup
	cand = set(nnbrs)
	smallcand = set(cand - pivotnbrs)
	done = set()
	stack = []
	clique_so_far = []
	# Start main loop
	while smallcand or stack:
		try:
			# Any nodes left to check?
			n = smallcand.pop()
		except KeyError:
			# back out clique_so_far
			cand, done, smallcand = stack.pop()
			clique_so_far.pop()
			continue
		# Add next node to clique
		clique_so_far.append(n)
		cand.remove(n)
		done.add(n)
		nn = nnbrs[n]
		new_cand = cand & nn
		new_done = done & nn
		# check if we have more to search
		if not new_cand:
			if not new_done:
				# Found a clique!
				yield clique_so_far[:]
			clique_so_far.pop()
			continue
		# Shortcut--only one node left!
		if not new_done and len(new_cand) == 1:
			yield clique_so_far + list(new_cand)
			clique_so_far.pop()
			continue
		# find pivot node (max connected in cand)
		# look in done nodes first
		numb_cand = len(new_cand)
		maxconndone = -1
		for n in new_done:
			cn = new_cand & nnbrs[n]
			conn = len(cn)
			if conn > maxconndone:
				pivotdonenbrs = cn
				maxconndone = conn
				if maxconndone == numb_cand:
					break
		# Shortcut--this part of tree already searched
		if maxconndone == numb_cand:
			clique_so_far.pop()
			continue
		# still finding pivot node
		# look in cand nodes second
		maxconn = -1
		for n in new_cand:
			cn = new_cand & nnbrs[n]
			conn = len(cn)
			if conn > maxconn:
				pivotnbrs = cn
				maxconn = conn
				if maxconn == numb_cand - 1:
					break
		# pivot node is max connected in cand from done or cand
		if maxconndone > maxconn:
			pivotnbrs = pivotdonenbrs
		# save search status for later backout
		stack.append((cand, done, smallcand))
		cand = new_cand
		done = new_done
		smallcand = cand - pivotnbrs


# Funcao que auxilia em obter uma clique maximal
def make_max_clique_graph(G, create_using=None, name=None):
	cliq = list(map(set, find_cliques(G)))
	if create_using:
		B = create_using
		B.clear()
	else:
		B = nx.Graph()
	if name is not None:
		B.name = name

	for i, cl in enumerate(cliq):
		B.add_node(i + 1)
		for j, other_cl in enumerate(cliq[:i]):
			# if not cl.isdisjoint(other_cl): #Requires 2.6
			intersect = cl & other_cl
			if intersect:  # Not empty
				B.add_edge(i + 1, j + 1)
	return B


# encontra todos os 2-ciclos de uma permutacao
def get_two_cycles(blackg, position, permutation, n):
	two_cyclesb = []
	two_cyclesg = []

	for edge_i in blackg.edges():
		black11 = edge_i[0]
		black12 = edge_i[1]

		gray11 = edge_i[0]
		gray1_2 = []

		gray21 = edge_i[1]
		gray2_2 = []

		if black11 < n + 1:
			if abs(position[black11] - position[black11 + 1]) != 1:
				gray1_2.append(black11 + 1)
		if black11 > 0:
			if abs(position[black11] - position[black11 - 1]) != 1:
				gray1_2.append(black11 - 1)
		if black12 < n + 1:
			if abs(position[black12] - position[black12 + 1]) != 1:
				gray2_2.append(black12 + 1)
		if black12 > 0:
			if abs(position[black12] - position[black12 - 1]) != 1:
				gray2_2.append(black12 - 1)

		for gray12 in gray1_2:
			for gray22 in gray2_2:
				if gray12 != gray22 and abs(position[gray12] - position[gray22]) == 1:
					if abs(gray12 - gray22) != 1:
						node1 = [min(black11, black12), max(black11, black12)]
						node2 = [min(gray12, gray22), max(gray12, gray22)]
						node3 = [min(gray11, gray12), max(gray11, gray12)]
						node4 = [min(gray21, gray22), max(gray21, gray22)]
						new2cycleb = [node1, node2]
						new2cycleg = [node3, node4]

						new2cycleb.sort()
						new2cycleg.sort()

						if new2cycleb not in two_cyclesb:
							two_cyclesb.append(new2cycleb)
							if new2cycleg not in two_cyclesg:
								two_cyclesg.append(new2cycleg)
							else:
								return "error!"

	return two_cyclesb, two_cyclesg


# encontra todos os 3-ciclos de uma permutacao
def get_three_cycles(blackg, position, permutation, n):
	three_cyclesb = []
	three_cyclesg = []

	for edge_i in blackg.edges():
		black11 = edge_i[0]
		black12 = edge_i[1]

		gray11 = edge_i[0]
		gray1_2 = []

		gray21 = edge_i[1]
		gray2_2 = []

		if black11 < n + 1:
			if abs(position[black11] - position[black11 + 1]) != 1:
				gray1_2.append(black11 + 1)
		if black11 > 0:
			if abs(position[black11] - position[black11 - 1]) != 1:
				gray1_2.append(black11 - 1)
		if black12 < n + 1:
			if abs(position[black12] - position[black12 + 1]) != 1:
				gray2_2.append(black12 + 1)
		if black12 > 0:
			if abs(position[black12] - position[black12 - 1]) != 1:
				gray2_2.append(black12 - 1)

		for gray12 in gray1_2:
			for gray22 in gray2_2:
				black21 = gray12
				black2_2 = []

				black31 = gray22
				black3_2 = []

				if position[black21] < n + 1:
					if abs(black21 - permutation[position[black21] + 1]) != 1:
						black2_2.append(permutation[position[black21] + 1])
				if position[black21] > 0:
					if abs(black21 - permutation[position[black21] - 1]) != 1:
						black2_2.append(permutation[position[black21] - 1])
				if position[black31] < n + 1:
					if abs(black31 - permutation[position[black31] + 1]) != 1:
						black3_2.append(permutation[position[black31] + 1])
				if position[black31] > 0:
					if abs(black31 - permutation[position[black31] - 1]) != 1:
						black3_2.append(permutation[position[black31] - 1])

				for black22 in black2_2:
					for black32 in black3_2:
						if not (min(black21, black22) == min(black31, black32) and max(black21, black22) == max(black31,
																												black32)):
							if abs(black22 - black32) == 1 and abs(position[black22] - position[black32]) != 1:
								node1 = [min(black11, black12), max(black11, black12)]
								node2 = [min(black21, black22), max(black21, black22)]
								node3 = [min(black31, black32), max(black31, black32)]
								node4 = [min(gray11, gray12), max(gray11, gray12)]
								node5 = [min(gray21, gray22), max(gray21, gray22)]
								node6 = [min(black22, black32), max(black22, black32)]

								new3cycleb = [node1, node2, node3]
								new3cycleg = [node4, node5, node6]
								new3cycleb.sort()
								new3cycleg.sort()
								if new3cycleb not in three_cyclesb:
									three_cyclesb.append(new3cycleb)
									if new3cycleg not in three_cyclesg:
										three_cyclesg.append(new3cycleg)
									else:
										return "error!"

	return three_cyclesb, three_cyclesg


def run(g_perm_string):
	permutation = eval("[%s]" % g_perm_string)

	n = len(permutation)

	permutation = [0] + permutation + [n + 1]
	position = [-1 for i in range(0, n + 2)]

	for i in range(0, n + 2):
		position[permutation[i]] = i

	one_cycles = []
	two_cycles = []
	three_cycles = []
	long_cycles = []

	blackg = nx.Graph()
	cyclegraph = nx.Graph()
	blackg.add_nodes_from([0, n + 1])

	for i in range(0, n + 1):
		if abs(permutation[i] - permutation[i + 1]) != 1:
			# se a diferenca entre dois elementos vizinhos eh diferente de um, adiciona ao grafo
			blackg.add_edge(permutation[i], permutation[i + 1], nodetype=1)
		else:
			# senao, eh um 1-ciclo e nao deve ser adicionado ao grafo
			one_cycles.append([permutation[i], permutation[i + 1]])

	# pega todos os 2-ciclos do grafo de ciclos da permutacao
	two_cb, two_cg = get_two_cycles(blackg, position, permutation, n)

	# pega todos os 3-ciclos do grafo de ciclos da permutacao
	three_cb, three_cg = get_three_cycles(blackg, position, permutation, n)

	## Constroi um grafo com todos os 2-ciclos e 3-ciclos para,
	## com ajuda da clique, retirar um conjunto maximal
	all23_cyclesb = two_cb + three_cb
	all23_cycleslist = copy.deepcopy(all23_cyclesb)
	all23_cyclesg = two_cg + three_cg

	for i in range(0, len(all23_cyclesb)):
		all23_cyclesb[i][0] = "(%s,%s)" % (all23_cycleslist[i][0][0], all23_cycleslist[i][0][1])
		all23_cyclesb[i][1] = "(%s,%s)" % (all23_cycleslist[i][1][0], all23_cycleslist[i][1][1])

		all23_cyclesg[i][0] = "(%s,%s)" % (all23_cyclesg[i][0][0], all23_cyclesg[i][0][1])
		all23_cyclesg[i][1] = "(%s,%s)" % (all23_cyclesg[i][1][0], all23_cyclesg[i][1][1])
		if len(all23_cyclesb[i]) == 3:
			all23_cyclesb[i][2] = "(%s,%s)" % (all23_cycleslist[i][2][0], all23_cycleslist[i][2][1])
			all23_cyclesg[i][2] = "(%s,%s)" % (all23_cyclesg[i][2][0], all23_cyclesg[i][2][1])

	cyclegraph.add_nodes_from(range(0, len(all23_cyclesb)))
	for i in range(0, len(all23_cyclesb)):
		for j in range(i + 1, len(all23_cyclesb)):
			if list(set(all23_cyclesb[i]) & set(all23_cyclesb[j])) != []:
				cyclegraph.add_edge(i, j)
			if list(set(all23_cyclesg[i]) & set(all23_cyclesg[j])) != []:
				cyclegraph.add_edge(i, j)

	## Pega o complemento do grafo, uma vez que se pegarmos uma clique maximal
	## de 2,3-ciclos no complemento, significa que eles sao independentes com
	## relacao a suas arestas. Com o complemento, procuramos uma clique maximal.
	cycle_graph_inv = nx.complement(cyclegraph)

	result = list(map(set, find_cliques(cycle_graph_inv)))

	## Se retornou um conjunto maximal, pega os ciclos deste conjunto
	## e adiciona na lista final de ciclos do grafo da permutacao,
	## separando entre 2-ciclos e 3-ciclos

	finalcycles = []
	if result != []:
		# print len(result)
		# print result
		# finalcycles = list(max(result,key=len))
		finalcycles = list(result[len(list(result)) - 1])
	# print finalcycles

	for finalcycle in finalcycles:
		cycleaux = []
		if len(all23_cycleslist[finalcycle]) == 2:
			cycleaux.append(all23_cycleslist[finalcycle][0][0])

			if abs(all23_cycleslist[finalcycle][1][0] - cycleaux[-1]) == 1:
				cycleaux.append(all23_cycleslist[finalcycle][1][0])
				cycleaux.append(all23_cycleslist[finalcycle][1][1])
			else:
				cycleaux.append(all23_cycleslist[finalcycle][1][1])
				cycleaux.append(all23_cycleslist[finalcycle][1][0])

			cycleaux.append(all23_cycleslist[finalcycle][0][1])

			two_cycles.append(cycleaux)

		elif len(all23_cycleslist[finalcycle]) == 3:
			cycleaux.append(all23_cycleslist[finalcycle][0][0])
			if abs(all23_cycleslist[finalcycle][1][0] - cycleaux[-1]) == 1:
				cycleaux.append(all23_cycleslist[finalcycle][1][0])
				cycleaux.append(all23_cycleslist[finalcycle][1][1])
			else:
				cycleaux.append(all23_cycleslist[finalcycle][1][1])
				cycleaux.append(all23_cycleslist[finalcycle][1][0])

			if abs(all23_cycleslist[finalcycle][2][0] - cycleaux[-1]) == 1 and abs(
					all23_cycleslist[finalcycle][2][1] - all23_cycleslist[finalcycle][0][1]) == 1:
				cycleaux.append(all23_cycleslist[finalcycle][2][0])
				cycleaux.append(all23_cycleslist[finalcycle][2][1])
			else:
				cycleaux.append(all23_cycleslist[finalcycle][2][1])
				cycleaux.append(all23_cycleslist[finalcycle][2][0])

			cycleaux.append(all23_cycleslist[finalcycle][0][1])

			three_cycles.append(cycleaux)
		else:
			print("Error")

	## Agora temos uma lista de 1-ciclos, uma de 2-ciclos e uma de
	## 3-ciclos. Gulosamente vamos criar os demais itens com as arestas
	## restantes. Como qualquer aresta cinza pode ser utilizada uma unica
	## vez, vamos salvar em available[i] se a aresta (i, i+1) ja foi utilizada.

	gray_available = [1 for i in range(0, n + 1)]
	black_available = [1 for i in range(0, n + 1)]

	for cycle in one_cycles:
		gray_edge = min(cycle[0], cycle[1])
		black_edge = min(position[cycle[0]], position[cycle[1]])
		gray_available[gray_edge] = gray_available[gray_edge] - 1
		black_available[permutation[black_edge]] = black_available[permutation[black_edge]] - 1

	for cycle in two_cycles:
		gray_edge = min(cycle[0], cycle[1])
		black_edge = min(position[cycle[1]], position[cycle[2]])
		gray_available[gray_edge] = gray_available[gray_edge] - 1
		black_available[permutation[black_edge]] = black_available[permutation[black_edge]] - 1

		gray_edge = min(cycle[2], cycle[3])
		black_edge = min(position[cycle[3]], position[cycle[0]])
		gray_available[gray_edge] = gray_available[gray_edge] - 1
		black_available[permutation[black_edge]] = black_available[permutation[black_edge]] - 1

	for cycle in three_cycles:
		gray_edge = min(cycle[0], cycle[1])
		black_edge = min(position[cycle[1]], position[cycle[2]])
		gray_available[gray_edge] = gray_available[gray_edge] - 1
		black_available[permutation[black_edge]] = black_available[permutation[black_edge]] - 1

		gray_edge = min(cycle[2], cycle[3])
		black_edge = min(position[cycle[3]], position[cycle[4]])
		gray_available[gray_edge] = gray_available[gray_edge] - 1
		black_available[permutation[black_edge]] = black_available[permutation[black_edge]] - 1

		gray_edge = min(cycle[4], cycle[5])
		black_edge = min(position[cycle[5]], position[cycle[0]])
		gray_available[gray_edge] = gray_available[gray_edge] - 1
		black_available[permutation[black_edge]] = black_available[permutation[black_edge]] - 1

	for i in range(0, n + 1):
		if gray_available[i]:
			start = i

			cycle = [start]
			end = start
			while True:
				## Gray edge
				if end < n + 1 and gray_available[end]:
					gray_available[end] = gray_available[end] - 1
					end = end + 1
					cycle.append(end)
				else:
					gray_available[end - 1] = gray_available[end - 1] - 1
					end = end - 1
					cycle.append(end)

				## Black edge
				next_end = None
				if end < n + 1 and black_available[end]:
					next_end = permutation[position[end] + 1]
					black_available[end] = black_available[end] - 1
				else:
					next_end = permutation[position[end] - 1]
					black_available[next_end] = black_available[next_end] - 1

				if next_end == start:
					break
				else:
					cycle.append(next_end)
					end = next_end

			long_cycles.append(cycle)

	## Retorna a lista final de ciclos
	return one_cycles + two_cycles + three_cycles + long_cycles


class FeatureGenerator:
	def get_normalized_perm(self, perm):
		return [0] + perm + [len(perm) + 1]

	def nof_breakpoints_rev(self, g_perm):
		perm = self.get_normalized_perm(g_perm)
		bp = 0
		n = len(perm)
		for i in range(n - 1):
			if (perm[i] - perm[i + 1] != 1 and perm[i] - perm[i + 1] != -1):
				bp += 1
		return bp

	def nof_breakpoints_trans(self, g_perm):
		perm = self.get_normalized_perm(g_perm)
		bp = 0
		n = len(perm)
		for i in range(n - 1):
			if (perm[i] + 1 != perm[i + 1]):
				bp += 1
		return bp

	def get_strips_info(self, g_perm):
		perm = g_perm
		n = len(perm)
		tam_strip = 0
		tam_maior = 0
		n_strips = 0
		sum_tam_strips = 0
		n_strips_crescente = 0
		n_strips_decrescente = 0
		sum_tam_strips_crescente = 0
		sum_tam_strips_decrescente = 0
		i = 0
		while i < n - 1:
			# crescent
			if perm[i + 1] == perm[i] + 1 and i < n:
				comeco_strip = i
				while i < n - 1 and perm[i + 1] == perm[i] + 1:
					i += 1

				tam_strip = i - comeco_strip + 1
				if tam_strip > tam_maior:
					tam_maior = tam_strip
				n_strips += 1
				sum_tam_strips = sum_tam_strips + tam_strip
				n_strips_crescente += 1
				sum_tam_strips_crescente = sum_tam_strips_crescente + tam_strip

			# decrescent
			elif perm[i + 1] == perm[i] - 1 and i < n:
				comeco_strip = i
				while i < n - 1 and perm[i + 1] == perm[i] - 1:
					i += 1
				tam_strip = i - comeco_strip + 1
				if tam_strip > tam_maior:
					tam_maior = tam_strip
				n_strips += 1
				sum_tam_strips = sum_tam_strips + tam_strip
				n_strips_decrescente += 1
				sum_tam_strips_decrescente = sum_tam_strips_decrescente + tam_strip

			i += 1

		return [tam_maior,
				n_strips,
				sum_tam_strips,
				n_strips_crescente,
				n_strips_decrescente,
				sum_tam_strips_crescente,
				sum_tam_strips_decrescente]

	def get_correct_position(self, g_perm):
		perm = g_perm
		a = np.asarray(perm)
		identity = np.asarray(list(range(1, len(perm) + 1)))
		return np.sum(a == identity)

	def get_fixed_elements(self, g_perm):
		perm = self.get_normalized_perm(g_perm)
		n = len(perm)
		maior = 0
		count = 0

		for i in range(1, n - 1):
			if (perm[i] == i):
				if (perm[i] > maior):
					count += 1

			if (maior < perm[i]):
				maior = perm[i]

		return count

	def get_inversions(self, g_perm):
		def merge(arr, temp, left, mid, right):
			inv_count = 0

			i = left
			j = mid
			k = left

			while (i <= mid - 1) and (j <= right):
				if (arr[i] <= arr[j]):
					temp[k] = arr[i]
					k += 1
					i += 1
				else:
					temp[k] = arr[j]
					k += 1
					j += 1
					inv_count = inv_count + (mid - i)

			while i <= mid - 1:
				temp[k] = arr[i]
				k += 1
				i += 1

			while j <= right:
				temp[k] = arr[j]
				k += 1
				j += 1

			for i in range(left, right + 1):
				arr[i] = temp[i]

			return inv_count

		def _merge_sort(arr, temp, left, right):
			mid = None
			inv_count = 0
			if (right > left):
				mid = int(math.floor((right + left) / 2));

				inv_count = _merge_sort(arr, temp, left, mid);
				inv_count += _merge_sort(arr, temp, mid + 1, right);

				inv_count += merge(arr, temp, left, mid + 1, right);

			return inv_count;

		def merge_sort(arr, temp, array_size):
			return _merge_sort(arr, temp, 0, array_size - 1)

		perm = self.get_normalized_perm(g_perm)
		arr = list(perm)  # makes copy to prevent in-place modification
		array_size = len(arr)
		temp = np.zeros(array_size)

		return merge_sort(arr, temp, array_size)

	def get_prefix(self, g_perm):
		perm = g_perm
		n = len(perm)

		cond = True
		count = 0

		for i in range(0, n):
			if not cond:
				break
			if perm[i] == i + 1:
				count += 1
			else:
				cond = False

		return count

	def get_suffix(self, g_perm):
		perm = self.get_normalized_perm(g_perm)
		cond = True
		count = 0
		n = len(perm)
		i = n - 2

		while i > 0 and cond:
			if perm[i] == i:
				count += 1
			else:
				cond = False
			i -= 1

		return count

	def lis_calc(self, perm, n):
		output = np.zeros(n)
		j = None
		lo = None
		hi = None
		mid = None
		pos = None
		L = None
		M = np.zeros(n + 1).astype(int)
		P = np.zeros(n + 1).astype(int)
		L = 0

		for i in range(1, n):
			if L == 0 or perm[M[1]] >= perm[i]:
				j = 0
			else:
				lo = 1
				hi = L + 1
				while (lo < (hi - 1)):
					if (lo + hi) % 2 == 0:
						mid = int(math.floor((lo + hi) / 2))
					else:
						mid = int(math.floor((lo + hi - 1) / 2))
					if perm[M[mid]] < perm[i]:
						lo = mid
					else:
						hi = mid

				j = lo

			P[i] = M[j]
			if j == L or perm[i] < perm[M[j + 1]]:
				M[j + 1] = i
				if (L < j + 1):
					L = j + 1

		i = 1
		pos = M[L]
		while L > 0:
			# print pos
			output[i] = perm[pos]
			i += 1
			pos = P[pos]
			L -= 1

		output[0] = i - 1
		return int(output[0])

	def get_lis(self, g_perm):
		perm = self.get_normalized_perm(g_perm)
		n = len(perm)
		return self.lis_calc(perm, n) - 1  # changed + for -

	def get_lds(self, g_perm):
		def reversal(perm, i, j):
			for k in range(i, j - i / 2 + 1):
				aux = perm[k]
				perm[k] = perm[j - k + i]
				perm[j - k + i] = aux

		perm = self.get_normalized_perm(g_perm)
		n = len(perm)
		perm = list(reversed(perm))
		return self.lis_calc(perm, n)

	def get_entropy(self, g_perm):
		def get_position(perm, n):
			position = [0] * n
			for i in range(0, n):
				position[perm[i]] = i

			return position

		def get_entro(position, n):
			total = 0
			for i in range(0, n):
				total += abs(i - position[i])

			return total

		perm = self.get_normalized_perm(g_perm)
		n = len(perm)
		return get_entro(get_position(perm, n), n)

	def get_left_right_code(self, g_perm):
		def coding(perm, n, type_):
			coding_array = np.zeros(n).astype(int)
			if type_ == 'left':
				for i in range(2, n - 1):
					for j in range(1, i):
						if perm[i] < perm[j]:
							coding_array[i] += 1

			elif type_ == 'right':
				for i in range(1, n - 2):
					for j in range(i + 1, n - 1):
						if perm[i] > perm[j]:
							coding_array[i] += 1

			else:
				raise Exception('TypeError')

			return coding_array

		def coding_value(coding_array, n):
			ca_value = 0
			i = 0

			ca_aux = coding_array[1]
			if ca_aux != 0:
				ca_value = 1
			else:
				ca_value = 0
			for i in range(2, n - 1):
				if (coding_array[i] != ca_aux):
					ca_aux = coding_array[i]
					if ca_aux > 0:
						ca_value += 1

			return ca_value

		perm = self.get_normalized_perm(g_perm)
		n = len(perm)
		right = coding_value(coding(perm, n, 'right'), n)
		left = coding_value(coding(perm, n, 'left'), n)

		return left, right

	def get_cycles(self, g_perm):
		def get_position(permutation):
			n = len(permutation) - 2
			position = [-1 for i in range(0, n + 2)]
			for i in range(0, n + 2):
				position[abs(permutation[i])] = i
			return position

		def get_rightmost_element(cycle, position):
			max_position = 0
			for i in range(len(cycle)):
				if position[cycle[i]] > position[cycle[max_position]]:
					max_position = i
			return max_position

		## The unordered cycle starts with a gray edge, we order them by
		## making it start with the rightmost black edge.
		def order_cycle(cycle, position):
			index = get_rightmost_element(cycle, position)
			new = []
			new.append(cycle[index])

			if index % 2 == 0:
				iter_el = (index - 1) % len(cycle)
				while iter_el != index:
					new.append(cycle[iter_el])
					iter_el = (iter_el - 1) % len(cycle)
			else:
				iter_el = (index + 1) % len(cycle)
				while iter_el != index:
					new.append(cycle[iter_el])
					iter_el = (iter_el + 1) % len(cycle)
			return new

		## This will transform this cycle representation in the other one that
		## numbers the black edges. This will simplify the transformation into
		## simple permutation since we will use the cycle graph linked
		## structure.
		def canonical_representation(cycle, position):
			cycle = order_cycle(cycle, position)
			canonical = []

			for i in range(0, len(cycle), 2):
				if position[cycle[i]] < position[cycle[i + 1]]:
					black = -position[cycle[i + 1]]
					canonical.append(black)
				else:
					black = position[cycle[i]]
					canonical.append(black)
			return canonical

		def get_canonicals(str_permutation, int_position):
			canonicals = []
			int_cycles = run(str_permutation)
			for cycle in int_cycles:
				canonicals.append(canonical_representation(cycle, int_position))

			return canonicals

		n = len(g_perm)
		str_permutation = str(g_perm)[1:-1]
		permutation = self.get_normalized_perm(g_perm)
		int_position = get_position(permutation)

		# str_cycles = get_canonicals(str_permutation, int_position)
		# print str_cycles
		can = get_canonicals(str_permutation, int_position)
		num_ciclos = len(can)
		num_ciclos_impares = 0
		num_ciclos_curtos = 0
		num_ciclos_pares_div = 0
		num_ciclos_orientados = 0
		num_ciclos_unitarios = 0
		tamanho_maior_ciclo = 1

		for cycles in can:
			if len(cycles) % 2 != 0:
				num_ciclos_impares = num_ciclos_impares + 1
			if len(cycles) < 4:
				num_ciclos_curtos = num_ciclos_curtos + 1
			if len(cycles) == 2 and cycles[0] * cycles[1] < 0:
				num_ciclos_pares_div = num_ciclos_pares_div + 1
			if len(cycles) > 2:
				found_oriented = 0
				for i in range(1, len(cycles) - 1):
					if not found_oriented:
						for j in range(i + 1, len(cycles)):
							if not found_oriented:
								if cycles[i] > 0 and cycles[j] > 0:
									if cycles[j] > cycles[i]:
										num_ciclos_orientados = num_ciclos_orientados + 1
										found_oriented = 1
								elif cycles[i] < 0 and cycles[j] < 0:
									if cycles[j] > cycles[i]:
										num_ciclos_orientados = num_ciclos_orientados + 1
										found_oriented = 1
								else:
									if abs(cycles[j]) > abs(cycles[i]):
										num_ciclos_orientados = num_ciclos_orientados + 1
										found_oriented = 1
			if len(cycles) == 1:
				num_ciclos_unitarios = num_ciclos_unitarios + 1
			if len(cycles) > tamanho_maior_ciclo:
				tamanho_maior_ciclo = len(cycles)
		# falta componentes
		# falta maior componente
		can2 = [s for s in can if len(s) > 1]

		num_of_components = len(can) - len(can2)
		biggest_component = 1
		biggest_cycles_component = 1

		while len(can2) > 0:
			current_component = []
			first = can2.pop(0)
			current_component.append(first)

			lenght_current_component = len(first)
			num_of_components = num_of_components + 1
			cycles_component = 1

			while len(current_component) > 0:
				atual = current_component.pop(0)
				for i in range(-1, len(atual) - 1):
					j = 0
					while j < len(can2):
						removed = 0
						for k in range(-1, len(can2[j]) - 1):
							a = min(abs(atual[i]), abs(atual[i + 1]))
							b = max(abs(atual[i]), abs(atual[i + 1]))
							c = min(abs(can2[j][k]), abs(can2[j][k + 1]))
							d = max(abs(can2[j][k]), abs(can2[j][k + 1]))

							if ((c > a) and (b > c) and (d > b)) or ((a > c) and (d > a) and (b > d)):
								cycles_component = cycles_component + 1
								lenght_current_component = lenght_current_component + len(can2[j])
								current_component.append(can2.pop(j))
								removed = 1
								break
						if not removed:
							j = j + 1

			if (cycles_component > biggest_cycles_component):
				biggest_cycles_component = cycles_component

			if (lenght_current_component > biggest_component):
				biggest_component = lenght_current_component

		return [num_ciclos,
				num_ciclos_impares,
				num_ciclos_curtos,
				num_ciclos_pares_div,
				num_ciclos_orientados,
				num_ciclos_unitarios,
				tamanho_maior_ciclo,
				num_of_components,
				biggest_component,
				biggest_cycles_component]

	def get_tuples(self, result):
		keys = [
			'breakpoints reversao',
			'breakpoints trans',
			'01 + 02',
			'tamanho maior strip nu',
			'numero de strip nu',
			'soma dos tamanhos das strips nu',
			'strips crescentes nu',
			'strips decrescentes nu',
			'soma tamanho strips crescentes nu',
			'soma tamanho strips decrescentes nu',
			'elementos na posicao correta',
			'elementos fixos',
			'inversoes',
			'prefixo correto',
			'sufixo correto',
			'lis',
			'lds',
			'entropia',
			'left code',
			'right code',
			'numero de ciclos',
			'ciclos impares',
			'ciclos curtos',
			'ciclos pares divergente',
			'ciclos orientados',
			'ciclos unitarios',
			'maior ciclo',
			'numero de componentes',
			'tamanho da maior componente em numero de arestas',
			'tamanho da maior componente em numero de ciclos'
		]

		data = []
		for i in range(0, len(keys)):
			data.append((keys[i], result[i]))

		return data

	def run(self, perm):
		result = []
		result.append(self.nof_breakpoints_rev(perm))
		result.append(self.nof_breakpoints_trans(perm))
		result.append(result[0] + result[1])
		result += self.get_strips_info(perm)
		result.append(self.get_correct_position(perm))
		result.append(self.get_fixed_elements(perm))
		result.append(self.get_inversions(perm))
		result.append(self.get_prefix(perm))
		result.append(self.get_suffix(perm))
		result.append(self.get_lis(perm))
		result.append(self.get_lds(perm))
		result.append(self.get_entropy(perm))
		result += self.get_left_right_code(perm)
		result += self.get_cycles(perm)

		return result

	def run_tuples(self, perm):
		return self.get_tuples(self.run(perm))


class MaximumFeatureValue:
	def __init__(self, perm_size):
		self.perm_size = perm_size

	def get_maximums(self):
		result = np.zeros(30).astype(int)
		result[[3, 5, 8, 9, 10, 11, 13, 14, 15, 16]] = self.perm_size
		result[[0, 1, 20, 21, 22, 25, 26, 27, 28]] = self.perm_size + 1
		result[[4, 6, 7, 23, 29]] = self.perm_size / 2
		result[2] = 2 * (self.perm_size + 1)
		result[12] = self.perm_size * (self.perm_size - 1) / 2
		result[17] = self.perm_size * self.perm_size / 2
		result[[18, 19]] = self.perm_size - 1
		result[24] = self.perm_size / 3

		return result.astype(np.float)


class FlaviosStateTransformer:
	def __init__(self, n):
		self.maximums = np.array(MaximumFeatureValue(n).get_maximums())
		self.generator = FeatureGenerator()
		self.dimensions = 30

	def transform(self, permutation):
		permutation = list(permutation + 1)
		features = self.generator.run(permutation)
		return np.atleast_2d(features / self.maximums)


class MaxStateTransformer:
	def __init__(self, n):
		self._n = n
		self.dimensions = n

	def transform(self, permutation):
		return np.atleast_2d(permutation / (self._n - 1))


class IdentityStateTransformer:
	def __init__(self, n):
		self._n = n
		self.dimensions = n

	@staticmethod
	def transform(permutation):
		return permutation


class OneHotStateTransformer:
	def __init__(self, n):
		self._n = n
		self._n2 = n * n
		self.dimensions = self._n2

	def transform(self, permutation):
		ans = np.zeros((1, self._n2))
		for i in range(self._n):
			ans[0, self._n * i + permutation[i]] = 1.0
		return ans


class RBFStateTransformer:
	def __init__(self, env, inner_transformer, gammas=(5.0, 2.0, 1.0, 0.5), n_components=1000, n_examples=1000):
		examples = np.array([inner_transformer.transform(env.observation_space.sample()) for _ in range(n_examples)])

		scaler = StandardScaler()
		featurizer = FeatureUnion(
			[('rbf%d' % idx, RBFSampler(gamma=gamma, n_components=n_components)) for idx, gamma in enumerate(gammas)])
		example_features = featurizer.fit_transform(scaler.fit_transform(examples))

		self.inner_transformer = inner_transformer
		self.dimensions = example_features.shape[1]
		self.scaler = scaler
		self.featurizer = featurizer

	def transform(self, obs):
		inner_transformed = self.inner_transformer.transform(obs)
		scaled = self.scaler.transform([inner_transformed])
		return self.featurizer.transform(scaled)[0]


class OneHotRBFStateTransformer:
	def __init__(self, env):
		self._n = env.observation_space.n
		self.rbf = RBFStateTransformer(env, OneHotStateTransformer(self._n))
		self.dimensions = self.rbf.dimensions

	def transform(self, permutation):
		return self.rbf.transform(permutation)
