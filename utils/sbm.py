import numpy as np 
from scipy.special import gamma
from itertools import combinations

def n_edges(n, d):

	return gamma(n + 1) / gamma(d + 1) / gamma(n - d + 1)

def hgsbm(q: int, m: int, p1: float, p2: float, d: float, prob = True, verbose = True):

	""" 
	--- probabilistic implementation of HSBM algorithm ---
	q:	number of communities
	m:	nodes per community 
	p1: probability of connecting nodes inside community 
	p2: probability of connecting nodes from different communities
	d:	interaction (hyper-edge) order

	prob: 	if true, use p1 and p2 as probabilities
			if false, p1 is the mixing parameter and p2 is the average degree
	verbose: print debugging information
	"""

	n = q * m 

	if verbose: print(f'- generating (hyper-)graph with {n} nodes, {q} communities of {m} nodes\n')

	n_tot = n_edges(n, d)
	n_in = q * n_edges(m, d)
	n_abc = m**d * n_edges(q, d)

	if prob:
		P, Q = p1, p2
		if verbose: print(f'- generating (hyper-)graph using input probabilities p: {P:.3f}, q: {Q:.3f}\n')
	else:
		mu = p1 
		c = p2 

		P = n * c * (1 - mu) / d / n_in 
		Q = n * c * mu / d / n_abc

		if verbose: print(f'- generating hypergraph using mixing parameter mu: {mu:.3f} and average degree <k>: {c:.3g}')
		if verbose: print(f'- resulting probabilities p: {P:.3f}, q: {Q:.3f}\n')

	if P > 1 or Q > 1:
		raise ValueError('input probabilities are not valid')

	in_edges, out_edges = [], []
	in_deg, out_deg = np.zeros(n), np.zeros(n)

	for edge in combinations(np.arange(n), d): 
		u = np.unique(np.array(edge)//m)
		if len(u) == 1:
			in_edges.append(list(edge)) if np.random.uniform() < P else ''
		if len(u) == d:
			out_edges.append(list(edge)) if np.random.uniform() < Q else ''

	for edge in in_edges:
		in_deg[edge] += 1 
	for edge in out_edges:
		out_deg[edge] += 1

	mu_a = out_deg / (in_deg + out_deg)
	mu_a[np.isnan(mu_a)] = 0
	mu_r = np.mean(mu_a)

    # if verbose: print(f'- generated (hyper-)graph of degree {d}')
    # if verbose: print(f'- {len(in_edges)} internal edges')
    # if verbose: print(f'- {len(out_edges)} external edges')
    # if verbose: print(f'- realized mixing parameter: {mu_r:.3f}')

	return in_edges + out_edges, mu_r


if __name__ == "__main__":

    for _ in range(10):
        edges, mu_r = hgsbm(2, 5, 1, .5, 2)
    # model = np.array([np.power(2,edge).sum() for edge in edges], dtype = int)
    # for i in model:
    #     print(f"{i:010b}")


