import networkx as nx
import numpy as np
import scipy.stats
from analysis import graph_metrics
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
import powerlaw 
from community import community_louvain
def make_histograms(
	value_arrs, num_bins=None, bin_width=None, bin_array=None, frequency=False,
	epsilon=1e-6
):
	"""
	Given a set of value arrays, converts them into histograms of counts or
	frequencies. The bins may be specified as a number of bins, a bin width, or
	a pre-defined array of bin edges. This function creates histograms such that
	all value arrays given are transformed into the same histogram space.
	Arguments:
		`value_arrs`: an iterable of N 1D NumPy arrays to make histograms of
		`num_bins`: if given, make the histograms have this number of bins total
		`bin_width`: if given, make the histograms have bins of this width,
			aligned starting at the minimum value
		`bin_array`: if given, make the histograms according to this NumPy array
			of bin edges
		`frequency`: if True, normalize each histogram into frequencies
		`epsilon`: small number for stability of last endpoint if `bin_width` is
			specified
	Returns an N x B array of counts or frequencies (N is parallel to the input
	`value_arrs`), where B is the number of bins in the histograms.
	"""
	# Compute bins if needed
	if num_bins is not None:
		assert bin_width is None and bin_array is None
		min_val = min(np.nanmin(arr) for arr in value_arrs)
		max_val = max(np.nanmax(arr) for arr in value_arrs)
		bin_array = np.linspace(min_val, max_val, num_bins + 1)
	elif bin_width is not None:
		assert num_bins is None and bin_array is None
		min_val = min(np.nanmin(arr) for arr in value_arrs)
		max_val = max(np.nanmax(arr) for arr in value_arrs) + bin_width + \
			epsilon
		bin_array = np.arange(min_val, max_val, bin_width)
	elif bin_array is not None:
		assert num_bins is None and bin_width is None
	else:
		raise ValueError(
			"Must specify one of `num_bins`, `bin_width`, or `bin_array`"
		)

	# Compute histograms
	hists = np.empty((len(value_arrs), len(bin_array) - 1))
	for i, arr in enumerate(value_arrs):
		hist = np.histogram(arr, bins=bin_array)[0]
		if frequency:
			hist = hist / len(arr)
		hists[i] = hist
	
	return hists


def gaussian_kernel(vec_1, vec_2, sigma=1):
	"""
	Computes the Gaussian kernel function on two vectors. This is also known as
	the radial basis function. 
	Arguments:
		`vec_1`: a NumPy array of values
		`vec_2`: a NumPy array of values; the underlying vector space must be
			the same as `vec_1`
		`sigma`: standard deviation for the Gaussian kernel
	Returns a scalar similarity value between 0 and 1.
	"""
	l2_dist_squared = np.sum(np.square(vec_1 - vec_2))
	return np.exp(-l2_dist_squared / (2 * sigma * sigma))


def gaussian_wasserstein_kernel(vec_1, vec_2, sigma=1):
	"""
	Computes the Gaussian kernel function on two vectors, where the similarity
	metric within the Gaussian is the Wasserstein distance (i.e. Earthmover's
	distance). The two vectors must be distributions represented as PMFs over
	the same probability space.
	Arguments:
		`vec_1`: a NumPy array representing a PMF distribution (values are
			probabilities)
		`vec_2`: a NumPy array representing a PMF distribution (values are
			probabilities); the underlying probability space (i.e. support) must
			be the same as `vec_1`
		`sigma`: standard deviation for the Gaussian kernel
	Returns a scalar similarity value between 0 and 1.
	"""
	assert vec_1.shape == vec_2.shape
	# Since the vectors are supposed to be PMFs, if everything is 0 then just
	# turn it into an (unnormalized) uniform distribution
	if np.all(vec_1 == 0):
		vec_1 = np.ones_like(vec_1)
	if np.all(vec_2 == 0):
		vec_2 = np.ones_like(vec_2)
	# The SciPy Wasserstein distance function takes in empirical observations
	# instead of histograms/distributions as an input, but we can get the same
	# result by specifying weights which are the PMF probabilities
	w_dist = scipy.stats.wasserstein_distance(
		np.arange(len(vec_1)), np.arange(len(vec_1)), vec_1, vec_2
	)
	return np.exp(-(w_dist * w_dist) / (2 * sigma * sigma))


def gaussian_total_variation_kernel(vec_1, vec_2, sigma=1):
	"""
	Computes the Gaussian kernel function on two vectors, where the similarity
	metric within the Gaussian is the total variation between the two vectors.
	Arguments:
		`vec_1`: a NumPy array of values
		`vec_2`: a NumPy array of values; the underlying vector space must be
			the same as `vec_1`
		`sigma`: standard deviation for the Gaussian kernel
	Returns a scalar similarity value between 0 and 1.
	"""
	tv_dist = np.sum(np.abs(vec_1 - vec_2)) / 2
	return np.exp(-(tv_dist * tv_dist) / (2 * sigma * sigma))


def compute_inner_prod_feature_mean(dist_1, dist_2, kernel_type, **kwargs):
	"""
	Given two empirical distributions of vectors, computes the inner product of
	the feature means using the specified kernel. This is equivalent to the
	expected/average kernel function on all pairs of vectors between the two
	distributions.
	Arguments:
		`dist_1`: an M x D NumPy array of M vectors, each of size D; all vectors
			must share the same underlying vector space (or probability space if
			the vectors represent a probability distribution) with each other
			and with `dist_2`
		`dist_2`: an M x D NumPy array of M vectors, each of size D; all vectors
			must share the same underlying vector space (or probability space if
			the vectors represent a probability distribution) with each other
			and with `dist_1`
		`kernel_type`: type of kernel to apply for computing the kernelized
			inner product; can be "gaussian", "gaussian_wasserstein", or
			"gaussian_total_variation"
		`kwargs`: extra keyword arguments to be passed to the kernel function
	Returns a scalar which is the average kernelized inner product between all
	pairs of vectors across the two distributions.
	"""
	if kernel_type == "gaussian":
		kernel_func = gaussian_kernel
	elif kernel_type == "gaussian_wasserstein":
		kernel_func = gaussian_wasserstein_kernel
	elif kernel_type == "gaussian_total_variation":
		kernel_func = gaussian_total_variation_kernel
	else:
		raise ValueError("Unknown kernel type: %s" % kernel_type)
	
	inner_prods = []
	for vec_1 in dist_1:
		for vec_2 in dist_2:
			inner_prods.append(kernel_func(vec_1, vec_2, **kwargs))
	
	return np.mean(inner_prods)


def compute_maximum_mean_discrepancy(
	dist_1, dist_2, kernel_type, normalize=True, **kwargs
):
	"""
	Given two empirical distributions of vectors, computes the maximum mean
	discrepancy (MMD) between the two distributions.
	Arguments:
		`dist_1`: an M x D NumPy array of M vectors, each of size D; all vectors
			must share the same underlying vector space (or probability space if
			the vectors represent a probability distribution) with each other
			and with `dist_2`
		`dist_2`: an M x D NumPy array of M vectors, each of size D; all vectors
			must share the same underlying vector space (or probability space if
			the vectors represent a probability distribution) with each other
			and with `dist_1`
		`kernel_type`: type of kernel to apply for computing the kernelized
			inner product; can be "gaussian", "gaussian_wasserstein", or
			"gaussian_total_variation"
		`normalize`: if True, normalize each D-vector to sum to 1
		`kwargs`: extra keyword arguments to be passed to the kernel function
	Returns the scalar MMD value.
	"""
	if normalize:
		dist_1 = dist_1 / np.sum(dist_1, axis=1, keepdims=True)
		dist_2 = dist_2 / np.sum(dist_2, axis=1, keepdims=True)
		

	term_1 = compute_inner_prod_feature_mean(
		dist_1, dist_1, kernel_type, **kwargs
	)
	term_2 = compute_inner_prod_feature_mean(
		dist_2, dist_2, kernel_type, **kwargs
	)
	term_3 = compute_inner_prod_feature_mean(
		dist_1, dist_2, kernel_type, **kwargs
	)
	return np.sqrt(term_1 + term_2 - (2 * term_3))

import copy
def get_graph_statistics(graphs: list[nx.Graph]) -> dict[str, np.ndarray]:
	degrees = graph_metrics.get_degrees(graphs)
	cluster_coefs = graph_metrics.get_clustering_coefficients(graphs)
	# modifs = graph_metrics.get_modifs(graphs)
	spectra = graph_metrics.get_spectra(graphs)
	try:
		graphs_orbit = copy.deepcopy(graphs)
		for G in graphs_orbit:
			G.remove_edges_from(nx.selfloop_edges(G))
		orbit_counts = graph_metrics.get_orbit_counts(graphs_orbit)
		orbit_counts = np.stack([
			np.mean(counts, axis=0) for counts in orbit_counts
		])
	except Exception as e:
		orbit_counts = np.nan

	return {
		"degree": degrees,
		"cluster": cluster_coefs,
		# "modifs": modifs
		"spectra": spectra,
		"orbit": orbit_counts,
	}


def evaluate_sampled_graphs(sampled_graphs: list[nx.Graph], real_graphs: list[nx.Graph], model_name: str) -> dict[str, float]:
	print(f"Processing sampled graphs of {model_name}")
	sampled_graphs = [nx.convert_node_labels_to_integers(nx.Graph(graph)) for graph in sampled_graphs]
	real_graphs = [nx.convert_node_labels_to_integers(nx.Graph(graph)) for graph in real_graphs]

	sampled_stats = get_graph_statistics(sampled_graphs)
	actual_stats = get_graph_statistics(real_graphs)
	out = {}
	kernel_type = "gaussian_total_variation"
	# degree
	max_deg = max(
		max(map(max, actual_stats["degree"])),
		max(map(max, sampled_stats["degree"])),
	)
	degree_bins = np.arange(0.5, max_deg + 0.5)
	actual_degree_hist = make_histograms(actual_stats["degree"], bin_array=degree_bins)
	sampled_degree_hist = make_histograms(sampled_stats["degree"], bin_array=degree_bins)
	out["degree_mmd"] = compute_maximum_mean_discrepancy(
		actual_degree_hist, sampled_degree_hist, kernel_type=kernel_type, sigma=1,
	)
	# cluster
	actual_cluster_hist = make_histograms(actual_stats["cluster"], num_bins=100)
	sampled_cluster_hist = make_histograms(sampled_stats["cluster"], num_bins=100)
	out["cluster_mmd"] = compute_maximum_mean_discrepancy(
		actual_cluster_hist, sampled_cluster_hist, kernel_type=kernel_type, sigma=0.1,
	)
	# cluster
	bin_array = np.linspace(1e-5, 2, 201)
	actual_spectra_hist = make_histograms(actual_stats["spectra"], bin_array=bin_array)
	sampled_spectra_hist = make_histograms(sampled_stats["spectra"], bin_array=bin_array)
	out["spectra_mmd"] = compute_maximum_mean_discrepancy(
		actual_spectra_hist, sampled_spectra_hist, kernel_type=kernel_type, sigma=1,
	)
	
	# specture
	bin_array = np.linspace(1e-5, 2, 201)
	actual_spectra_hist = make_histograms(actual_stats["spectra"], bin_array=bin_array)
	sampled_spectra_hist = make_histograms(sampled_stats["spectra"], bin_array=bin_array)
	out["spectra_mmd"] = compute_maximum_mean_discrepancy(
		actual_spectra_hist, sampled_spectra_hist, kernel_type=kernel_type, sigma=1,
	)

	# orbit

	out["orbit_mmd"] = compute_maximum_mean_discrepancy(
		actual_stats["orbit"], sampled_stats["orbit"], kernel_type=kernel_type,
		normalize=False, sigma=30,
	)

	# power law
	xmin = 3
	print(f"current xmin is {xmin}")
	actual_stats.update(cal_extra_metrics(real_graphs,xmin))
	sampled_stats.update(cal_extra_metrics(sampled_graphs,xmin))

	
	for k in actual_stats.keys():
		try:
			out[k] = np.mean(sampled_stats[k])
			out[f"{k}_std"] = np.std(sampled_stats[k])
		except Exception as e:
			filtered_l = list(filter(lambda x:not np.isnan(x), sampled_stats["alpha"]))
			out[k] = np.mean(filtered_l)
			out[f"{k}_std"] = np.std(filtered_l)
	
	# get valid power law ratio
	valid =0
	valid_range = [2, 3]
	for idx in range(len(sampled_graphs)):
		D = sampled_stats["D"][idx]
		alpha = sampled_stats["alpha"][idx]
		if D <0.1 and alpha>=valid_range[0] and alpha <=valid_range[1]:
			valid+=1
	out["valid"] = valid/len(sampled_graphs)
	# try:
	# 	# orbit
	# 	out["orbit_mmd"] = compute_maximum_mean_discrepancy(
	# 		actual_stats["orbit"], sampled_stats["orbit"], kernel_type=kernel_type,
	# 		normalize=False, sigma=30,
	# 	)
	# except Exception as e:
	# 	out["orbit_mmd"] = np.nan
	
	### new metric
	print(f"--- Running Advanced Community Evaluation for {model_name} ---")
	alpha = 0.05  # Significance level for KS-tests
	sampled_community_props = get_all_community_properties(sampled_graphs)
	actual_community_props = get_all_community_properties(real_graphs)

	community_metrics = ['sizes', 'densities', 'conductances', 'internal_clustering', 'expansions']

	for prob in community_metrics:
		key_prefix = f"community_{prob}"
		actual_data = actual_community_props[prob]
		sampled_data = sampled_community_props[prob]

		# if actual_data and sampled_data:
		# 	ks_stat, p_val = scipy.stats.ks_2samp(actual_data, sampled_data)
		# 	out[f"{key_prefix}_ks_stat"] = ks_stat
		# 	out[f"{key_prefix}_p_value"] = p_val
		# 	out[f"{key_prefix}_ks_significant"] = (p_val < alpha)
		# else:
		# 	out[f"{key_prefix}_ks_stat"] = np.nan
		# 	out[f"{key_prefix}_p_value"] = np.nan
		# 	out[f"{key_prefix}_ks_significant"] = np.nan
		
		bin_strategies = {
			'sizes': {'type': 'dynamic', 'bins': 20},
			'densities': {'type': 'fixed', 'range': [0, 1], 'bins': 50},
			'conductances': {'type': 'fixed', 'range': [0, 1], 'bins': 50},
			'internal_clustering': {'type': 'fixed', 'range': [0, 1], 'bins': 50},
			'expansions': {'type': 'dynamic', 'bins': 20}
		}

		for prob, strategy in bin_strategies.items():
			key_prefix = f"community_{prob}"
			all_values = actual_community_props[prob] + sampled_community_props[prob]

			if not all_values:
				print(f"No values found for {prob} of {model_name}, skipping MMD computation")
				out[f"{key_prefix}_mmd"] = np.nan
				continue
			
			if strategy['type'] == 'fixed':
				bin_array = np.linspace(strategy['range'][0], strategy['range'][1], strategy['bins'] + 1)
			else:
				min_val, max_val = min(all_values), max(all_values)
				num_bins = strategy['bins'] if max_val > min_val else 1
				bin_array = np.linspace(min_val, max_val, num_bins + 1)

			actual_hists = get_community_property_histograms(real_graphs, prob, bin_array)
			sampled_hists = get_community_property_histograms(sampled_graphs, prob, bin_array)

			out[f"{key_prefix}_mmd"] = compute_maximum_mean_discrepancy(
				actual_hists, sampled_hists, kernel_type=kernel_type, sigma=1
			)

	return out

	# if sampled_community_props["sizes"] and actual_community_props["sizes"]:
	# 	ks_stat, p_val = scipy.stats.ks_2samp(sampled_community_props["sizes"], actual_community_props["sizes"])
	# 	out["community_size_ks_stat"] = ks_stat
	# 	out["community_size_p_value"] = p_val
	# else:
	# 	out["community_size_ks_stat"] = np.nan
	# 	out["community_size_p_value"] = np.nan

	# # b. 社区密度
	# if sampled_community_props["densities"] and actual_community_props["densities"]:
	# 	ks_stat, p_val = scipy.stats.ks_2samp(sampled_community_props["densities"], actual_community_props["densities"])
	# 	out["community_density_ks_stat"] = ks_stat
	# 	out["community_density_p_value"] = p_val
	# else:
	# 	out["community_density_ks_stat"] = np.nan
	# 	out["community_density_p_value"] = np.nan

	# # c. 社区传导性
	# if sampled_community_props["conductances"] and actual_community_props["conductances"]:
	# 	ks_stat, p_val = scipy.stats.ks_2samp(sampled_community_props["conductances"], actual_community_props["conductances"])
	# 	out["community_conductance_ks_stat"] = ks_stat
	# 	out["community_conductance_p_value"] = p_val
	# else:
	# 	out["community_conductance_ks_stat"] = np.nan
	# 	out["community_conductance_p_value"] = np.nan
	
	# all_sizes = sampled_community_props["sizes"] + actual_community_props["sizes"]
	# if all_sizes:
	# 	min_size, max_size = min(all_sizes), max(all_sizes)
	# 	num_bins = 20 if max_size > min_size else 1
	# 	bin_array = np.linspace(min_size, max_size, num_bins + 1)

	# 	sampled_cs_hists = get_community_size_histograms(sampled_graphs, bin_array)
	# 	actual_cs_hists = get_community_size_histograms(real_graphs, bin_array)

	# 	out["community_size_mmd"] = compute_maximum_mean_discrepancy(
	# 		actual_cs_hists, sampled_cs_hists, kernel_type=kernel_type, sigma=1
	# 	)
	# else:
	# 	out["community_size_mmd"] = np.nan

	return out

def cal_extra_metrics(graphs:list,xmin=None):
	ms = []
	alphas = []
	Ds =[]
	
	for graph in graphs:
		degree_list = [graph.degree(n) for n in graph.nodes()]
		results = powerlaw.Fit(list(degree_list), 
								discrete=True,
									# fit_method="KS",
									xmin=xmin
									)
		try:
			alpha = results.power_law.alpha
			# xmin = results.power_law.xmin
			sigma = results.power_law.sigma
			D = results.power_law.D
			
		except:
			pass
		try:
			partition = community_louvain.best_partition(graph)
			# 计算这种划分的modularity
			modularity = community_louvain.modularity(partition, graph)
		except:
			modularity =0
		ms.append(modularity)
		alphas.append(alpha)
		Ds.append(D)
	return {
		"modularity":ms,
		"D":Ds,
		"alpha":alphas,
		"nodes": [len(g.nodes()) for g in graphs]
	}

def get_all_community_properties(graphs: list[nx.Graph]) -> dict:
	"""
	对一组图，计算其中所有社区的规模、密度和传导性。
	"""
	props = {
        'sizes': [], 
        'densities': [], 
        'conductances': [],
        'internal_clustering': [],
        'expansions': []
    }
	for i, G in enumerate(graphs):
		if G.number_of_nodes() == 0:
			print(f"Graph {i} has no nodes, skipping community detection.")
			continue
		if G.number_of_edges() == 0:
			print(f"Graph {i} has no edges, skipping community detection.")
			continue
		try:
			partition = community_louvain.best_partition(G)
			communities = {}
			for node, comm_id in partition.items():
				communities.setdefault(comm_id, []).append(node)
			for nodes in communities.values():
				if len(nodes) <= 1:
					continue
				subgraph = G.subgraph(nodes)
				props['sizes'].append(len(nodes))
				props['densities'].append(nx.density(subgraph))
				try:
					props['conductances'].append(nx.algorithms.cuts.conductance(G, nodes))
				except nx.NetworkXError:
					props['conductances'].append(0.0)
				props['internal_clustering'].append(nx.average_clustering(subgraph))

				boundary_edges = list(nx.edge_boundary(G, nodes))
				props['expansions'].append(len(boundary_edges) / len(nodes))
		
		except Exception as e:
			print(f"Warning: Community detection failed for graph at index {i}. Reason: {e}")
			continue
	return props

	# all_sizes = []
	# all_densities = []
	# all_conductances = []

	# for i, G in enumerate(graphs):
	# 	if G.number_of_nodes() == 0:
	# 		print(f"Graph {i} has no nodes, skipping community detection.")
	# 		continue
	# 	if G.number_of_edges() == 0:
	# 		print(f"Graph {i} has no edges, skipping community detection.")
	# 		continue
	# 	try:
	# 		partition = community_louvain.best_partition(G)
			
	# 		# 创建从社区ID到节点列表的映射
	# 		communities = {}
	# 		for node, comm_id in partition.items():
	# 			communities.setdefault(comm_id, []).append(node)
	# 			# if comm_id not in communities:
	# 			# 	communities[comm_id] = []
	# 			# communities[comm_id].append(node)
	# 		community_sizes = [len(c) for c in communities.values()]
	# 		if not any(s > 1 for s in community_sizes):
	# 			print(f"Info: Graph at index {i} only has trivial communities (all size 1).")
	# 			all_sizes.extend(community_sizes)
	# 			continue
	# 		# 遍历每个社区
	# 		for comm_id, nodes in communities.items():
	# 			community_size = len(nodes)
	# 			if community_size > 1:
	# 				all_sizes.append(community_size)
					
	# 				# 密度计算
	# 				sub = G.subgraph(nodes)
	# 				all_densities.append(nx.density(sub))

	# 				# 传导性计算
	# 				# 需要整个图G和社区的节点列表
	# 				conductance = nx.algorithms.cuts.conductance(G, nodes)
	# 				all_conductances.append(conductance)
	# 	except Exception as e:
	# 		# 如果社区检测失败，则跳过这个图
	# 		print(f"Warning: Community detection failed for graph at index {i}. Reason: {e}")
	# 		continue

	# return {
	# 	"sizes": all_sizes,
	# 	"densities": all_densities,
	# 	"conductances": all_conductances,
	# }

def get_community_property_histograms(
		graphs: list[nx.Graph],
		prob_name: str,
		bin_array: np.ndarray,
		) -> np.ndarray:
	"""
    为每个图计算其指定社区属性的直方图。
    """
	all_hists = []
	for G in graphs:
		graph_probs = []
		if G.number_of_nodes() > 0:
			try:
				partition = community_louvain.best_partition(G)
				communities = {}
				for node, comm_id in partition.items():
					communities.setdefault(comm_id, []).append(node)
				for nodes in communities.values():
					if len(nodes) <= 1:
						continue
					subgraph = G.subgraph(nodes)
					prob_val = 0
					if prob_name == "sizes":
						prob_val = len(nodes)
					elif prob_name == "densities":
						prob_val = nx.density(subgraph)
					elif prob_name == "conductances":
						try:
							prob_val = nx.algorithms.cuts.conductance(G, nodes)
						except nx.NetworkXError as e:
							print(f"Error calculating conductance for graph {G}: {e}")
							prob_val = 0.0
					elif prob_name == "internal_clustering":
						prob_val = nx.average_clustering(subgraph)
					elif prob_name == "expansions":
						boundary_edges = list(nx.edge_boundary(G, nodes))
						prob_val = len(boundary_edges) / len(nodes)
					graph_probs.append(prob_val)
			except Exception as e:
				print(f"Warning: Community detection failed for graph {G}. Reason: {e}")
				continue
		hist, _ = np.histogram(graph_probs, bins=bin_array)
		all_hists.append(hist)
	return np.array(all_hists)





def get_community_size_histograms(graphs: list[nx.Graph], bin_array: np.ndarray) -> np.ndarray:
	"""
	对一组图，为每个图计算其社区规模的直方图。
	"""
	all_hists = []
	for G in graphs:
		if G.number_of_nodes() == 0:
			all_hists.append(np.zeros(len(bin_array) - 1))
			continue
		
		try:
			partition = community_louvain.best_partition(G)
			communities = {}
			for node, comm_id in partition.items():
				communities.setdefault(comm_id, []).append(node)
			community_sizes = [len(c) for c in communities.values()]
			# community_sizes = [len(c) for c in community_louvain.community_louvain.convert_partition_format(partition).values()]
			hist, _ = np.histogram(community_sizes, bins=bin_array)
			all_hists.append(hist)
		except Exception:
			# 如果失败，添加一个零向量
			all_hists.append(np.zeros(len(bin_array) - 1))
			
	return np.array(all_hists)