import networkx as nx
import numpy as np
import random
from tqdm import tqdm

def calculate_effective_diameter(G, percentile=0.9):
    """
    Calculate the effective diameter of the graph G.

    :param G: NetworkX graph
    :param percentage: percentage of node pairs to consider (default is 0.9 for 90th percentile)
    :return: effective diameter of the graph
    """
    # 计算所有节点对之间的最短路径长度
    if nx.is_directed(G):
        assert isinstance(G, nx.DiGraph)
        G = G.to_undirected()
    lengths = dict(nx.all_pairs_shortest_path_length(G))
    
    # 将所有路径长度放入一个列表中
    all_lengths = []
    for source, targets in lengths.items():
        all_lengths.extend(targets.values())
        
    # 过滤掉长度为0的路径（这些是节点自己到自己的路径）
    all_lengths = [length for length in all_lengths if length > 0]
    
    # Total number of pairs
    total_pairs = len(all_lengths)
    
    # Find the smallest integer d such that g(d) >= percentile
    cumulative_distribution = np.cumsum(np.bincount(all_lengths, minlength=max(all_lengths) + 1)) / total_pairs
    d = np.searchsorted(cumulative_distribution, percentile, side='right')
    
    # Interpolate between d and d+1
    if d == 0:
        effective_diameter = 0
    else:
        g_d = cumulative_distribution[d - 1] if d > 0 else 0
        g_d_plus_1 = cumulative_distribution[d]
        if g_d_plus_1 == g_d:
            effective_diameter = d
        else:
            effective_diameter = d - 1 + (percentile - g_d) / (g_d_plus_1 - g_d)
    
    return effective_diameter

# def calculate_effective_diameter(G: nx.Graph, sample_size: int = 1000, percentile: float = 0.9) -> float:
#     """
#     【推荐方法】计算图的有效直径（基于百分位数定义）。
#     内部封装了处理非连通图和随机抽样加速的逻辑。

#     Args:
#         G (nx.Graph): networkx图对象。
#         sample_size (int): 用于计算的源节点采样数量。
#         percentile (float): 百分位数，默认为0.9 (90%)。

#     Returns:
#         float: 计算出的有效直径。
#     """
#     subgraph = G
#     if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
#         return np.nan
    
#     # 1. 确定计算子图（处理非连通情况）
#     is_connected = False
#     if G.is_directed():
#         largest_scc_nodes = max(nx.strongly_connected_components(G), key=len, default=set())
#         if len(largest_scc_nodes) > 1:
#             subgraph = G.subgraph(largest_scc_nodes)
#             is_connected = True
#     else:
#         largest_cc_nodes = max(nx.connected_components(G), key=len, default=set())
#         if len(largest_cc_nodes) > 1:
#             subgraph = G.subgraph(largest_cc_nodes)
#             is_connected = True

#     if not is_connected:
#         return np.nan

#     # 2. 随机抽取源节点用于计算
#     num_nodes_in_subgraph = subgraph.number_of_nodes()
#     actual_sample_size = min(sample_size, num_nodes_in_subgraph)
#     source_nodes = random.sample(list(subgraph.nodes()), actual_sample_size)
    
#     # 3. 从抽样节点计算最短路径
#     all_shortest_path_lengths = []
#     try:
#         for source in source_nodes:
#             lengths = nx.shortest_path_length(subgraph, source=source)
#             all_shortest_path_lengths.extend(lengths.values())
#     except nx.NetworkXNoPath:
#         print("警告：在连通分量中检测到无路径的节点对。")
#         return np.nan

#     # 4. 使用插值法计算最终的有效直径
#     if not all_shortest_path_lengths:
#         return np.nan
        
#     all_lengths = [length for length in all_shortest_path_lengths if length > 0]
#     if not all_lengths:
#         return 0.0

#     total_pairs = len(all_lengths)
#     all_lengths_int = [int(round(l)) for l in all_lengths]
#     max_len = max(all_lengths_int)
    
#     counts = np.bincount(all_lengths_int, minlength=max_len + 1)
#     cumulative_distribution = np.cumsum(counts) / total_pairs
    
#     d = np.searchsorted(cumulative_distribution, percentile, side='left')
    
#     if d == 0:
#         return 0.0
    
#     g_d_minus_1 = cumulative_distribution[d - 1]
#     g_d = cumulative_distribution[d]
    
#     if g_d == g_d_minus_1:
#         return float(d)
#     else:
#         effective_diameter = (d - 1) + (percentile - g_d_minus_1) / (g_d - g_d_minus_1)
        
#     return effective_diameter


def calculate_all_metrics(G: nx.Graph, graph_name: str, diameter_mode: str = 'approx') -> dict:
    """
    计算给定图的所有结构指标。
    """
    N = G.number_of_nodes()
    E = G.number_of_edges()
    
    if N == 0:
        return {
            'graph_name': graph_name, '|V|': 0, '|E|': 0, 'cc': 0,
            'cc_er_ratio': np.nan, 'cc_ba_ratio': np.nan,
            'r': np.nan, 'De': np.nan, 'De_exact': np.nan
        }

    cc = nx.average_clustering(G) if N > 2 else 0
    p = (2 * E) / (N * (N - 1)) if N > 1 else 0
    m = int(round(E / N)) if N > 0 else 0
    cc_er_est = p
    cc_ba_est = (m - 1) / (8 * N) * (np.log(N))**2 if N > 1 and m > 1 else 0
    ratio_er = cc / cc_er_est if cc_er_est > 0 else np.nan
    ratio_ba = cc / cc_ba_est if cc_ba_est > 0 else np.nan
    r = nx.degree_assortativity_coefficient(G) if E > 0 else 0
        
    # 直接调用新的、独立的直径计算函数
    effective_diameter = calculate_effective_diameter(G)
    
    # 如果需要，计算精确直径
    exact_diameter = np.nan
    if diameter_mode == 'both':
        print("警告：正在计算精确直径，这可能需要很长时间...")
        # (这里可以添加一个独立的精确直径计算函数调用)
        # exact_diameter = ...
        pass

    return {
        'graph_name': graph_name,
        '|V|': N,
        '|E|': E,
        'cc': cc,
        'cc_er_ratio': ratio_er,
        'cc_ba_ratio': ratio_ba,
        'r': r,
        'De': effective_diameter,
        'De_exact': exact_diameter
    }