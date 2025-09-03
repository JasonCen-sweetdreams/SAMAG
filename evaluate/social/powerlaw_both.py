import pickle
from matplotlib import ticker
import networkx as nx
import numpy as np
from scipy.spatial.distance import jensenshannon
from networkx.algorithms.similarity import graph_edit_distance
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
import powerlaw
import sys
import os
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)
from evaluate.social import graph_metrics
from evaluate.visualize import plot_utils

import pandas as pd
import json
from PIL import Image
from Emulate.utils.io import readinfo, writeinfo
from evaluate.visualize.social import (plot_degree_figures, 
                                       plot_indegree_outdegree,
                                       plot_shrinking_diameter,
                                       plot_relative_size,
                                       plot_mean_degree,
                                       plot_densification_power_law,
                                       plot_friend_degree,
                                       plot_outdegree_cc,
                                       plot_avg_path,
                                       plot_gcc_proportion)
import copy
from evaluate.social.gcc_diameter import calculate_lcc_proportion

from evaluate.matrix import calculate_directed_graph_matrix
from datetime import datetime, date
from tqdm import tqdm


# ==============================================================================
# 步骤2: 定义所有需要的函数
# ==============================================================================

# --- Part A: Citation Network 相关函数 ---

def build_citation_graph(article_meta_data:dict = {}):
    """根据文章元数据构建一个有向图"""
    DG = nx.DiGraph()
    map_index = {
        title:str(idx) for idx,title in enumerate(article_meta_data.keys())
    }
    
    for title in article_meta_data.keys():
        cited_idx = map_index.get(title)
        time = article_meta_data[title]["time"]
        DG.add_node(cited_idx, title=title, time=time, topic=article_meta_data[title]["topic"])
    
    for title, article_info in article_meta_data.items():
        cited_articles = article_info.get("cited_articles",[])
        title_idx = map_index.get(title)
        
        edges = []
        for cite_title in cited_articles:
            cited_idx = map_index.get(cite_title)
            if cited_idx is not None:
                edges.append((cited_idx,title_idx))          
        DG.add_edges_from(edges)
        
    return DG

# --- Part B: Action Network 相关函数 ---

def build_nx_action_graph(action_logs:list, timestamp:date, nodes):
    DG = nx.DiGraph()
    DG.add_nodes_from(nodes)
    action_logs_filtered = list(filter(
        lambda x: datetime.strptime(x[3], "%Y-%m-%d").date() <= timestamp,
        action_logs
    ))
    for action_one in action_logs_filtered:
        act_id = action_one[0]
        own_id = action_one[1]
        DG.add_edge(act_id, own_id, action_type=action_one[2])
    return DG

def build_nx_graph(social_member_data_path, action_logs, pos, date_str, date_map, transitive_nodes:dict = {}):
    social_member_data = pd.read_csv(social_member_data_path)
    DG = nx.DiGraph()
    G = nx.Graph()
    delete_nodes = transitive_nodes.get("delete_ids", [])
    
    for _, row in social_member_data.iterrows():
        user_index = row['user_index']
        if user_index not in date_map:
            date_map[user_index] = date_str
        if user_index in delete_nodes:
            continue
        DG.add_node(user_index, date=date_map[user_index])
        G.add_node(user_index, date=date_map[user_index])

    for _, row in social_member_data.iterrows():
        user_index = row['user_index']
        if user_index not in pos:
            pos[user_index] = None
        follow_ids = json.loads(row['follow'])
        friend_ids = json.loads(row['friend'])
        for follow_id in follow_ids:
            if user_index in delete_nodes or follow_id in delete_nodes:
                continue
            if user_index != follow_id:
                DG.add_edge(user_index, follow_id)
        for friend_id in friend_ids:
            if user_index in delete_nodes or friend_id in delete_nodes:
                continue
            if user_index != friend_id:
                DG.add_edge(user_index, friend_id)
                DG.add_edge(friend_id, user_index)
                G.add_edge(user_index, friend_id)
    
    timestamp_date = datetime.strptime(date_str, "%Y%m%d").date()
    nodes = list(DG.nodes)
    action_graph = build_nx_action_graph(action_logs, timestamp_date, nodes)
    return DG, G, action_graph, pos, date_map

def build_sn_graphs(data_root):
    sn_root = os.path.join(data_root, "social_network")
    transitive_nodes_path = os.path.join(data_root, "transitive_agent_log.json")
    action_logs_path = os.path.join(data_root, "action_logs.json")
    
    if not os.path.exists(sn_root) or not os.path.exists(action_logs_path):
        print(f"错误: 找不到 Action Network 的数据路径 {sn_root} 或 {action_logs_path}")
        return
        
    transitive_nodes_log = readinfo(transitive_nodes_path) if os.path.exists(transitive_nodes_path) else []
    csv_files = sorted([f for f in os.listdir(sn_root) if f.endswith('.csv')])
    
    if not transitive_nodes_log:
        transitive_nodes_log = [{"delete_ids": []} for _ in range(len(csv_files))]

    positions = {}
    date_map = {}
    action_logs = readinfo(action_logs_path)
    transitive_nodes_all = {"delete_ids": [], "add_ids": []}
    
    for i, csv_file in enumerate(csv_files):
        transitive_nodes = transitive_nodes_log[i] if i < len(transitive_nodes_log) else {"delete_ids": []}
        date_str = csv_file.split("_")[3][:8]
        file_path = os.path.join(sn_root, csv_file)
        for k in transitive_nodes.keys():
            if k in transitive_nodes_all:
                transitive_nodes_all[k].extend(transitive_nodes[k])
        
        DG, G, action_graph, positions, date_map = build_nx_graph(
            file_path, action_logs, positions, date_str, date_map, transitive_nodes_all
        )
        yield DG, G, action_graph, date_str, positions

# --- Part C: 绘图辅助函数 ---

def plot_power_law_on_ax(ax: plt.Axes, in_degrees: list, title: str):
    """在给定的 matplotlib Axes 对象上计算并绘制幂律分布图。"""
    if not in_degrees:
        ax.text(0.5, 0.5, "No valid data", ha='center', va='center')
        ax.set_title(title, fontsize=40) # ◄◄◄【修改】字体大小从 20 -> 12
        print(f"警告: '{title}' 中没有入度大于0的节点。")
        return

    fit = powerlaw.Fit(in_degrees, discrete=True)
    alpha = fit.power_law.alpha
    kmin = fit.power_law.xmin
    Dk = fit.power_law.D
    print(f"为 '{title}' 拟合结果: alpha = {alpha:.2f}, k_min = {kmin}, D_k = {Dk:.2f}")

    fit.plot_pdf(ax=ax, color='b', marker='d', linestyle='None', markersize=3, label='Linearly-binned Degree') # ◄◄◄【修改】markersize
    fit.power_law.plot_pdf(ax=ax, color='r', linestyle='-', linewidth=1.5, label='Power Law Fit') # ◄◄◄【修改】linewidth
    ax.set_title(title, fontsize=45) # ◄◄◄【修改】字体大小从 22 -> 12
    ax.tick_params(axis='both', which='major', labelsize=35, direction='out', length=10, width=2)

    

    locatorX = ticker.LogLocator(numticks=4)
    locatorY = ticker.LogLocator(numticks=4)
    ax.xaxis.set_major_locator(locatorX)
    ax.yaxis.set_major_locator(locatorY)
    
    # Optional but recommended: Format labels as scientific notation (e.g., 10¹, 10²)
    formatter = ticker.LogFormatterSciNotation(base=10)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)

    text_str = (f'$\\alpha = {alpha:.2f}$\n'
                f'$D_k = {Dk:.2f}$\n'
                f'$k_{{min}} = {kmin}$')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    ax.text(0.7, 0.95, text_str, transform=ax.transAxes, fontsize=30,
            verticalalignment='top', horizontalalignment='left', bbox=props)


# ==============================================================================
# 步骤3: 主执行逻辑
# ==============================================================================
if __name__ == "__main__":
    # mpl.rcParams.update({
    #     "font.family": "serif",  # 使用衬线字体
    #     "font.serif": ["Times New Roman"],  # 首选Times New Roman
    #     "text.usetex": True,     # 启用LaTeX渲染所有文本
    #     "pgf.rcfonts": False,    # 不替换字体，使用LaTeX默认
    #     "axes.unicode_minus": False, # 正确显示负号
    # })
    from matplotlib.font_manager import FontProperties
    font_path = '/home/users/wangzh/cenjch3/GraphAgent_writing/test/Times_New_Roman/TimesNewerRoman-Regular.otf'
    if os.path.exists(font_path):
        custom_font = FontProperties(fname=font_path)
        mpl.rcParams['font.family'] = custom_font.get_name()
        # mpl.rcParams['font.serif'] = [custom_font.get_name()]
    # --- Part 1: 加载 Citation Network ---
    cache_filepath = "plotting_data_cache.pkl"
    if os.path.exists(cache_filepath):
        print(f"--- 发现缓存文件，正在从 '{cache_filepath}' 加载绘图数据 ---")
        with open(cache_filepath, 'rb') as f:
            plotting_data = pickle.load(f)
        citation_degrees = plotting_data['citation_degrees']
        action_degrees = plotting_data['action_degrees']
    else:
        print("--- 正在加载 Citation Network ---")
        citation_graph = None
        try:
            citation_data_path = "Emulate/tasks/citeseer/configs/big/data/article_meta_info.pt"
            article_meta_data = torch.load(citation_data_path)
            citation_graph = build_citation_graph(article_meta_data)
            print(f"Citation Network 加载成功: {len(citation_graph.nodes())} 个节点, {len(citation_graph.edges())} 条边。")
        except FileNotFoundError:
            print(f"警告：Citation Network 数据文件未找到: {citation_data_path}")
        except Exception as e:
            print(f"加载 Citation Network 时发生错误: {e}")

        # --- Part 2: 加载 Action Network ---
        print("\n--- 正在加载 Action Network ---")
        action_graph = None
        graph_lists = []
        try:
            # 根据你的脚本，设置 task 和 config
            task = "tweets"
            config = "big" # 或者你使用的其他 config, e.g., "big"
            action_data_root = f"Emulate/tasks/{task}/configs/{config}/data/generated/data"
            
            # 从 build_sn_graphs 生成器中获取所有图快照
            graph_generator = build_sn_graphs(action_data_root)
            count = 0
            for DG, G, ag, date_str, positions in graph_generator:
                count += 1
                if count == 5:
                    graph_lists.append((DG, G, ag, date_str, positions))
                    break
            
            if graph_lists:
                # 我们需要的是最后一个快照中的 action_graph
                action_graph = graph_lists[-1][2]
                print(f"Action Network 加载成功 (来自最后一个快照): {len(action_graph.nodes())} 个节点, {len(action_graph.edges())} 条边。")
            else:
                print("警告: 未能从指定路径生成任何 Action Network 快照。")

        except Exception as e:
            print(f"加载 Action Network 时发生错误: {e}")


        print("\n--- 正在从图中提取度数数据 ---")
        citation_degrees = [d for n, d in citation_graph.in_degree() if d > 0]
        action_degrees = [d for n, d in action_graph.in_degree() if d > 0]

        # Part 4: 将提取的数据保存到缓存文件 (这是新增部分)
        plotting_data = {
            'citation_degrees': citation_degrees,
            'action_degrees': action_degrees
        }
        with open(cache_filepath, 'wb') as f:
            pickle.dump(plotting_data, f)
        print(f"--- 绘图数据已成功缓存到 '{cache_filepath}' ---")

    # --- Part 3: 合并绘图 ---
    print("\n--- 正在生成合并图像 ---")
    if citation_degrees and action_degrees:
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))

        print("正在绘制 Citation Network...")
        plot_power_law_on_ax(axes[0], citation_degrees, "Citation Network")

        print("正在绘制 Action Network...")
        plot_power_law_on_ax(axes[1], action_degrees, "Action Network")

        fig.supxlabel('Degree $k$', fontsize=35, y=0.1)
        fig.supylabel('$P_k$', fontsize=35, rotation=0, x=0.01, y=0.55)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0001), 
                   ncol=2, fontsize=30, fancybox=True, shadow=True)

        fig.tight_layout()
        fig.subplots_adjust(bottom=0.2, top=0.90)

        output_filename = "combined_power_law_plot_final.pdf"
        plt.savefig(output_filename)
        print(f"\n合并后的图像已成功保存到: {output_filename}")
    else:
        print("\n错误：由于一个或两个图未能成功加载，无法生成合并图像。")