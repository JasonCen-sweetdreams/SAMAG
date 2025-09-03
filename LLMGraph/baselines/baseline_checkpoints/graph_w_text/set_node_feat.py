import os
import json
import time
import torch
import networkx as nx
import random
import logging

# ==============================================================================
# 1. 配置和常量定义 (Configuration and Constants)
# ==============================================================================

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 输入路径 ---
# 图结构路径
BASE_G_PATH = "/home/users/wangzh/cenjch3/GraphAgent_writing/LLMGraph/baselines/baseline_checkpoints"
# BiGG.L  bigg_gen
# GRAN.L  gran_gen
# BwR.L  bwr_graphrnn
# GraphMaker.L  graphmaker_sync
# L-PPGN.L  l_ppgn
MODEL_CLASS = "bigg_gen"
DATA_CLASS = "llmcitationciteseer"
F_NAME = "pred_graphs.pt"
GRAPH_INPUT_PATH = os.path.join(BASE_G_PATH, MODEL_CLASS, DATA_CLASS, F_NAME)

# 节点文本属性路径
BASE_TEXT_PATH = "/home/users/wangzh/cenjch3/GraphAgent_writing/LLMGraph/baselines/baseline_checkpoints/node_text"
# 根据你的描述，文件名格式为 f"{MODEL_CLASS}_{DATA_CLASS}.json"
ATTRIBUTES_INPUT_FILENAME = f"{MODEL_CLASS}_{DATA_CLASS}.json"
ATTRIBUTES_INPUT_PATH = os.path.join(BASE_TEXT_PATH, ATTRIBUTES_INPUT_FILENAME)

# --- 输出路径 ---
# 保存最终带有属性的图
ASSORTATIVITY_THRESHOLD = 0.3
FINAL_GRAPH_OUTPUT_PATH = os.path.join(
    BASE_G_PATH,
    f"{MODEL_CLASS}_{DATA_CLASS}_{str(ASSORTATIVITY_THRESHOLD).replace('.', '_')}.pt"
)

# --- 约束条件 ---
MAX_ITERATIONS = 500 # 设置一个最大尝试次数，防止无限循环

# ==============================================================================
# 2. 主执行流程 (Main Execution Flow)
# ==============================================================================

def main():
    """
    主函数，执行节点属性分配、检验和保存的全过程。
    """
    logging.info("--- 开始为图节点分配属性 ---")

    # --- 步骤 1: 加载图结构和节点属性 ---
    logging.info(f"从 '{GRAPH_INPUT_PATH}' 加载图结构...")
    if not os.path.exists(GRAPH_INPUT_PATH):
        logging.error("图文件未找到！请检查路径。")
        return
    try:
        data = torch.load(GRAPH_INPUT_PATH)
        graph = data["pred_graphs"][0]
        num_nodes = graph.number_of_nodes()
        logging.info(f"图加载成功，包含 {num_nodes} 个节点。")
    except Exception as e:
        logging.error(f"加载图文件失败: {e}")
        return

    logging.info(f"从 '{ATTRIBUTES_INPUT_PATH}' 加载节点属性...")
    if not os.path.exists(ATTRIBUTES_INPUT_PATH):
        logging.error("节点属性JSON文件未找到！请检查路径和文件名。")
        return
    try:
        with open(ATTRIBUTES_INPUT_PATH, 'r', encoding='utf-8') as f:
            node_attributes = json.load(f)
        num_attributes = len(node_attributes)
        logging.info(f"节点属性加载成功，共 {num_attributes} 条。")
    except Exception as e:
        logging.error(f"加载JSON文件失败: {e}")
        return
        
    # 检查节点数和属性数是否匹配
    if num_nodes != num_attributes:
        logging.error(f"错误：图的节点数 ({num_nodes}) 与属性数据条数 ({num_attributes}) 不匹配！")
        return

    # --- 步骤 2 & 3: 循环进行随机分配与检验 ---
    logging.info(f"开始随机分配属性 {MODEL_CLASS}，目标同配性系数 <= {ASSORTATIVITY_THRESHOLD}")
    
    final_assignment_found = False
    for i in range(MAX_ITERATIONS):
        current_seed = int(time.time() * 1000) + i
        random.seed(current_seed)
        logging.info(f"--- 尝试次数: {i + 1}/{MAX_ITERATIONS} ---")
        
        # 核心步骤：随机打乱属性列表
        node_attributes_shuffled = node_attributes.copy()
        random.shuffle(node_attributes_shuffled)
        
        # 将打乱后的属性分配给节点
        # 创建一个从节点ID到属性字典的映射
        # graph.nodes() 返回节点视图，我们将其转换为列表以确保顺序稳定
        node_list = list(graph.nodes())
        attribute_mapping = {node_id: attr for node_id, attr in zip(node_list, node_attributes_shuffled)}
        
        # 使用 set_node_attributes 一次性为所有节点设置属性
        nx.set_node_attributes(graph, attribute_mapping)
        
        # 计算基于 'topic' 的属性同配性系数
        # networkx 会自动处理将字符串标签映射为整数进行计算
        current_assortativity = nx.attribute_assortativity_coefficient(graph, "topic")
        
        logging.info(f"当前分配方案的同配性系数为: {current_assortativity:.4f}")
        
        # 检查是否满足条件
        if current_assortativity <= ASSORTATIVITY_THRESHOLD:
            logging.info(f"成功！找到满足条件的分配方案 (系数 {current_assortativity:.4f} <= {ASSORTATIVITY_THRESHOLD})。可用于复现本次结果的随机种子(seed)是: {current_seed}")
            final_assignment_found = True
            break
        else:
            logging.warning(f"同配性系数过高，重新尝试...")

    if not final_assignment_found:
        logging.error(f"在 {MAX_ITERATIONS} 次尝试后，仍未找到满足条件的分配方案。")
        logging.error("可能是图结构本身具有极强的同配性潜力。你可以尝试放宽阈值或增加尝试次数。")
        return

    # --- 步骤 4: 验证并保存结果 ---
    logging.info("验证最终的节点属性...")
    # 随机抽查一个节点，看属性是否已正确设置
    sample_node = list(graph.nodes())[0]
    logging.info(f"节点 {sample_node} 的属性: {graph.nodes[sample_node]}")
    
    logging.info(f"将最终的图对象保存到 '{FINAL_GRAPH_OUTPUT_PATH}'...")
    try:
        # 为了与输入格式保持一致，我们同样将其保存在一个字典中再用torch.save
        torch.save({'final_graph': graph}, FINAL_GRAPH_OUTPUT_PATH)
        logging.info("最终图保存成功！")
    except Exception as e:
        logging.error(f"保存最终图失败: {e}")

if __name__ == "__main__":
    main()