from collections import defaultdict
import os
import torch
import networkx as nx
import json
import logging

# ==============================================================================
# 1. 配置
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_G_PATH = "/home/users/wangzh/cenjch3/GraphAgent_writing/LLMGraph/baselines/baseline_checkpoints/graph_w_text"
# BiGG.L  bigg_gen
# GRAN.L  gran_gen
# BwR.L  bwr_graphrnn
# GraphMaker.L  graphmaker_sync
# L-PPGN.L  l_ppgn
MODEL_CLASS = "l_ppgn"
DATA_CLASS = "llmcitationciteseer"
F_NAME = "pred_graphs.pt"
# GRAPH_INPUT_PATH = os.path.join(BASE_G_PATH, MODEL_CLASS, DATA_CLASS, F_NAME)

# 节点文本属性路径
# BASE_TEXT_PATH = "/home/users/wangzh/cenjch3/GraphAgent_writing/LLMGraph/baselines/baseline_checkpoints/node_text"
# 根据你的描述，文件名格式为 f"{MODEL_CLASS}_{DATA_CLASS}.json"
# ATTRIBUTES_INPUT_FILENAME = f"{MODEL_CLASS}_{DATA_CLASS}.json"
# ATTRIBUTES_INPUT_PATH = os.path.join(BASE_TEXT_PATH, ATTRIBUTES_INPUT_FILENAME)


ASSORTATIVITY_THRESHOLD = 0.3

# --- 输入文件 ---
# 这是我们上一阶段生成的带有节点属性的图文件
INPUT_GRAPH_PT_PATH = os.path.join(
    BASE_G_PATH,
    f"{MODEL_CLASS}_{DATA_CLASS}_{str(ASSORTATIVITY_THRESHOLD).replace('.', '_')}.pt"
)
# --- 输出目录 ---
# 所有转换后的文件都将保存在这个目录下
OUTPUT_DATA_DIR = f"./{MODEL_CLASS}_{DATA_CLASS}"
# 存放摘要文本文件的子目录
ARTICLES_TEXT_DIR = os.path.join(OUTPUT_DATA_DIR, "articles")

# ==============================================================================
# 2. 主转换逻辑
# ==============================================================================
def transform_graph_to_pipeline_input():
    logging.info("开始数据转换 (包含边信息)...")
    if not os.path.exists(INPUT_GRAPH_PT_PATH):
        logging.error(f"输入文件未找到: {INPUT_GRAPH_PT_PATH}")
        return

    os.makedirs(ARTICLES_TEXT_DIR, exist_ok=True)
    logging.info(f"输出目录 '{OUTPUT_DATA_DIR}' 和 '{ARTICLES_TEXT_DIR}' 已准备就绪。")

    data = torch.load(INPUT_GRAPH_PT_PATH)
    graph = data.get('final_graph')
    if not graph:
        logging.error("在 .pt 文件中未找到 key 'final_graph'。")
        return
    logging.info(f"成功加载图，包含 {graph.number_of_nodes()} 个节点和 {graph.number_of_edges()} 条边。")

    # --- 核心修改：预处理边信息 ---
    # NetworkX的边方向 (u, v) 表示 u -> v。在引文网络中，这通常意味着 u 引用了 v。
    # 我们需要构建一个 "被引用" 列表，即对于每个节点 v，有哪些节点 u 引用了它。
    # 但根据你的代码逻辑，它似乎把 (u,v) 理解为 v 引用 u。
    # 为了匹配你的 `llmgraph_dataloader.py` 逻辑，我们需要构建一个 "引用了谁" 的列表。
    # 让我们创建一个从 节点ID -> 节点标题 的映射
    node_to_title = {node: attrs.get('title') for node, attrs in graph.nodes(data=True)}
    
    # 创建一个字典来存储每个节点的引用列表
    # key是引用者(citing_node_id), value是被引用者(cited_node_id)的标题列表
    citations = defaultdict(list)
    for u, v in graph.edges():
        citing_title = node_to_title.get(u)
        cited_title = node_to_title.get(v)
        if citing_title and cited_title:
             citations[citing_title].append(cited_title)

    logging.info(f"成功从图中提取了 {len(citations)} 篇论文的引用关系。")
    # --- 核心修改结束 ---

    article_meta_data = {}
    author_data = {}
    author_to_id = {}
    author_id_counter = 0

    logging.info("正在遍历图节点并转换数据...")
    for node_id, attrs in graph.nodes(data=True):
        # ... (作者信息处理部分与之前相同) ...
        author_key = (attrs.get("author_name"), attrs.get("author_institution"))
        if author_key not in author_to_id:
            current_author_id = author_id_counter
            author_to_id[author_key] = current_author_id
            author_data[current_author_id] = {
                "name": attrs.get("author_name", "Unknown"),
                "institution": attrs.get("author_institution", ""),
                "country": attrs.get("author_country", ""),
                "time": attrs.get("publish_time"),
                "cited": 0
            }
            author_id_counter += 1
        else:
            current_author_id = author_to_id[author_key]
            
        article_text_path = os.path.join(ARTICLES_TEXT_DIR, f"{node_id}.txt")
        with open(article_text_path, 'w', encoding='utf-8') as f:
            f.write(attrs.get("paper_abstract", ""))

        original_title = attrs.get("title")
        if not original_title:
            logging.warning(f"节点 {node_id} 缺少标题，将跳过此节点。")
            continue

        final_key = original_title
        if final_key in article_meta_data:
            final_key = f"{original_title} (node_{node_id})"
            logging.warning(f"发现重复标题: '{original_title}'。为节点 {node_id} 创建唯一键: '{final_key}'")
        
        # --- 核心修改：将提取的引用关系添加到元数据中 ---
        article_meta_data[final_key] = {
            "path": article_text_path,
            "author_ids": [current_author_id],
            "time": attrs.get("publish_time"),
            "topic": attrs.get("topic", "AI"),
            "cited": 0,
            "keywords": attrs.get("paper_keywords", []),
            "cited_articles": citations.get(original_title, []) # <--- 新增的关键字段！
        }
        # --- 核心修改结束 ---

    logging.info(f"数据转换完成。共处理了 {len(article_meta_data)} 篇文章和 {len(author_data)} 位作者。")

    article_output_path = os.path.join(OUTPUT_DATA_DIR, "article_meta_info.pt")
    author_output_path = os.path.join(OUTPUT_DATA_DIR, "author.pt")
    
    torch.save(article_meta_data, article_output_path)
    logging.info(f"文章元数据已保存到: {article_output_path}")
    
    torch.save(author_data, author_output_path)
    logging.info(f"作者数据已保存到: {author_output_path}")
    
    logging.info("所有转换任务已成功完成！")

if __name__ == "__main__":
    transform_graph_to_pipeline_input()