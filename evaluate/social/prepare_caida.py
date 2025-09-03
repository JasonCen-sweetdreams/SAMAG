import networkx as nx
import pandas as pd
import os
import sys
from tqdm import tqdm
import scipy.io

# 确保你的项目根目录已在PYTHONPATH中
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# **修改点：直接导入新的、独立的 `calculate_effective_diameter` 函数**
from evaluate.social.graph_metrics import calculate_effective_diameter

def process_caida_snapshots_mtx(data_dir):
    """
    处理CAIDA数据集快照（.mtx格式），只计算每个快照的有效直径，并保存结果。
    """
    names_file_path = os.path.join(data_dir, "as-caida_Gname.txt")
    if not os.path.exists(names_file_path):
        print(f"错误：未找到图名称文件: {names_file_path}")
        return
        
    with open(names_file_path, 'r') as f:
        snapshot_names = [line.strip() for line in f if line.strip()]

    mtx_files = sorted([f for f in os.listdir(data_dir) if f.startswith('as-caida_G_') and f.endswith('.mtx')])
    
    metrics_list = []
    
    print(f"开始处理CAIDA数据集，共找到 {len(mtx_files)} 个快照文件。")
    
    for i in tqdm(range(len(mtx_files)), desc="处理CAIDA快照"):
        file_name = mtx_files[i]
        file_path = os.path.join(data_dir, file_name)
        
        try:
            sparse_matrix = scipy.io.mmread(file_path)
            G = nx.from_scipy_sparse_array(sparse_matrix, create_using=nx.DiGraph)
        except Exception as e:
            print(f"错误：无法读取文件 {file_name}，跳过。原因：{e}")
            continue
        
        # **修改点：直接、清晰地调用函数计算 De**
        effective_diameter = calculate_effective_diameter(G)
        
        metrics_list.append({
            'date': snapshot_names[i],
            'De': effective_diameter
        })

    # 保存结果
    df = pd.DataFrame(metrics_list)
    df.index = [f"{date}_caida" for date in df['date'].values]
    df.drop('date', axis=1, inplace=True)
    
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                            'configs', 'mid_hub', 'evaluate')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "caida_matrix.csv")
    df.to_csv(save_path)
    print(f"CAIDA数据处理完成，已保存至：{save_path}")
    
    return save_path

if __name__ == '__main__':
    caida_data_dir = '/home/users/wangzh/cenjch3/GraphAgent_writing/evaluate/social/as-caida'
    process_caida_snapshots_mtx(caida_data_dir)