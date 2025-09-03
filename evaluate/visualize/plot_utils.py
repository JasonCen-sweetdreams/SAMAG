import matplotlib.pyplot as plt
import pandas as pd
import os
import networkx as nx

def plot_shrinking_diameter(df_paths: dict, save_dir: str):
    """
    绘制基于 effective diameter 的 shrinking diameter 图。
    
    Args:
        df_paths (dict): 字典，键为网络标签，值为包含直径数据的CSV文件路径。
                         例如：{'Action Net.': 'path/to/action_matrix.csv'}
        save_dir (str): 保存图像的目录。
    """
    plt.figure(figsize=(8, 6))
    
    for label, file_path in df_paths.items():
        if not os.path.exists(file_path):
            print(f"警告: 未找到文件 {file_path}, 跳过绘图。")
            continue
            
        df = pd.read_csv(file_path, index_col=0)
        dates = df.index.str.split('_').str[0]
        diameters = df['De'].to_list()
        
        plt.plot(dates, diameters, marker='o', linestyle='-', label=label)

    plt.ylabel('$D_e$', fontsize=20)
    plt.xlabel('Time', fontsize=20)
    plt.legend(fontsize=14, loc="upper right")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    
    # 保存为PNG和PDF
    png_path = os.path.join(save_dir, "shrinking_diameter.png")
    pdf_path = os.path.join(save_dir, "shrinking_diameter.pdf")
    
    plt.savefig(png_path)
    plt.savefig(pdf_path)
    plt.clf()
    
    print(f"shrinking diameter图已保存至：\n {png_path}\n {pdf_path}")