import os
import numpy as np
import json
import torch
import copy
from datetime import date, datetime
from typing import Any, List, Optional, Dict
from pathlib import Path

from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders.text import TextLoader


def parse_date_str(time_val: Any) -> datetime:
    """
    一个健壮的时间解析函数，用于将多种格式的输入转换为 datetime 对象。
    """
    if isinstance(time_val, datetime):
        return time_val
    if isinstance(time_val, date):
        return datetime.combine(time_val, datetime.min.time())
    if not isinstance(time_val, str):
        # 如果是其他非字符串类型，无法解析
        return datetime.min

    # 尝试解析多种常见的字符串格式
    try:
        if len(time_val) == 10: # "YYYY-MM-DD"
            return datetime.strptime(time_val, "%Y-%m-%d")
        if len(time_val) == 7: # "YYYY-MM"
            return datetime.strptime(time_val, "%Y-%m")
        if len(time_val) > 10: # 尝试 ISO 格式，例如 "2023-04-12T10:00:00"
            return datetime.fromisoformat(time_val)
    except ValueError:
        # 如果所有已知格式都解析失败，返回一个默认值
        print(f"Warning: Could not parse date string '{time_val}'. Using default date.")
        return datetime.min
    
    return datetime.min

def transfer_time(json_data:dict):
    """
    [您提供的函数]
    遍历字典，并使用 parse_date_str 处理时间字段。
    """
    for k,v in json_data.items():
        if "time" in v:
            v["time"] = parse_date_str(v["time"])
    return json_data

def readinfo(data_dir):
    """使用 torch.load 加载 .pt 文件，并有 .json 的备选方案"""
    # (代码与上一版相同)
    file_type = os.path.basename(data_dir).split('.')[-1]
    if not os.path.exists(data_dir):
        if file_type == "pt":
            json_dir = data_dir.replace(".pt", ".json")
            if os.path.exists(json_dir): data_dir = json_dir
            else: raise FileNotFoundError(f"File not found: {data_dir}")
        else: raise FileNotFoundError(f"File not found: {data_dir}")

    if file_type == "pt": return torch.load(data_dir)
    elif file_type == "json":
        with open(data_dir, 'r', encoding='utf-8') as f: return json.load(f)
    else: raise ValueError("Unsupported file type", data_dir)

def writeinfo(data_dir, info):
    """根据文件扩展名，使用 torch.save 或 json.dump 保存信息"""
    # (代码与上一版相同)
    file_type = os.path.basename(data_dir).split('.')[-1]
    if file_type == "pt":
        torch.save(info, data_dir)
    elif file_type == "json":
        with open(data_dir, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=4, separators=(',', ':'), ensure_ascii=False)
    else: raise ValueError("file type not supported")
    print(f"write info to {data_dir}")

def pre_process(article_meta_data:dict, author_data:dict, cur_time:date, load_history:bool=False):
    """调用 transfer_time，并根据 load_history 决定是否过滤数据"""
    article_meta_data = transfer_time(article_meta_data)
    author_data = transfer_time(author_data)
    if load_history:
        return article_meta_data, author_data
    
    article_meta_data = sorted(article_meta_data.items(), key=lambda x: x[1]["time"])
    author_data = dict(sorted(author_data.items(), key=lambda x: x[1]["time"]))
    article_meta_data = dict(filter(lambda x: x[1]["time"] <= cur_time, article_meta_data))
    return article_meta_data, author_data

# --- 数据加载与格式化模块 (DirectoryArticleLoader) ---
class DirectoryArticleLoader(DirectoryLoader):
    def __init__(self, article_meta_data: Dict = {}, loader_cls=TextLoader, loader_kwargs: Optional[Dict] = None):
        self.docs, self.doc_map = [], {}
        self.loader_kwargs = loader_kwargs if loader_kwargs is not None else {}
        self.load_docs_from_meta(article_meta_data, loader_cls, loader_kwargs)

    def load_docs_from_meta(self, article_meta_data: Dict, loader_cls, loader_kwargs):
        print("   - 开始从元数据加载文件...")
        for title, info in article_meta_data.items():
            try:
                path = info.get("path")
                # path = path.replace("/XYFS01/nsccgz_ywang_wzh/cenjc/GraphAgent_GRAG/", "").replace("/XYFS01/nsccgz_ywang_wzh/cenjc/GraphAgent/", "")
                if path and os.path.exists(path):
                    loader = loader_cls(str(path), **loader_kwargs)
                    doc = loader.load()[0]
                    doc.metadata["title"] = title
                    doc.metadata["cited"] = info.get("cited", 0)
                    time_val = info.get("time")
                    doc.metadata["time"] = time_val.strftime("%Y-%m") if isinstance(time_val, (date, datetime)) else str(time_val or "Unknown")
                    doc.metadata["topic"] = info.get("topic", "AI")
                    self.doc_map[title] = len(self.docs)
                    self.docs.append(doc)
                else: print(f"   - 警告: 路径不存在或未提供, 跳过: title='{title}, path='{path}")
            except Exception as e: print(f"   - 错误: 加载文件失败 for title='{title}', path='{info.get('path', 'N/A')}': {e}")
        print(f"   - 文件加载完成。")

    def format_document(self, article_meta_data: Dict, author_data: Dict, doc: Document) -> str:
        title = doc.metadata.get("title", "Unknown Title")
        best_author_info = {"author_cited": "Unknown", "country": "", "institution": "", "author_name": "Unknown"}
        try:
            author_ids = [aid for aid in article_meta_data[title].get("author_ids", []) if aid in author_data]
            if author_ids:
                best_author_id = max(author_ids, key=lambda aid: author_data.get(aid, {}).get("cited", 0))
                best_author = author_data[best_author_id]
                best_author_info = {"author_cited": best_author.get("cited", 0), "country": best_author.get("country", ""), "institution": best_author.get("institution", ""), "author_name": best_author.get("name", "Unknown")}
        except Exception: pass
        prompt = """\
Title: {title}
Cited: {cited}
Author: {author_name} {institution} {country} cited: {author_cited}
Publish Time: {time}
Content: {page_content}"""
        doc_infos = {**doc.metadata, "page_content": doc.page_content[:200], **best_author_info}
        # print(f"*************\n{title} page content: {doc.page_content}\npath:{doc.metadata}**************")
        return prompt.format_map(doc_infos)

# --- 主流程 ---
def calculate_and_save_features(config: Dict):
    """加载数据，计算特征，并保存结果。"""
    print("1. 正在加载元数据 (.pt 文件)...")
    article_meta_data = readinfo(config["article_meta_path"])
    author_data = readinfo(config["author_data_path"])
    print(f"   - `article_meta_data` 加载了 {len(article_meta_data)} 条记录。")
    print(f"   - `author_data` 加载了 {len(author_data)} 条记录。")

    print("2. 正在对元数据进行预处理 (调用 pre_process)...")
    article_meta_data, author_data = pre_process(article_meta_data, author_data, cur_time=datetime.now(), load_history=True)

    print("3. 正在初始化 Embedding 模型...")
    embeddings = HuggingFaceEmbeddings(model_name=config["embedding_model_path"])

    print("4. 正在根据元数据加载文档...")
    article_loader = DirectoryArticleLoader(article_meta_data=article_meta_data, loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
    docs_to_process = article_loader.docs
    if not docs_to_process:
        print("\n 错误: 没有加载到任何文档。请检查 `article_meta.pt` 中的 'path' 字段是否正确。")
        return
    print(f"共加载了 {len(docs_to_process)} 篇有效文档。")

    print("5. 正在格式化文档用于 Embedding...")
    ### testing
    # for doc in docs_to_process:
    #     article_loader.format_document(article_meta_data, author_data, doc)
    #     break
    # exit(0)
    ### testing
    doc_str_list = [article_loader.format_document(article_meta_data, author_data, doc) for doc in docs_to_process]
    
    print("6. 正在计算 Embeddings...")
    embeddings_list = embeddings.embed_documents(doc_str_list)
    embeddings_array = np.array(embeddings_list)
    print(f"   Embeddings 计算完成，矩阵形状: {embeddings_array.shape}")

    print("7. 正在保存结果...")
    os.makedirs(config["output_dir"], exist_ok=True)
    
    # 保存 embedding.npy 和对齐的元数据 .pt 文件
    np.save(os.path.join(config["output_dir"], "embeddings.npy"), embeddings_array)
    aligned_meta_data = {doc.metadata['title']: article_meta_data[doc.metadata['title']] for doc in docs_to_process}
    writeinfo(os.path.join(config["output_dir"], "article_meta_info.pt"), aligned_meta_data)

    print("\n Pipeline 执行完毕！")


if __name__ == "__main__":
    # BiGG.L  bigg_gen
    # GRAN.L  gran_gen
    # BwR.L  bwr_graphrnn
    # GraphMaker.L  graphmaker_sync
    # L-PPGN.L  l_ppgn
    MODEL_CLASS = "l_ppgn"
    DATA_CLASS = "llmcitationciteseer"
    pipeline_config = {
        "article_meta_path": f"{MODEL_CLASS}_{DATA_CLASS}/article_meta_info.pt",
        "author_data_path": f"{MODEL_CLASS}_{DATA_CLASS}/author.pt",
        
        "embedding_model_path": "/home/users/wangzh/cenjch3/all-MiniLM-L6-v2",
        
        "output_dir": f"./output_features/{MODEL_CLASS}/{DATA_CLASS}/"
    }

    calculate_and_save_features(pipeline_config)