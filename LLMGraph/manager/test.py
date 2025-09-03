#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch_geometric.data import HeteroData
from langchain_core.documents import Document
from typing import List, Dict, Any, Tuple
import torch
import random
from torch_geometric.data import HeteroData
from langchain_core.documents import Document
def build_article_graph(
    article_meta: Dict[str, Any],
    author_meta:  Dict[str, Any],
    docs:         List[Document]
) -> Tuple[HeteroData, Dict[int,str], Dict[str,int], Dict[int,Document]]:
    """
    构建一个 HeteroData 图，包含两个节点类型 'article' 和 'author'，
    以及 'cites' 和 'writes' 两种边。

    返回：
      - data:         构建好的 HeteroData
      - node_id_to_title: {article_node_id -> title}
      - title_to_node_id: {title -> article_node_id}
      - index_to_doc: {article_node_id -> Document}
    """
    data = HeteroData()

    # 1) 构建 article 节点映射
    titles = list(article_meta.keys())
    title_to_node_id = {t: i for i, t in enumerate(titles)}
    node_id_to_title = {i: t for t, i in title_to_node_id.items()}
    data['article'].num_nodes = len(titles)

    # 2) 构建 author 节点映射
    author_ids = list(author_meta.keys())
    author_to_node_id = {aid: i for i, aid in enumerate(author_ids)}
    node_id_to_author = {i: aid for aid, i in author_to_node_id.items()}
    data['author'].num_nodes = len(author_ids)

    # 3) 添加 article -> article 引用边（cites）
    src, dst = [], []
    for title, meta in article_meta.items():
        u = title_to_node_id[title]
        for cited in meta.get('cited_articles', []):
            if cited in title_to_node_id:
                v = title_to_node_id[cited]
                src.append(u)
                dst.append(v)
    if src:
        data['article', 'cites', 'article'].edge_index = torch.tensor([src, dst], dtype=torch.long)

    # 4) 添加 author -> article 写作边（writes）
    src, dst = [], []
    for title, meta in article_meta.items():
        v = title_to_node_id[title]
        for aid in meta.get('author_ids', []):
            if aid in author_to_node_id:
                u = author_to_node_id[aid]
                src.append(u)
                dst.append(v)
    if src:
        data['author', 'writes', 'article'].edge_index = torch.tensor([src, dst], dtype=torch.long)

    # 5) 建立 node_idx -> Document 的映射
    index_to_doc: Dict[int, Document] = {}
    # docs 是 DirectoryArticleLoader.load() 返回的 list[Document]
    # 每个 doc.metadata['title'] 对应 article_meta 的 key
    for doc in docs:
        t = doc.metadata.get('title')
        if t in title_to_node_id:
            idx = title_to_node_id[t]
            index_to_doc[idx] = doc

    return data, node_id_to_title, title_to_node_id, index_to_doc


# from LLMGraph.retriever import retriever_registry
from langchain_core.vectorstores import VectorStoreRetriever
from typing import (
    List,
    Dict,
    Callable,
    Optional,
    Union,
    Any)
from langchain_core.pydantic_v1 import root_validator, Field
from langchain_core.documents import Document
from functools import cmp_to_key
import torch
from torch_geometric.data import Data, HeteroData
class GraphRAGRetriever:
    """
    基于异构图的文献检索器，支持两类节点：article 与 author；
    并提供可解释的检索路径信息。
    """
    def __init__(
        self,
        pyg_graph_data: HeteroData,
        node_id_to_title: Dict[int, str],
        title_to_node_id: Dict[str, int],
        index_to_doc: Dict[int, Document],
        edge_types: Optional[List[str]] = None,
        allow_co_citation: bool = False,
        allow_co_author: bool = False,
    ):
        self.graph = pyg_graph_data
        self.node_id_to_title = node_id_to_title
        self.title_to_node_id = title_to_node_id
        self.index_to_doc = index_to_doc
        self.edge_types = edge_types or ["cites", "authored_by", "writes"]
        self.allow_co_citation = allow_co_citation
        self.allow_co_author = allow_co_author

    def retrieve_by_title(self, title: str, hops: int = 1) -> List[Document]:
        """
        以给定标题为起点，返回多跳邻居文献列表，并在文档 metadata 中附加检索路径。
        """
        start_idx = self.title_to_node_id.get(title)
        if start_idx is None:
            return []

        # 保存每个文档的可解释路径
        retrieved: Dict[int, Dict[str, Any]] = {}

        # 初始化当前层级节点集合
        current = {start_idx}
        for hop in range(hops):
            next_level = set()
            # 1. 引用关系扩展
            # article -> cites -> article
            if "cites" in self.edge_types:
                src, dst = self.graph['article', 'cites', 'article'].edge_index
                for u, v in zip(src.tolist(), dst.tolist()):
                    if u in current and v not in retrieved:
                        reason = f"hop_{hop+1}: {self.node_id_to_title[u]} cites {self.node_id_to_title[v]}"
                        next_level.add(v)
                        retrieved.setdefault(v, {"paths": []})["paths"].append(reason)
            # article <- cites <- article (被引用)
            if self.allow_co_citation:
                src, dst = self.graph['article', 'cites', 'article'].edge_index
                for u, v in zip(src.tolist(), dst.tolist()):
                    if v in current and u not in retrieved:
                        reason = f"hop_{hop+1}: {self.node_id_to_title[u]} cited by {self.node_id_to_title[v]}"
                        next_level.add(u)
                        retrieved.setdefault(u, {"paths": []})["paths"].append(reason)

            # 2. 作者扩展
            if self.allow_co_author and "writes" in self.graph.node_types:
                # 获取作者节点
                # article <- writes - author
                aw_src, aw_dst = self.graph['author', 'writes', 'article'].edge_index
                # article -> authored_by -> author
                for aid, art in zip(aw_src.tolist(), aw_dst.tolist()):
                    if art in current:
                        # author aid 写过 article art
                        # 基于该 author 寻找其他文章
                        for a2, art2 in zip(aw_src.tolist(), aw_dst.tolist()):
                            if a2 == aid and art2 not in retrieved and art2 not in current:
                                reason = f"hop_{hop+1}: co-author via author_{aid} wrote {self.node_id_to_title[art2]}"
                                next_level.add(art2)
                                retrieved.setdefault(art2, {"paths": []})["paths"].append(reason)

            current = next_level

        # 构造返回文档列表，并写入 metadata 路径信息
        docs: List[Document] = []
        for idx, info in retrieved.items():
            doc = self.index_to_doc.get(idx)
            if doc:
                # 将检索路径附加到 metadata
                doc.metadata = dict(doc.metadata)
                doc.metadata['retrieval_paths'] = info['paths']
                docs.append(doc)
        return docs

    @classmethod
    def from_db(
        cls,
        pyg_graph_data: HeteroData,
        node_id_to_title: Dict[int, str],
        title_to_node_id: Dict[str, int],
        index_to_doc: Dict[int, Document],
        edge_types: Optional[List[str]] = None,
        allow_co_citation: bool = False,
        allow_co_author: bool = False,
        **kwargs
    ):
        return cls(
            pyg_graph_data=pyg_graph_data,
            node_id_to_title=node_id_to_title,
            title_to_node_id=title_to_node_id,
            index_to_doc=index_to_doc,
            edge_types=edge_types,
            allow_co_citation=allow_co_citation,
            allow_co_author=allow_co_author,
        )

def load_meta(pt_path: str) -> dict:
    """从 .pt 文件加载元数据（article_meta_info 或 author_meta）"""
    return torch.load(pt_path)

def load_docs(article_meta: dict) -> list[Document]:
    """根据 article_meta 的 path 字段读入文本，创建 Document 列表"""
    docs: list[Document] = []
    for title, meta in article_meta.items():
        txt_path = meta['path']
        with open(txt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        docs.append(Document(page_content=content, metadata={'title': title}))
    return docs

def main():
    # 1. 加载 .pt 元数据
    article_meta = load_meta(r'C:\Users\DELL\Desktop\tianhe code\citeseer\445-raw-data\data\article_meta_info.pt')
    author_meta  = load_meta(r'C:\Users\DELL\Desktop\tianhe code\citeseer\445-raw-data\data\author.pt')

    # 2. 加载所有文章文档
    docs = load_docs(article_meta)

    # 3. 构建异构图
    #    build_article_graph(article_meta, author_meta, docs)
    #    返回 (HeteroData, node_id_to_title, title_to_node_id, index_to_doc)
    data, node_id_to_title, title_to_node_id, index_to_doc = build_article_graph(
        article_meta, author_meta, docs
    )

    print(f"data: {data}")
    # 4. 实例化检索器
    retriever = GraphRAGRetriever.from_db(
        pyg_graph_data   = data,
        node_id_to_title = node_id_to_title,
        title_to_node_id = title_to_node_id,
        index_to_doc     = index_to_doc,
        # 确保与图中边名称一致
        edge_types       = ['cites', 'writes'],  
        allow_co_citation= True,
        allow_co_author  = True,
    )

    # 5. 随机选一个 seed 节点（文章标题），测试检索
    seed_title = random.choice(list(article_meta.keys()))
    print(f"\n=== 测试检索起点: {seed_title} ===\n")

    # hops=1 或 2，可自行调整
    results = retriever.retrieve_by_title(seed_title, hops=2)

    # 6. 打印返回的候选集
    for idx, doc in enumerate(results, start=1):
        title = doc.metadata.get('title', '<未知标题>')
        paths = doc.metadata.get('retrieval_paths', [])
        print(f"{idx}. {title}")
        for p in paths:
            print("    ↳", p)
    if not results:
        print(">>> 未检索到任何邻居节点，请检查 edge_types 或 allow_* 参数是否正确。")

if __name__ == "__main__":
    main()
