import networkx as nx
import os
import json 
import torch

def readinfo(data_dir):
    file_type = os.path.basename(data_dir).split('.')[-1]
    try:
        if file_type == "pt":
            return torch.load(data_dir)
    except:
        data_dir = os.path.join(os.path.dirname(data_dir), f"{os.path.join(os.path.basename(data_dir).split('.')[:-1])}.json")
    
    print(data_dir)
    try:
        with open(data_dir, 'r', encoding='utf-8') as f:
            data_list = json.load(f)
            return data_list
    except:
        raise ValueError("file type not supported")

def build_citation_graph(article_meta_data:dict = {}):
    DG = nx.DiGraph()
    
    map_index = {
        title:str(idx) for idx,title in enumerate(article_meta_data.keys())}
    
    
    for title in article_meta_data.keys():
        cited_idx = map_index.get(title)
        time = article_meta_data[title]["time"]
        DG.add_node(cited_idx,title=title,time=time, topic = article_meta_data[title]["topic"])
    
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


import torch

generated_graphs = {}

test_config = "vllm_445"
prefix_graph_sub = 445 # for big, set to the length of seed graph

# test_config = "big"
# prefix_graph_sub = 2997 # for big, set to the length of seed graph
graph_path_2 = f"Emulate/tasks/citeseer/configs/{test_config}/data/article_meta_info.pt"
G_2 = build_citation_graph(readinfo(graph_path_2))
print(f"Node number: {G_2.number_of_nodes()}, Edge number: {G_2.number_of_edges()}")
generated_graphs["llmcitationciteseer"] = []
for idx in range(1,21):
    generated_graphs["llmcitationciteseer"].append(
    nx.Graph(G_2.subgraph(list(G_2.nodes())[prefix_graph_sub+idx:prefix_graph_sub+1000+idx])))


torch.save(
    generated_graphs,
    "Emulate/baselines/baseline_checkpoints/samag_generated.pt"
)