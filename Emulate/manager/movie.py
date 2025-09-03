from collections import Counter, defaultdict
import json
import os

from loguru import logger
from torch_geometric.data import HeteroData
from Emulate.utils.graph import build_movie_graph

from . import manager_registry as ManagerRgistry
from typing import Dict, List, Optional, Tuple,Union,Any
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from Emulate.retriever import retriever_registry
import random
import networkx as nx
from datetime import datetime,date,timedelta
import copy
from agentscope.message import Msg
from langchain_core.prompts import PromptTemplate
from Emulate.tool import create_movie_retriever_tool
from Emulate.loader import Movie1MDatLoader
from Emulate.manager.base import BaseManager
import numpy as np

import time
from Emulate.utils.process_time import transfer_time
from Emulate.tool import create_movie_retriever_tool,create_get_movie_html_tool,create_get_reviews_tool

def writeinfo(data_dir,info):
    with open(data_dir,'w',encoding = 'utf-8') as f:
            json.dump(info, f, indent=4,separators=(',', ':'),ensure_ascii=False)

def readinfo(data_dir):
    assert os.path.exists(data_dir),"no such file path: {}".format(data_dir)
    with open(data_dir,'r',encoding = 'utf-8') as f:
        data_list = json.load(f)
    return data_list


def process_array_time(array):
    """处理最后一列的timestamp"""
    for idx,row in enumerate(array):
        timestamp = row[-1]
        time = transfer_time(timestamp)
        assert isinstance(time,date)
        array[idx][-1] = time
    return array

def pre_process(movie_array:np.ndarray,
                ratings:np.ndarray,
                users:np.ndarray,
                start_time,
                cur_time,
                filter_initial_train = True):
    
    """返回按照时间排序的movie,以及切分到当前时间步的ratings矩阵"""
    
    movie_array = process_array_time(movie_array)
    users = process_array_time(users)
    ratings = process_array_time(ratings)
    
    sorted_indices = np.argsort(movie_array[:, -1])
    movies = movie_array[sorted_indices]
    
    sorted_indices = np.argsort(users[:, -1])
    users = users[sorted_indices]
    
    ratings = ratings[np.isin(ratings[:,0], users[:,0])] 
    ratings = ratings[np.isin(ratings[:,1], movies[:,0])] 
    
    if filter_initial_train:
        ratings = ratings[ratings[:,-1] <= cur_time]
        movies = movies[movies[:,-1] >= start_time]
        users = users[users[:,-1] >= start_time]
    return movies, ratings, users


@ManagerRgistry.register("movie")
class MovieManager(BaseManager):
    """,
        manage infos of movie db
    """,
    
    link_movie_path:str
    movie_data_dir:str
    simulation_time:int = 0
    
    html_tool_kwargs:dict ={
        "upper_token": 1000,
        "url_keys": ["imdbId_url",
                    "tmdbId_url"],
        "llm_kdb_summary": True,
        "llm_url_summary": False,
    }
    
    ratings_log:list = []
    
    movie_loader: Movie1MDatLoader
    
    db: Any = None # 存储历史电影 DB为None时表示没有可看的电影
    db_cur_movies: Any = None # 存储正在热映的电影
    retriever:Any = None
    retriever_cur:Any = None

    watcher_data: np.ndarray
    ratings_data: np.ndarray
    movie_scores: dict = {} # movie_id: score
    
    # llm: OpenAI
    embeddings: Any
    
    generated_data_dir: str
    
    # 核心改动点1: 添加缺失的属性，并移除旧的、重复的属性
    retriever_kargs_update: dict = {}
    movie_graph: Optional[HeteroData] = None
    reviews_index: Optional[Dict[int, List[Dict]]] = None

    watcher_pointer_args:dict = {
        "cnt_p": -1,
        "cnt_watcher_ids":[]
    }
    
    age_map:dict = {
            1:  "Under 18",
            18:  "18-24",
            25:  "25-34",
            35:  "35-44",
            45:  "45-49",
            50:  "50-55",
            56:  "56+"}
        
    occupation_map:dict = {
    0:  "other",
	1:  "academic/educator",
	2:  "artist",
	3:  "clerical/admin",
	4:  "college/grad student",
	5:  "customer service",
	6:  "doctor/health care",
	7:  "executive/managerial",
	8:  "farmer",
	9:  "homemaker",
	10:  "K-12 student",
	11:  "lawyer",
	12:  "programmer",
	13:  "retired",
	14:  "sales/marketing",
	15:  "scientist",
	16:  "self-employed",
	17:  "technician/engineer",
	18:  "tradesman/craftsman",
	19:  "unemployed",
	20:  "writer"}
    graph_data: Any = None
    graph_maps: Dict = None
    docs_map: Dict = None
    graph: Any = None
    

    class Config:
        arbitrary_types_allowed = True
    
    
    
    @classmethod
    def load_data(cls,
                movie_data_dir,
                retriever_kwargs,
                html_tool_kwargs,
                ratings_data_name,
                generated_data_dir, # store all the generated movies, watcher and rating infos
                cur_time: datetime,
                start_time: datetime,
                movie_time_delta: timedelta,
                tool_kwargs,
                control_profile,
                link_movie_path:str = "Emulate/tasks/movielens/data/ml-25m/links.csv"
                ):
        
        if os.path.exists(os.path.join(generated_data_dir,"data")):
            movie_path = os.path.join(movie_data_dir,"movies.npy") 
            ratings_path = os.path.join(generated_data_dir,"data","ratings.npy")
            users_path = os.path.join(generated_data_dir,"data","users.npy")
            ratings_log_path = os.path.join(generated_data_dir,"data","ratings_log.npy")
            ratings_log = np.load(ratings_log_path, allow_pickle=True).tolist()
            filter_initial_train = False
            
        else:
            movie_path = os.path.join(movie_data_dir,"movies.npy") 
            ratings_path = os.path.join(movie_data_dir,ratings_data_name) 
            users_path = os.path.join(movie_data_dir,"users.npy")
            ratings_log = []
            filter_initial_train = True

        
            
        movies = np.load(movie_path,allow_pickle=True)
        ratings = np.load(ratings_path,allow_pickle=True)
        watcher_data = np.load(users_path,allow_pickle=True)
        
        movies, ratings, watcher_data = pre_process(movies,
                                                ratings,
                                                watcher_data,
                                                start_time,
                                                cur_time,
                                                filter_initial_train)
            
        movie_loader = Movie1MDatLoader(movie_data_array = movies,
                                        link_movie_path = link_movie_path,
                                        cur_time=cur_time,
                                        load_movie_html=False)
        
        movie_loader.init(cur_time,
                          movie_time_delta)

        # embeddings = OpenAIEmbeddings()
        try:
            embeddings = HuggingFaceEmbeddings(model_name="/XYFS01/nsccgz_ywang_wzh/cenjc/all-MiniLM-L6-v2")
        except:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # embeddings = HuggingFaceEmbeddings(model_name ="thenlper/gte-small")
        db, db_cur_movies = None, None
        if len(movie_loader.docs)>0:
            db = FAISS.from_documents(movie_loader.docs, 
                                        embeddings)
        else:
            raise Exception("empty online movie DB!")
        if len(movie_loader.cur_movie_docs)>0:
            db_cur_movies = FAISS.from_documents(movie_loader.cur_movie_docs, 
                                                 embeddings)

        watcher_pointer_args ={
            "cnt_p":0,
            "cnt_watcher_ids":[]
        }
        
        movie_scores = {}
        
        for movie_id in movie_loader.movie_data_array[:,0]:
            ratings_sub = ratings[ratings[:, 1] == int(movie_id)]
            assert isinstance(ratings_sub,np.ndarray)
            if ratings_sub[:,2].sum() == 0:
                movie_scores[movie_id] = 0
            else:
                movie_scores[movie_id] = ratings_sub[:,2].mean()

        return cls(
            link_movie_path = link_movie_path,
            movie_data_dir = movie_data_dir,
            embeddings = embeddings,
            watcher_pointer_args = watcher_pointer_args,
            watcher_data = watcher_data,
            ratings_data = ratings,
            movie_loader = movie_loader,
            db = db,
            db_cur_movies = db_cur_movies,
            retriever_kwargs = retriever_kwargs,
            generated_data_dir = generated_data_dir,
            movie_scores = movie_scores,
            html_tool_kwargs = html_tool_kwargs,
            ratings_log = ratings_log,
            tool_kwargs = tool_kwargs,
            control_profile = control_profile,
            movie_graph = None,
            reviews_index = None
        )
        
    def load_history(self):
        log_info_path = os.path.join(self.generated_data_dir,
                                     "data",
                                     "log_info.json")
        content = None
        if os.path.exists(log_info_path):
            log_info = readinfo(log_info_path)
            cur_time = log_info.get("cur_time")
            cur_time = datetime.strptime(cur_time,'%Y-%m-%d').date()
            self.simulation_time = log_info.get("simulation_time",0)
            # self.watcher_pointer_args["cnt_p"] = log_info.get("cur_watcher_num",0)
            content = {"cur_time":cur_time.strftime('%Y-%m-%d'), 
                       "cur_rate":len(self.ratings_log)}
        
        return content
       
    def update_movie_scores(self,
                            movie_scores:dict = {} # movie_id: avg(all rating movie_score)
                            ):
        self.movie_scores.update(movie_scores)
    
    # 核心改动点2: 清理重复方法，只保留最新的get_retriever
    def get_retriever(self, **retriever_kargs_update):
        """
        获取混合检索器。
        内部包含缓存逻辑，仅在配置变更或首次调用时才重新构建。
        """
        if self.movie_graph is None:
            logger.info("Knowledge graph not found. Building for the first time...")
            # 调用与 _update_graph_and_index 相同的逻辑来初始化
            self._update_graph_and_index()
        # 检查配置是否变化，或者retriever尚未初始化
        if self.retriever is None or self.retriever_kargs_update != retriever_kargs_update:
            logger.info(f"Retriever cache miss or config changed. Rebuilding retriever...\nretriever_kargs_update:{retriever_kargs_update}")
            self.retriever_kargs_update = copy.deepcopy(retriever_kargs_update)
            
            # vectorstore 的 online/offline 逻辑可以根据参数决定
            online = retriever_kargs_update.get("online", True)
            vectorstore = self.db if online else self.db_cur_movies
            if vectorstore is None:
                return None

            # 准备构造参数
            retriever_config = self.retriever_kwargs.copy()
            search_kwargs = retriever_config.get("search_kwargs", {})
            k_vector = search_kwargs.get("k_vector", 10)
            k_graph = search_kwargs.get("k_graph", 10)
            retriever_config.update(retriever_kargs_update)
            retriever_type = retriever_config.pop("type", "graph_vector_retriever") 
            
            vector_search_kwargs = search_kwargs.copy()
            vector_search_kwargs['k'] = k_vector
            # 构建底层检索器
            base_vector_retriever = retriever_registry.build(
                retriever_type,
                vectorstore=vectorstore,
                compare_function_type="movie",
                **vector_search_kwargs
            )
            graph_retriever = retriever_registry.build(
                "movie_graph_rag_retriever",
                pyg_graph_data=self.graph_data,
                graph_maps=self.graph_maps,
                docs_map=self.docs_map
            )
            
            # 封装为混合检索器并缓存
            self.retriever = retriever_registry.build(
                "hybrid_movie_retriever",
                vector_retriever=base_vector_retriever,
                graph_retriever=graph_retriever,
                k_graph=k_graph
            )
        return self.retriever

    def update(self, **kwargs):
        """
        此方法接收动态参数，并用它来触发 get_retriever 的缓存更新机制。
        """
        # online, interested_genres 等参数会在这里被接收并传递
        self.get_retriever(**kwargs)

    def get_reviews_tool(self):
        """
        通过标准流程创建 GetReviewsForMovie 工具。
        """
        # 调用在 tool/movie.py 中定义的新函数
        tool_func, func_dict = create_get_reviews_tool(
            manager=self,
            name="GetReviewsForMovie",
            description="Get the existing public reviews for a specific movie. Use this to understand what others think about a movie."
        )
        return tool_func, func_dict  
    
    
    def get_movie_retriever_tool(self,
                                 online = True,
                                interested_genres:list = [],
                                watched_movie_ids:list = [],
                                **retriever_kargs_update):
        retriever = self.get_retriever(online=online, **retriever_kargs_update)

        if retriever is None: return None, None
        
        document_prompt = PromptTemplate.from_template("""
Title: {Title}
Genres: {Genres}
Content: {page_content}""")
        k_final = self.retriever_kwargs.get("k_final", 10)
        tool_func,func_dict = create_movie_retriever_tool(
            retriever,
            "SearchMovie",
            "Search for movies, you should provide some keywords for the movie you want to watch (like genres, plots and directors...)",
            document_prompt = document_prompt,
            k_final=k_final,
            filter_keys=self.tool_kwargs["filter_keys"],
            interested_genres=interested_genres,
            watched_movie_ids=watched_movie_ids)

        return tool_func, func_dict
    
    def get_movie_html_tool(self,
                            online = True,
                            **retriever_kargs_update):
        retriever = self.get_retriever(online=online, **retriever_kargs_update)

        if retriever is None: return None

        movie_html_tool = create_get_movie_html_tool(
            retriever = retriever,
            movie_scores = self.movie_scores,
            name = "GetOneMovie",
            description="You can get the movie html information of one movie you want to watch using this tool.\
[!Important!] You should always give your rating after using this tool!! you should give one movie name",
            **self.html_tool_kwargs
            )
        return movie_html_tool
    
    
    
    def get_cur_movie_docs_len(self):
        return len(self.movie_loader.cur_movie_docs)
    
    def add_movies(self,
                  cur_time:str,
                  movie_time_delta:int):
        """update movie DB"""
        cur_time = datetime.strptime(cur_time,'%Y-%m-%d').date()
        movie_time_delta = timedelta(days=movie_time_delta)

        if self.no_available_movies():
            return
        
        if (cur_time - self.movie_loader.cur_time) < movie_time_delta:
            return
        
        self.movie_loader.update(cur_time)
        
        if len(self.movie_loader.docs)>0:
            self.db = FAISS.from_documents(self.movie_loader.docs, 
                                       self.embeddings)
        else:
            self.db = None
        if len(self.movie_loader.cur_movie_docs)>0:
            self.db_cur_movies = FAISS.from_documents(self.movie_loader.cur_movie_docs, 
                                                  self.embeddings)
        else:
            self.db_cur_movies = None

        
    def filter_rating_movie(self,
                            movie_rating:dict = {},
                            online = True):
        
        try:
            movie_title = movie_rating["movie"]
            
            # 为了全面搜索，我们应该同时在 online 和 offline 的 retriever 中查找
            online_retriever = self.get_retriever(online=True)
            offline_retriever = self.get_retriever(online=False)
            
            retrieved_docs = []
            if online_retriever:
                retrieved_docs.extend(online_retriever.invoke(movie_title))
            if offline_retriever:
                retrieved_docs.extend(offline_retriever.invoke(movie_title))

            for movie_doc in retrieved_docs:
                if movie_title.strip().lower() in movie_doc.metadata["Title"].lower():
                    movie_rating.update({
                        "movie": movie_doc.metadata["Title"],
                        "movie_id": movie_doc.metadata["MovieId"],
                        "genres": movie_doc.metadata["Genres"],  
                    })
                    return movie_rating
        except Exception as e:
            pass  
        return {}
        
    
    def add_watcher(self,
                    cur_time,
                    watcher_num:int =-1,
                    watcher_add:bool = False):
        if not watcher_add:
            if self.watcher_pointer_args["cnt_watcher_ids"] ==[]:
                self.watcher_pointer_args["cnt_watcher_ids"] = list(range(len(self.watcher_data[:watcher_num])))
                self.watcher_pointer_args["cnt_p"] = len(self.watcher_data[:watcher_num])
            return
            
        left_p = self.watcher_pointer_args["cnt_p"]
        upper_idx = np.argmax(cur_time <= self.watcher_data[:,-1])
        # if upper_idx ==0:upper_idx = len(self.watcher_data)
        
        self.watcher_pointer_args["cnt_p"] = upper_idx
        self.watcher_pointer_args["cnt_watcher_ids"] = list(range(left_p,upper_idx))
    
    def add_and_return_watcher_profiles(self,
                                            cur_time:str,
                                            watcher_num:int =-1,
                                            watcher_add:bool = False):
        cur_time = datetime.strptime(cur_time,'%Y-%m-%d').date()
        self.add_watcher(cur_time,
                         watcher_num = watcher_num,
                         watcher_add= watcher_add)
        return self.return_cur_watcher_profiles()
        
    def return_cur_watcher_profiles(self):
       
        return [{
                "infos":
                    {
                        "gender": "Female" if self.watcher_data[idx][1]=="F" else "Male",
                        "age": self.age_map.get(self.watcher_data[idx][2]),
                        "job": self.occupation_map.get(self.watcher_data[idx][3])
                },
                "id":int(self.watcher_data[idx][0]),
                } 
                for idx in self.watcher_pointer_args["cnt_watcher_ids"]]
        
    
    def update_db_ratings(self, ratings: List[Dict]):
        print(f"\nDEBUG INFO --- Manager received {len(ratings)} ratings to update.")
        
        ratings_cur_turn = []
        new_logs = []  # 用于添加到 self.ratings_log

        # 直接遍历扁平的评分字典列表
        for rating_dict in ratings:
            # 从字典中获取 agent_id
            agent_id = rating_dict.get("agent_id")
            
            # 安全检查：如果字典中没有agent_id，则跳过这条记录
            if agent_id is None:
                print(f"ERROR: Rating dictionary is missing 'agent_id'. Skipping. Data: {rating_dict}")
                continue
            
            try:
                timestamp = rating_dict["timestamp"]
                timestamp = transfer_time(timestamp)
                
                # 准备要加入Numpy数组的数据行
                new_row = [
                    int(agent_id),
                    int(rating_dict["movie_id"]),
                    int(rating_dict["rating"]),
                    timestamp
                ]
                ratings_cur_turn.append(new_row)
                
                # 将完整的字典（包含thought）加入log
                new_logs.append(rating_dict)

            except Exception as e:
                # 捕获其他可能的错误，如缺少'movie_id'或类型转换失败
                print(f"ERROR: Failed to process rating dictionary. Error: {e}. Data: {rating_dict}")
                continue

        print(f"\nDEBUG INFO --- Manager processed {len(ratings_cur_turn)} valid ratings.")
        
        if len(ratings_cur_turn) == 0:
            return 0

        ratings_cur_turn_np = np.asarray(ratings_cur_turn)
        
        # 检查并合并Numpy数组
        if self.ratings_data.size == 0: # 检查初始数组是否为空
            self.ratings_data = ratings_cur_turn_np
        elif self.ratings_data.shape[1] != ratings_cur_turn_np.shape[1]:
            print(f"ERROR: Shape mismatch! Existing shape: {self.ratings_data.shape}, New shape: {ratings_cur_turn_np.shape}. Cannot concatenate.")
            return 0
        else:
            self.ratings_data = np.concatenate([self.ratings_data, ratings_cur_turn_np])

        # 更新日志和图谱
        self.ratings_log.extend(new_logs)
        self._update_graph_and_index()
        
        print(f"DEBUG INFO --- Total ratings_data after update: {self.ratings_data.shape[0]}")
        print(f"DEBUG INFO --- Total ratings_log after update: {len(self.ratings_log)}")
        
        return len(ratings_cur_turn)


    def _update_graph_and_index(self):
        """
        辅助函数，用于在评分数据更新后，重建知识图谱和内存中的评论索引。
        """
        logger.info("Updating knowledge graph and reviews index with new data...")
        
        # 1. 重建知识图谱
        # 使用最新的 self.ratings_data 来构建图
        self.movie_graph = build_movie_graph(
            self.watcher_data,
            self.movie_loader.movie_data_array,
            self.ratings_data 
        )
        
        # 2. 重建评论索引
        # 使用最新的 self.ratings_log (它包含了thought文本)
        self.reviews_index = defaultdict(list)
        if len(self.ratings_log) > 0:
            for review in self.ratings_log:
                # 确保review字典中有'movie_id'这个key
                if 'movie_id' in review:
                    self.reviews_index[review['movie_id']].append(review)
        
        logger.info("Knowledge graph and reviews index update finished.")
        

    def get_watcher_rating_infos(self, watcher_id) -> dict: 
        # count 不同movie的观看频率 以及平均打分
        # 现在的做法会time_consuming 仅仅在创建的时候进行调用
        
        arr =  self.ratings_data
        ratings_sub = arr[arr[:, 0] == int(watcher_id)]
        rating_count = {}
        movie_array = self.movie_loader.movie_data_array
        assert isinstance(ratings_sub,np.ndarray)
        for rating in ratings_sub:
            movie_id = rating[1]
            try:
                movie_info = movie_array[movie_array[:,0] == movie_id][0] 
            except: continue
            
            rating_time = rating[3]
            if isinstance(rating_time,date):
                rating_time = rating_time.strftime("%Y-%m-%d")
            rating_info = [(movie_id,rating[2],rating_time)]
            
            genres = movie_info[2].split("|")
            for genre in genres:
                if genre not in rating_count:
                    
                    rating_count[genre] = rating_info
                else: 
                    rating_count[genre].extend(rating_info)

        return rating_count
   
    
    def form_interest_groups(self, all_agent_ids: List[int], group_size_range:  Tuple[int, int] = (3, 5), top_n_genres: int = 3, group_participation_rate: float = 0.4) -> Tuple[List[List[int]], List[int]]:
        """
        根据用户的历史高分评分，将品味相似的用户划分到兴趣小组中。
        :param all_agent_ids: 当前环境中所有用户的ID列表。
        :param group_size_range: 每个小组的人数范围。
        :param top_n_genres: 基于多少个最喜欢的类型来匹配。
        :return: 一个元组，包含(小组列表, 尚未分组的用户ID列表)。
        """
        min_size, max_size = group_size_range
        target_grouped_count = int(len(all_agent_ids) * group_participation_rate)
        final_groups = []
        current_grouped_count = 0
        # 1. 为每个用户计算其最喜欢的Top N电影类型
        user_top_genres = {}
        for user_id in all_agent_ids:
            user_ratings = self.ratings_data[self.ratings_data[:, 0] == user_id]
            high_ratings = user_ratings[user_ratings[:, 2] >= 4]

            if high_ratings.shape[0] == 0:
                continue

            rated_movie_ids = high_ratings[:, 1]
            genre_list = []
            for movie_id in rated_movie_ids:
                try:
                    movie_info = self.movie_loader.movie_data_array[self.movie_loader.movie_data_array[:, 0] == movie_id][0]
                    genres = movie_info[2].split("|")
                    genre_list.extend(genres) # 使用 extend
                except IndexError:
                    continue
            
            if not genre_list:
                continue

            top_genres = [genre for genre, count in Counter(genre_list).most_common(top_n_genres)]
            user_top_genres[user_id] = tuple(sorted(top_genres))

        # 2. 根据共同的兴趣类型对用户进行分组
        groups_by_preference = defaultdict(list)
        for user_id, genres_tuple in user_top_genres.items():
            if not genres_tuple: continue # 忽略没有偏好的用户
            groups_by_preference[genres_tuple].append(user_id)
        valid_potential_groups = [users for users in groups_by_preference.values() if len(users) >= min_size]
        total_potential_users = sum(len(users) for users in valid_potential_groups)
        if total_potential_users >= target_grouped_count:
            print("--- Mode: Surplus. Randomly selecting from interest groups. ---")
            random.shuffle(valid_potential_groups) # 随机打乱这些潜在的小组列表

            for users in valid_potential_groups:
                if current_grouped_count >= target_grouped_count:
                    break
                
                random.shuffle(users) # 组内成员也打乱
                for i in range(0, len(users), max_size):
                    if current_grouped_count >= target_grouped_count:
                        break
                    
                    chunk = users[i:i + max_size]
                    if len(chunk) >= min_size:
                        final_groups.append(chunk)
                        current_grouped_count += len(chunk)

        # 场景二：潜在人数 < 目标人数 (名额不足，需要用随机分组补充)
        else:
            print("--- Mode: Deficit. Taking all interest groups and supplementing with random grouping. ---")
            # Part A: 先收取所有能形成的兴趣小组
            all_interest_candidates = set()
            for users in valid_potential_groups:
                all_interest_candidates.update(users)
                random.shuffle(users)
                for i in range(0, len(users), max_size):
                    chunk = users[i:i + max_size]
                    if len(chunk) >= min_size:
                        final_groups.append(chunk)
                        current_grouped_count += len(chunk)
            
            # Part B: 再用随机分组补足剩余名额
            # 找出所有没有被划入任何有效兴趣小组的用户
            random_candidates = [uid for uid in all_agent_ids if uid not in all_interest_candidates]
            random.shuffle(random_candidates)

            for i in range(0, len(random_candidates), max_size):
                if current_grouped_count >= target_grouped_count:
                    break
                
                chunk = random_candidates[i:i + max_size]
                if len(chunk) >= min_size:
                    final_groups.append(chunk)
                    current_grouped_count += len(chunk)

        # 4. 汇总
        all_grouped_users = {user for group in final_groups for user in group}
        final_ungrouped_users = []
        for user_id in all_agent_ids:
            if user_id not in all_grouped_users:
                final_ungrouped_users.append(user_id)
        return final_groups, final_ungrouped_users
    

    def get_movie_description(self):
        return self.movie_loader.get_movie_description()
    
    def get_movie_types(self):
        return self.movie_loader.get_movie_types()
    
    
    def get_watcher_infos(self,
                          watcher_id,
                          first_person = True):
        if isinstance(watcher_id,str):
            watcher_id = int(watcher_id)
        if first_person:
            prompt_template = """,
I'm a viewer. \
I'm {gender}.
I'm {age} years old, my job is {job}.\
Now I want to watch a movie and give a rating score of the movie I get.\
My task is to give a rating score. """
        else:
            prompt_template = """
You're a viewer. You're {gender}. You're {age} years old, your job is {job}. \
Now you want to watch a movie and give a rating score of the movie you get. \
Your task is to give a rating score. """
        watcher_idx = np.where(self.watcher_data[:,0] == watcher_id)[0][0]
        infos = self.watcher_data[watcher_idx,:]

        age_info = infos[2]
        if self.age_map.get(infos[2]) is not None:
            age_info = self.age_map.get(infos[2])
        infos = self.watcher_data[watcher_idx]
        
        id_oc = infos[3]
        if isinstance(id_oc,int):
            occupation = self.occupation_map.get(id_oc)
        else: occupation = id_oc
        
        infos_dict = {
            "gender": "Female" if infos[1]=="F" else "Male",
            "age": age_info,
            "job": occupation
        }

        return prompt_template.format_map(infos_dict)
    
    
    def save_infos(self,
                   cur_time,
                   start_time):
        data_dir = os.path.join(self.generated_data_dir,"data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        np.save(os.path.join(data_dir,"users.npy"), self.watcher_data)
        np.save(os.path.join(data_dir,"ratings.npy"), self.ratings_data)
        np.save(os.path.join(data_dir,"ratings_log.npy"), self.ratings_log)
        # np.save(os.path.join(data_dir,"movies.npy"), self.movie_loader.movie_data_array)
        simulation_time = time.time() - start_time
        simulation_time += self.simulation_time
        log_info = {"cur_time":datetime.strftime(cur_time, '%Y-%m-%d'),
                    "generated_ratings": len(self.ratings_log),
                    "simulation_time":int(simulation_time),
                    "cur_watcher_num":int(self.watcher_pointer_args["cnt_p"]),
                    "cur_movie_num":int(self.movie_loader.data_ptr)}
        
        writeinfo(os.path.join(data_dir,"log_info.json"), log_info)
        

    
    
    def save_networkx_graph(self):
        model_dir = os.path.join(os.path.dirname(self.generated_data_dir),\
            "model")
        
        # 创建一个空的二部图
        B = nx.Graph()

        # 添加节点，节点可以有属性。这里我们用节点属性'bipartite'标记属于哪个集合
        for watcher_idx in self.watcher_pointer_args["cnt_watcher_ids"]:
            watcher_info = self.watcher_data[watcher_idx]
            watcher_id = watcher_info[0]
            id_oc = watcher_info[3]
            if isinstance(id_oc,int):
                occupation = self.occupation_map.get(id_oc)
            else: occupation = id_oc
            B.add_node(f"watcher_{watcher_id}", 
                           bipartite=0, 
                           gender= "Female" if watcher_info[1]=="F" else "Male",
                           age = watcher_info[2],
                           occupation = occupation
                           )
        docs = self.movie_loader.docs
        cur_docs = self.movie_loader.cur_movie_docs
        docs_all =[*docs, *cur_docs]
        for doc in docs_all:
            movie_id = doc.metadata["MovieId"]
            B.add_node(f"movie_{movie_id}", 
                           bipartite=1,
                           title = doc.metadata["Title"],
                           genres = ", ".join(doc.metadata["Genres"]),
                           timestamp = doc.metadata["Timestamp"].strftime("%Y%m%d")
                           )  
        from tqdm import tqdm
        for row in tqdm(self.ratings_data):
            timestamp = row[3]
            if isinstance(timestamp,int):
                timestamp_str = datetime.fromtimestamp(timestamp).strftime("%Y%m%d")
            if isinstance(timestamp,datetime):
                timestamp_str = timestamp.strftime("%Y%m%d")
            elif isinstance(timestamp,date):
                timestamp_str = timestamp.strftime("%Y%m%d")
            edge_kargs ={
                "rating": row[2],
                "timestamp": timestamp_str
            }
            try:
                assert f"watcher_{int(row[0])}" in B.nodes(), f"watcher_{int(row[0])} not available"
                assert f"movie_{int(row[1])}" in B.nodes(), f"movie_{int(row[1])} not available"
                B.add_edge(f"watcher_{int(row[0])}",
                            f"movie_{int(row[1])}",
                            **edge_kargs)
            except Exception as e:
                logger.info(f"Exception when saving: {e}\ndata: {row}")
                continue
        # 添加边，连接集合0和集合1的节点
        # B.add_edges_from([('a', 1), ('b', 2), ('c', 3), ('a', 3), ('a', 4), ('b', 4)])
        model_path = os.path.join(model_dir,"graph.graphml")
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
        nx.write_graphml(B, model_path)
    
    def get_movie_rating_score(self,
                               movie_id):
        arr =  self.ratings_data
        ratings_sub = arr[arr[:, 1] == int(movie_id)]
        assert isinstance(ratings_sub,np.ndarray)
        if ratings_sub[:,2].sum() == 0:
            rating_score = 0
        else:
            rating_score = ratings_sub[:,2].mean()
            
        return rating_score
    
    def update_watcher(self,
                       llm):
        prompt ="""
Your task is to give me a list of watcher's profiles. Respond in this format:
[
{
"gender": (F/M)
"age":(the age of the watcher)
"job":(the job of the watcher)
}
]

Now respond:
"""
        response = llm.invoke(prompt)
        content = response.content
        id_prefix = self.watcher_data.shape[0]
        try:
            watcher_data_update = json.loads(content)
            watcher_data_update_array = []
            for idx,watcher in enumerate(watcher_data_update):
                try:
                    gender = watcher["gender"]
                    if gender not in ["F",'M']:
                        gender = random.choice(["F","M"])
                    occupation = watcher["job"]
                    age = int(watcher["age"])  
                    watcher_profile = [id_prefix+idx,
                                       gender,
                                       age,
                                       occupation,
                                       np.nan # No zip:code
                                       ]
                    watcher_data_update_array.append(watcher_profile)
                except:continue
                
            self.watcher_data = np.concatenate([self.watcher_data,
                                                watcher_data_update_array])
        except Exception as e:return
        
    def no_available_movies(self):
        if self.movie_loader.movie_data_array.shape[0] == \
            self.movie_loader.data_ptr:
            return True
        return False
    
    def get_movie_array(self):
        return self.movie_loader.movie_data_array
    
    def get_docs_len(self):
        return len(self.movie_loader.docs)
    
    def get_reviews_for_movie(self, movie_id: int, n_latest: int = 5) -> Dict:
        """
        Get movie latest ratings & comment.
        """
        if not self.ratings_log:
            return {"reviews": "No public reviews available yet."}
        
        relevant_reviews = [
            r for r in self.ratings_log if r.get("movie_id") == movie_id
        ]

        if not relevant_reviews:
            return {"reviews": "No public reviews available for this movie yet."}

        relevant_reviews.sort(key=lambda r: r.get('timestamp', '0'), reverse=True)
        latest_reviews = relevant_reviews[:n_latest]

        reviews_text = "\n".join(
            [f"- A user gave a rating of {r['rating']}/5 and commented: '{r['thought']}'" for r in latest_reviews]
        )

        return {"reviews": reviews_text}

    def get_movie_available_num(self,
                                 watched_movie_ids:list = []):
        if self.movie_loader.data_ptr >= self.movie_loader.movie_data_array.shape[0]:
            movie_ids = self.movie_loader.movie_data_array[:,0]
        else:
            movie_ids = self.movie_loader.movie_data_array[
                    :self.movie_loader.data_ptr,0]
        return len(list(filter(lambda movie_id: movie_id not in watched_movie_ids, movie_ids)))
    

    def get_rating_counts(self,
                          rating_counts_id:dict = {}):
        rating_counts = {}
        ratings = {}
        for genre, movie_ratings in rating_counts_id.items():
            rating_counts[genre] = [movie_rating[1] for movie_rating in \
                movie_ratings]
            for movie_rating in movie_ratings:
                movie_id, rating, timestamp = movie_rating
                try:
                    movie_info = self.movie_loader.movie_data_array[
                        self.movie_loader.movie_data_array[:,0] == movie_id]
                    # time = transfer_time(timestamp)
                    ratings[movie_id] = {
                        "movie":movie_info[0][1],
                        "thought": "",
                        "rating": rating,
                        "timestamp": timestamp
                    }
                except:pass

        return {
            "rating_counts":rating_counts,
            "ratings":ratings
        }


    def get_offline_movie_info(self,
                                   filter: dict = {}, 
                                   max_movie_number:int = 20):
        movie_template = """{Title}: {page_content}"""
        cur_movie_doc_len = self.get_cur_movie_docs_len()
        if cur_movie_doc_len == 0:
            return ""
        
        # 使用新的 get_retriever
        retriever = self.get_retriever(online=False)
        if not retriever:
            return ""

        all_movies = []
        # 确保filter字典和其键存在
        if "interested_genres" in filter and filter["interested_genres"]:
            for genre in filter["interested_genres"]:
                sub_docs = retriever.invoke(genre)
                if len(sub_docs) > 5:
                    sub_docs = sub_docs[:5]
                all_movies.extend(sub_docs)
        
        # 去重
        seen_ids = set()
        unique_movies = []
        for doc in all_movies:
            if doc.metadata['MovieId'] not in seen_ids:
                unique_movies.append(doc)
                seen_ids.add(doc.metadata['MovieId'])
        
        if len(unique_movies) > max_movie_number:
            all_movies = random.sample(unique_movies, max_movie_number)
        else:
            all_movies = unique_movies


        searched_movie_info = []
        for doc in all_movies:
            searched_movie_info.append(movie_template.format_map({
                "page_content":doc.page_content,
                **doc.metadata
            }))
            
        searched_movie_info = "\n".join(searched_movie_info)
        return searched_movie_info
    
    def get_role_description(self):
        pass