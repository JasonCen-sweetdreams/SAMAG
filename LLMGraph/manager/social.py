""" load the basic infos of authors""",

import json
import os
import traceback
from LLMGraph.manager.base import BaseManager
from . import manager_registry as ManagerRgistry
from typing import Dict, List, Union, Any, Optional
from langchain_core.prompts import BasePromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
import random
import networkx as nx
from LLMGraph.retriever import retriever_registry
from LLMGraph.tool import create_forum_retriever_tool
from LLMGraph.utils.graph import build_social_graph
from datetime import datetime
import copy
from langchain_core.prompts import PromptTemplate
import pandas as pd
from datetime import datetime, date, timedelta
from agentscope.models import ModelWrapperBase
from LLMGraph.loader.social import SocialLoader
from LLMGraph.output_parser.base_parser import find_and_load_json
from agentscope.message import Msg
from agentscope.models import load_model_by_config_name
from collections import Counter
from typing import ClassVar, List
from loguru import logger
import torch
from torch_geometric.data import HeteroData
from langchain_core.documents import Document
import threading

def writeinfo(data_dir,info):
    try:
        with open(data_dir,'w',encoding = 'utf-8') as f:
            json.dump(info, f, indent=4,separators=(',', ':'),ensure_ascii=False)
    except Exception as e:
        print(f"[CRITICAL ERROR] Failed to write to {data_dir}. Error: {e}")

def readinfo(data_dir):
    assert os.path.exists(data_dir),"no such file path: {}".format(data_dir)
    with open(data_dir,'r',encoding = 'utf-8') as f:
        data_list = json.load(f)
    return data_list

@ManagerRgistry.register("social")
class SocialManager(BaseManager):
    """,
        manage infos of social db
    """,
    
    class Config:
        arbitrary_types_allowed = True
    
    social_member_data: pd.DataFrame

    forum_loader: SocialLoader
    action_logs: list = [] # 存储 ac_id, f_id, timestamp, ac_type
    
    embeddings: Any
    generated_data_dir:str
    
    last_added_time:date = date.min
    last_added_index:int = 0
    
    follow_map:dict = {} # 大v:num
    
    start_time: date = None
    simulation_time:int =0
    db:Any = None
    retriever_kargs_update:dict = {}
    retriever:Any = None
    big_name_list:list = []
    social_graph: Optional[HeteroData] = None
    _idx_to_doc: Dict[int, Document] = {}
    active_users: set = set()
    # debug 
    transitive_agent_log:list = [] # 记录每一轮的agent增减ids
    
    class Config:
        arbitrary_types_allowed = True
    
    @classmethod
    def load_data(cls,
                  cur_time,
                  social_data_dir,
                  generated_data_dir,
                  data_name,
                  retriever_kwargs,
                  control_profile,
                  tool_kwargs:dict = {
                        "filter_keys": ["follow", "big_name", "topic"],
                        "hub_connect": False
                    }
                  ):
        social_member_data_path = os.path.join(social_data_dir,f"{data_name}_members.csv")
        data_path = os.path.join(social_data_dir,f"{data_name}.csv")
        forum_df = pd.read_csv(data_path,index_col=None)
        simulation_time = 0
        # load history experiment
        social_member_dir = os.path.join(generated_data_dir,"data","social_network")
        last_added_time = date.min
        if os.path.exists(social_member_dir):
            paths = os.listdir(social_member_dir)
            paths = sorted(paths)
            social_member_path = os.path.join(social_member_dir,
                                              paths[-1])
            cur_time = datetime.strptime(os.path.basename(paths[-1]).split(".")[0][-8:],"%Y%m%d").date()
            social_member_data = pd.read_csv(social_member_path,index_col=None)
            list_cols = ["follow","friend"]
            for list_col in list_cols:
                social_member_data[list_col]  = [json.loads(sub_data) for sub_data in\
                                social_member_data[list_col]]
            action_log_path = os.path.join(generated_data_dir,"data","action_logs.json")
            action_logs = readinfo(action_log_path)
            print(f"Loading transitive_agent_log at current time: {cur_time}...")
            transitive_agent_log_path = os.path.join(generated_data_dir,"data","transitive_agent_log.json")
            transitive_agent_log = readinfo(transitive_agent_log_path)
            print(f"Loaded transitive_agent_log at current time: {cur_time}, log length: {len(transitive_agent_log)}")
            for ac_log in action_logs:
                ac_log[-1] = datetime.strptime(ac_log[-1],"%Y-%m-%d").date()
            
            forum_path = os.path.join(generated_data_dir,"data","forum.json")
            forum_loader = SocialLoader(data_path=forum_path)
            ex_logs_path = os.path.join(generated_data_dir,"data","ex_logs.json")
            ex_logs = readinfo(ex_logs_path)
            # last_added_time = datetime.strptime(ex_logs["last_added_time"],"%Y%m%d").date()
            simulation_time = ex_logs.get("simulation_time",0)
        else:
            # forum_df['user_index'] = forum_df.index
            # social_member_data = forum_df[['user_index',
            #                             "user_name",
            #                             "user_description",
            #                             "user_followers"]].drop_duplicates(
            #                                 subset="user_name", keep='first')
            # social_member_data["follow"] = [[] for i in range(social_member_data.shape[0])]
            # # 这个部分可能要增加一个action，如果 不是大v，理论上来说有人关注你很大概率是会互关的
            # social_member_data["friend"] = [[] for i in range(social_member_data.shape[0])]
            social_member_data  = pd.read_csv(social_member_data_path,index_col = None)
            social_member_data["follow"] = [eval(follow_list) for follow_list in social_member_data["follow"]]
            social_member_data["friend"] = [eval(friend_list) for friend_list in social_member_data["friend"]]
            forum_loader = SocialLoader(social_data=forum_df)
            action_logs = []
            transitive_agent_log = []
        
        follow_map = {}
        for user_index in social_member_data['user_index']:
            follow_map[user_index] = len(social_member_data.loc[user_index,"follow"]) + \
                len(social_member_data.loc[user_index,"friend"])
        # embeddings = OpenAIEmbeddings()
        try:
            embeddings = HuggingFaceEmbeddings(model_name="/XYFS01/nsccgz_ywang_wzh/cenjc/all-MiniLM-L6-v2")
        except:
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        #embeddings = HuggingFaceEmbeddings(model_name ="thenlper/gte-small",cache_folder="LLMGraph/utils/model")
  
        initial_ids = social_member_data["user_index"].tolist()
        active_users = set(initial_ids)
        print(f"++++++++++++ SocialManager instance is being created via load_data. Time: {datetime.now()} ++++++++++++")

        return cls(
            forum_loader = forum_loader,
            social_member_data = social_member_data,
            retriever_kwargs =retriever_kwargs,
            embeddings = embeddings,
            generated_data_dir=generated_data_dir,
            action_logs = action_logs,
            start_time = cur_time,
            last_added_time = last_added_time,
            simulation_time = simulation_time,
            follow_map = follow_map,
            transitive_agent_log = transitive_agent_log,
            tool_kwargs = tool_kwargs,
            control_profile = control_profile,
            active_users = active_users
        )
        
    def get_start_time(self):
        date = datetime.strftime(self.start_time,"%Y-%m-%d")
        return date

    
    def get_user_role_description(self,user_index):
        if isinstance(user_index,str):
            user_index = int(user_index)
        
        template ="""
You are a visitor on Twitter. You are {user_description}. 
You have {user_followers} followers.

{follower_description}
"""      
        follower_des = self.get_follower_description(user_index)
        try:
            infos = self.social_member_data.loc[user_index,:].to_dict()
            infos["user_followers"] = self.follow_map.get(user_index,0)
            infos["follower_description"] = follower_des
        except:
            pass
        
        return template.format_map(infos)
    
    def get_user_short_description(self,user_index):
        if isinstance(user_index,str):
            user_index = int(user_index)
        template ="""{user_name}, {user_description}"""
        infos = self.social_member_data.loc[user_index,:].to_dict()
        return template.format_map(infos)
    
    def get_user_friend_info(self,
                             user_index,
                             threshold:int = 30 # 超过30, 不显示friend信息
                             ):
        if isinstance(user_index,str):
            user_index = int(user_index)
        template ="""
Your friend include: 
{friends}
"""      
        friend_template = """{idx}:{short_description}"""
        
        infos = self.social_member_data.loc[user_index,:].to_dict()
        
        if len(infos["friend"]) > threshold:
            return f"""You have {len(infos["friend"])} friends, which are too many to show. You may consider not to follow others."""
        friend_infos = []
        for idx, friend_id in enumerate(infos["friend"]):
            friend_des = friend_template.format(idx = idx,
            short_description = self.get_user_short_description(friend_id))
            friend_infos.append(friend_des)
        
        return template.format(friends = "\n".join(friend_infos))
    

    def delete_user_profiles(self,
                            cur_time:str,
                            add_user_time_delta:int,
                            num_delete:int = 5) -> list:
        cur_time = datetime.strptime(cur_time,"%Y-%m-%d").date()
        add_user_time_delta = timedelta(days=add_user_time_delta)
        num_agents = self.social_member_data.shape[0]
        num_delete = min(num_agents - 1,num_delete)
        if num_delete < 1:return []
        if self.last_added_time == date.min or \
            add_user_time_delta > (cur_time - self.last_added_time):
            return []
        else:
            ids_delete = random.sample(self.social_member_data.index.to_list(),
                                       num_delete)
            # 图数据库更新
            for uid in ids_delete:
                if uid in self.active_users:
                    self.active_users.remove(uid)
            self._update_user_graph()
            return ids_delete
        
    def rerun_set_time(self,last_added_time):
        if isinstance(last_added_time,str):
            last_added_time = datetime.strptime(last_added_time,"%Y-%m-%d").date()
        self.last_added_time = last_added_time

    def rerun(self):
        print(f"self.transitive_agent_log is {len(self.transitive_agent_log)>0}")
        return len(self.transitive_agent_log)>0

    def denote_transitive_log(self, 
                              delete_ids,
                              add_ids,
                              cur_time: str):
        print(f"Appending log for {cur_time}. Manager id: {id(self)}")
        pid = os.getpid()
        tid = threading.get_ident()
        print(f"[PID: {pid}, TID: {tid}] Appending log for date: {cur_time}")
        self.transitive_agent_log.append({
            "date": cur_time,
            "delete_ids":delete_ids,
            "add_ids":add_ids
        })
        

    def update_docs(self):
        docs = self.forum_loader.load()
        self.db = FAISS.from_documents(docs, 
                                    self.embeddings)
    
    def get_forum_retriever(self,
                            **retriever_kargs_update):
        """offline的时候不进行filter"""
        if self.db is None:
            docs = self.forum_loader.load()
            self.db = FAISS.from_documents(docs, 
                                    self.embeddings)
        if (self.retriever_kargs_update != retriever_kargs_update) or \
            self.retriever is None:
            retriever_args = copy.deepcopy(self.retriever_kwargs)
            retriever_args["vectorstore"] = self.db
            retriever_args.update(retriever_kargs_update)
            retriever_args["compare_function_type"] = "social"
            # self.retriever = retriever_registry.from_db(**retriever_args)
            vector_retriever = retriever_registry.from_db(**retriever_args)

            if self.social_graph is None:
                all_docs = self.forum_loader.load()
                graph_data, idx_to_doc = build_social_graph(self.social_member_data, all_docs)
                self.social_graph = graph_data
                self._idx_to_doc = idx_to_doc
            else:
                all_docs = self.forum_loader.load()
                _, idx_to_doc = build_social_graph(self.social_member_data, all_docs)
                self._idx_to_doc = idx_to_doc
            
            graph_retriever_args = {
                "type": "graph_structure_retriever_for_social",
                "data": self.social_graph,
                "idx_to_doc": self._idx_to_doc,
                "max_hops": 1
            }
            graph_retriever = retriever_registry.from_db(**graph_retriever_args)

            hybrid_retriever_args = {
                "type": "hybrid_graph_vector_retriever_for_social",
                "vector_retriever": vector_retriever,
                "graph_retriever": graph_retriever
            }
            hybrid = retriever_registry.from_db(**hybrid_retriever_args)
            self.retriever = hybrid
            # self.retriever_kargs_update = copy.deepcopy(retriever_kargs_update)
        return self.retriever
    
    def get_forum_retriever_tool(self,
                            document_prompt: Optional[BasePromptTemplate] = None,
                            social_follow_map:dict = {
                                "follow_ids": [],
                                "friend_ids": []
                            },
                            max_search:int = 5,
                            interested_topics:list = [],
                            **retriever_kargs_update):
        
        retriever = self.get_forum_retriever(**retriever_kargs_update)
        
        if retriever is None:return
        
        document_prompt = PromptTemplate.from_template("""
{tweet_idx}:
    user: {user_name}
    topic: {topic}
    tweet: {page_content}""")
        retriever_tool = create_forum_retriever_tool(
                    retriever,
                    "search_forum",
                    "You can search for anything you are interested on this platform.",
                    document_prompt = document_prompt,
                    big_name_list = self.big_name_list,
                    filter_keys = self.tool_kwargs["filter_keys"],
                    social_follow_map = social_follow_map,
                    interested_topics = interested_topics,
                    max_search = max_search,
                    hub_connect = self.tool_kwargs.get("hub_connect",True))
        return retriever_tool
    

    def update_add_user_time(self,
                             cur_time:str):
        cur_time = datetime.strptime(cur_time,"%Y-%m-%d").date()
        self.last_added_time = cur_time


    def add_and_return_user_profiles(self, 
                                     cur_time:str,
                                     add_user_time_delta:int,
                                     num_added:int = 5) -> dict:
        cur_time = datetime.strptime(cur_time,"%Y-%m-%d").date()
        add_user_time_delta = timedelta(days=add_user_time_delta)
        # llm = ChatOpenAI(model="gpt-3.5-turbo-16k-0613",
        #                  temperature=0.7,
        #                  max_tokens=2000)
        llm = load_model_by_config_name("default")
        profiles = {}
        ids_added = []

        if self.last_added_time == date.min:
            print("初始化")
            if self.social_member_data.shape[0] < self.last_added_index:
                return {}
            # init data
            message_threshold = 10000
            profiles = self.social_member_data.iloc[self.last_added_index:]
            if profiles.shape[0] > message_threshold:
                profiles = profiles.iloc[:message_threshold]
                self.last_added_index += profiles.shape[0]
            ids_added = profiles.index.to_list()
            for uid in ids_added:
                self.active_users.add(uid)
            if ids_added:
                self._update_user_graph()
            return profiles.to_dict()

        elif add_user_time_delta > (cur_time - self.last_added_time):
            profiles = {}
        else:
            try:
                ids_added_num = 0
                step = num_added if num_added < 5 else 5
                for i in range(0,num_added,step):
                    # ids_sub = self.update_person(llm, step)
                    ids_sub = self.update_person(llm, min(step, num_added - ids_added_num))
                    # ids_added = ids_sub
                    ids_added.extend(ids_sub)
                    ids_added_num += len(ids_sub)
                    if ids_added_num >= num_added:
                        break
                ids = self.social_member_data.index.to_list()[-ids_added_num:]
                profiles_df = self.social_member_data.loc[ids,:]
                friend_ids = profiles_df['user_index'].to_list()

                for index, profile_row in profiles_df.iterrows():
                    friend_ids_cp = copy.deepcopy(friend_ids)
                    friend_ids_cp.remove(profile_row['user_index'])
                    for friend_id_ in friend_ids_cp:
                        if friend_id_ not in profiles_df.loc[index, "friend"]:
                            profiles_df.loc[index, "friend"].append(friend_id_)
                try:
                    assert profiles_df.shape[0]<= num_added, print("error", f"update_agents \
                                                                   {num_added}/{profiles_df.shape[0]}")
                except Exception as e:
                    print(f"Exception in add_and_return_user_profiles1: {e}, {cur_time}")
                    return {}
                
                profiles = profiles_df.to_dict()
            except Exception as e:
                print(f"Exception in add_and_return_user_profiles2: {e}, {cur_time}")
                profiles = {}
                ids_added = []
        for uid in ids_added:
            self.active_users.add(uid)
        if ids_added:
            self._update_user_graph()
        else:
            print(f"ids_added 为空！TIME: {cur_time}")
        return profiles
    

    def add_tweets_batch(self,
                         data_list: list[dict]):
        """
        data_list: 
          [
            {"agent_id": 3, "cur_time": "2025-06-06", "twitters": [ {...}, {...} ]},
            {"agent_id": 5, "cur_time": "2025-06-06", "twitters": [ {...} ]},
            ...
          ]
        返回：总共真正写入的推文条数（int）
        """
        # 1）先把所有 action_logs 和 tweets 汇总到同一个列表里
        all_action_logs = []
        all_tweets = []
        available_actions = ["tweet", "retweet", "reply"]
        logger.info(f"Adding {len(data_list)} tweets of one batch!")
        num_docs = len(self.forum_loader.docs)
        for entry in data_list:
            agent_id = entry["agent_id"]
            # 可能传过来的 agent_id 是字符串，先转成 int
            if isinstance(agent_id, str):
                agent_id = int(agent_id)
            cur_time_str = entry["cur_time"]
            cur_date = datetime.strptime(cur_time_str, "%Y-%m-%d").date()
            twitters = entry.get("twitters", [])
            if isinstance(twitters, dict):
                twitters = [twitters]

            for tweet in twitters:
                try:
                    action = tweet.get("action", "retweet").lower()
                    if action == "tweet":
                        send_tweet = {
                            "text": tweet.get("input"),
                            "user_name": self.social_member_data.loc[agent_id, "user_name"],
                            "user_index": agent_id,
                            "topic": tweet.get("topic"),
                            "action": action,
                            "origin_tweet_idx": -1,
                            "owner_user_index": agent_id,
                        }
                        all_tweets.append(send_tweet)
                        all_action_logs.append([agent_id, agent_id, action, cur_date])

                    elif action in available_actions:
                        t_id_str = tweet.get("tweet_index")
                        if t_id_str is None:
                            logger.warning(f"Agent {agent_id} a retweet/reply action lacks 'tweet_index'. Skipping.")
                            continue
                        # t_id = int(tweet.get("tweet_index"))
                        t_id = int(t_id_str)
                        if t_id >= num_docs or t_id < 0:
                            logger.warning(
                                f"Invalid tweet_index {t_id} from agent {agent_id}. "
                                f"Index is out of bounds for docs list size {num_docs}. Skipping this action."
                            )
                            continue
                        try:
                            tweet_info_db = self.forum_loader.docs[int(t_id)]
                        except Exception as e:
                            logger.warning(
                                f"Error retweet tweet_index {t_id} from agent {agent_id}. "
                                f"{e}. \nSkipping this action."
                            )
                            continue

                        if action == "reply":
                            reply_tweet = {
                                "text": tweet.get("input"),
                                "user_name": self.social_member_data.loc[agent_id, "user_name"],
                                "user_index": agent_id,
                                "topic": tweet_info_db.metadata.get("topic", "default"),
                                "action": action,
                                "origin_tweet_idx": tweet_info_db.metadata.get("tweet_idx", -1),
                                "owner_user_index": tweet_info_db.metadata.get("owner_user_index", agent_id),
                            }
                            all_tweets.append(reply_tweet)

                        owner_id = tweet_info_db.metadata.get("owner_user_index")
                        assert owner_id is not None

                        if tweet.get("follow"):
                            if (
                                owner_id not in self.social_member_data.loc[agent_id, "follow"]
                                and owner_id not in self.social_member_data.loc[agent_id, "friend"]
                                and owner_id != agent_id
                            ):
                                if agent_id in self.social_member_data.loc[owner_id, "follow"]:
                                    # 互关，改为 friend
                                    self.social_member_data.loc[owner_id, "friend"].append(agent_id)
                                    self.social_member_data.loc[owner_id, "follow"].remove(agent_id)
                                    self.social_member_data.loc[agent_id, "friend"].append(owner_id)
                                else:
                                    self.social_member_data.loc[agent_id, "follow"].append(owner_id)

                                all_action_logs.append([agent_id, owner_id, "follow", cur_date])

                        # 不管是不是 follow，只要是 retweet 或 reply，都要记 action_log
                        all_action_logs.append([agent_id, owner_id, action, cur_date])

                        # 如果有 @ 提及
                        for mention_id in tweet.get("mention", []):
                            mention_id = int(mention_id)
                            if mention_id != agent_id:
                                all_action_logs.append([agent_id, mention_id, "mention", cur_date])

                except Exception as e:
                    # 任何单条推文处理异常，都跳过
                    logger.exception(f"[ERROR] agent_id={agent_id}: {e}\n{traceback.print_exc()}\nTweet: {tweet}")
                    continue

        # 2）把所有 action_logs 追加到 self.action_logs
        self.action_logs.extend(all_action_logs)

        # 3）如果没有真正要写入的推文，直接返回 0
        if len(all_tweets) == 0:
            logger.warning(f"all_tweets len is 0! Check below:\n{data_list[:3]}")
            return 0

        # 4）把所有推文一次性转 DataFrame 并送给 forum_loader.add_social
        tweets_df = pd.DataFrame(all_tweets)
        docs = self.forum_loader.add_social(tweets_df)
        logger.info(f"Added {len(docs)} new docs to loader!")

        # 5）一次性算 embedding，合并到底层索引
        db_update = FAISS.from_documents(docs, self.embeddings)
        logger.info("Get new tweets embedding (batch)!")
        self.db.merge_from(db_update)
        logger.info("Database updated (batch)!")

        # 6）返回真正写入的推文数
        return len(docs)


    def add_tweets(self, 
                   agent_id: Union[str,int],
                   cur_time:str,
                   twitters:list = []):
        
        if isinstance(agent_id,str):
            agent_id = int(agent_id)
        cur_time = datetime.strptime(cur_time,"%Y-%m-%d").date()
        """update action logs"""
        tweets = []
        action_logs = []
        available_actions = ["tweet","retweet","reply"]
        for tweet in twitters:
            try:
                action = tweet.get("action","retweet").lower()
                if action == "tweet":
                    send_tweet = {
                        "text": tweet.get("input"),
                        "user_name": self.social_member_data.loc[agent_id,
                                                                 "user_name"],
                        "user_index": agent_id,
                        "topic":tweet.get("topic"),
                        "action":action,
                        "origin_tweet_idx": -1,
                        "owner_user_index": agent_id,
                    }
                    tweets.append(send_tweet)
                
                elif action in available_actions:
                    t_id = int(tweet.get("tweet_index"))
                    tweet_info_db = self.forum_loader.docs[int(t_id)]

                    if action == "reply":
                        reply_tweet = {
                            "text":tweet.get("input"),
                            "user_name": self.social_member_data.loc[agent_id,
                                                                    "user_name"],
                            "user_index": agent_id,
                            "topic": tweet_info_db.metadata.get("topic","default"),
                            "action":action,
                            "origin_tweet_idx": tweet_info_db.metadata.get("tweet_idx",-1),
                            "owner_user_index": tweet_info_db.metadata.get("owner_user_index",agent_id),
                        }
                        tweets.append(reply_tweet)

                    owner_id = tweet_info_db.metadata.get("owner_user_index")
                    assert owner_id is not None
                    if tweet.get("follow"):
                        if owner_id not in self.social_member_data.loc[agent_id,"follow"] and \
                            owner_id not in self.social_member_data.loc[agent_id,"friend"] and\
                            owner_id != agent_id:
                            if agent_id in self.social_member_data.loc[owner_id,"follow"]:
                                # 互关了，需要改成friend
                                self.social_member_data.loc[owner_id,"friend"].append(agent_id)
                                self.social_member_data.loc[owner_id,"follow"].remove(agent_id)
                                self.social_member_data.loc[agent_id,"friend"].append(owner_id)
                            else:
                                self.social_member_data.loc[agent_id,"follow"].append(owner_id) 

                            action_logs.append([agent_id,owner_id,"follow",cur_time]) 
                 
                if action in available_actions:   
                    action_logs.append([agent_id,owner_id,action,cur_time])
                
                for mention_id in tweet.get("mention",[]):
                    mention_id = int(mention_id)
                    if mention_id != agent_id:
                        action_logs.append([agent_id,mention_id,"mention",cur_time])

            except Exception as e:
                continue
            
        self.action_logs.extend(action_logs)
        ### 更新向量数据库
        if len(tweets) > 0:
            tweets = pd.DataFrame(tweets)
            docs = self.forum_loader.add_social(tweets)
            logger.info(f"Added docs to loader!")
            db_update = FAISS.from_documents(docs, self.embeddings)
            logger.info(f"Get new tweets embedding!")
            self.db.merge_from(db_update)
            logger.info(f"Database updated!")
            self._incremental_update_graph(docs)
        
        return len(tweets)
            
    def _incremental_update_graph(self, new_docs: List[Document]):
        """
        把 new_docs 中的每个 Document 对应的 'tweet' 节点 和 'user->tweets' / 'user->retweets' 边，
        插入到 self.social_graph 里。
        """
        # 若社交图尚未构建，直接整体重建
        if self.social_graph is None:
            all_docs = self.forum_loader.load()
            graph_data, idx_to_doc = build_social_graph(self.social_member_data, all_docs)
            self.social_graph = graph_data
            self._idx_to_doc = idx_to_doc
            return

        for doc in new_docs:
            t_idx = int(doc.metadata["tweet_idx"])
            u_idx = int(doc.metadata["user_index"])
            action = doc.metadata.get("action", "tweet").lower()

            # 1) 更新 tweet 节点总数
            if t_idx >= self.social_graph['tweet'].num_nodes:
                self.social_graph['tweet'].num_nodes = t_idx + 1

            # 2) 更新 idx_to_doc 映射（以便后续在图检索中能拿到 metadata）
            self._idx_to_doc[t_idx] = doc

            # 3) 插入 “user->tweet” 或 “user->retweet” 边
            if action == "tweet":
                old_ei = self.social_graph['user','tweets','tweet'].edge_index \
                         if ('user','tweets','tweet') in self.social_graph.edge_types else None
                new_pair = torch.tensor([[u_idx],[t_idx]], dtype=torch.long)
                if old_ei is not None:
                    new_ei = torch.cat([old_ei, new_pair], dim=1)
                else:
                    new_ei = new_pair
                self.social_graph['user','tweets','tweet'].edge_index = new_ei

            else:  # 转发/回复
                old_ei = self.social_graph['user','retweets','tweet'].edge_index \
                         if ('user','retweets','tweet') in self.social_graph.edge_types else None
                new_pair = torch.tensor([[u_idx],[t_idx]], dtype=torch.long)
                if old_ei is not None:
                    new_ei = torch.cat([old_ei, new_pair], dim=1)
                else:
                    new_ei = new_pair
                self.social_graph['user','retweets','tweet'].edge_index = new_ei
    
    def update_person(self,
                       llm:ModelWrapperBase,
                       num_added:int = 5):
        prompt ="""
Your task is to give me a list of {num_added} person's profiles for twitter users . Respond in this format:
[
{{
"user_name": "(str;The name of this user)",
"user_description":"(str;short and concise, a general description of this user, ordinary users or super \
large users and the topics this person interested in)"
}}
]

Now please generate:
"""

        prompt = prompt.format(num_added = num_added)
        prompt_msg = llm.format(Msg("user",prompt,"user"))
        response = llm(prompt_msg)
        content = response.text
        followers_update = []
        ids_added = []
        
        try:
            # person_data_update = json.loads(content)
            person_data_update = find_and_load_json(content, "list")
            for idx,person in enumerate(person_data_update):
                try:
                    name = person["user_name"]
                    description = person["user_description"]
                    # followers = int(person["user_followers"])
                    followers_update.append({
                        "user_name":name,
                        "user_description":description,
                        # "user_followers":followers,
                        "follow":[],
                        "friend":[]
                    })

                except:continue
            followers_update = pd.DataFrame(followers_update)
            self.social_member_data = pd.concat([self.social_member_data,followers_update],ignore_index=True)
            ids_added = self.social_member_data.index.to_list()[-followers_update.shape[0]:]
            self.social_member_data['user_index'] = self.social_member_data.index
        except Exception as e:
            print(f"\n[CRITICAL ERROR] Failed to generate new person profiles in update_person.")
            print(f"Error Type: {type(e).__name__}, Message: {e}")
            print("LLM raw response content was:")
            print(content)
            traceback.print_exc()
            ids_added = []
            # pass
        return ids_added
    
    def save_infos(self,
                   cur_time:str, 
                   start_time):
        cur_time = datetime.strptime(cur_time,"%Y-%m-%d")
        pid = os.getpid()
        tid = threading.get_ident()
        print(f"[PID: {pid}, TID: {tid}] Saving logs at date: {cur_time}")
        data_dir = os.path.join(self.generated_data_dir,"data")
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        social_member_data_dir = os.path.join(data_dir,"social_network")
        if not os.path.exists(social_member_data_dir):
            os.makedirs(social_member_data_dir)
            
        save_path = os.path.join(social_member_data_dir,
                                    "social_member_data_{day}.csv".format(
                                        day = cur_time.strftime("%Y%m%d"))
                                     )            
        self.social_member_data.to_csv(save_path,index=False)
        
        self.forum_loader.save(os.path.join(data_dir,"forum.json"))
        
        action_logs_searlize = copy.deepcopy(self.action_logs)
        for ac_log in action_logs_searlize:
            ac_log[-1] = ac_log[-1].strftime("%Y-%m-%d")
        
        writeinfo(os.path.join(data_dir,"action_logs.json"),
                  action_logs_searlize)
        print(f"Writing transitive log of date {cur_time}")
        print(f"Current log list length: {len(self.transitive_agent_log)}")
        print(f"Log dates in memory: {[log.get('date') for log in self.transitive_agent_log]}")
        writeinfo(os.path.join(data_dir,"transitive_agent_log.json"),
                  self.transitive_agent_log)
        print(f"Writed transitive log of date {cur_time}")
        
        ex_logs_path = os.path.join(data_dir,"ex_logs.json")
        import time
        simulation_time = time.time() - start_time
        simulation_time += self.simulation_time
        
        if os.path.exists(ex_logs_path):
            ex_logs = readinfo(ex_logs_path)
            start_time = ex_logs["simulation_time"]
            this_round = int(simulation_time) - int(start_time)
            round_times = ex_logs.get("round_times",[])
            round_times.append(this_round)
        else:
            this_round = int(simulation_time)
            round_times = [this_round]

        user_indexs = self.social_member_data['user_index'].to_list()
        delete_indexs = []
        for  transitive_agent_log_ in self.transitive_agent_log:
            delete_indexs.extend(transitive_agent_log_["delete_ids"])

        user_indexs = list(filter(lambda x: x not in delete_indexs, user_indexs))
        ex_logs = {
            "last_added_time":self.last_added_time.strftime("%Y%m%d"),
            "simulation_time":int(simulation_time),
            "twitters":len(self.action_logs),
            "cur_user_num":len(user_indexs),
            "round_times":round_times
        }
        writeinfo(os.path.join(data_dir,"ex_logs.json"),
                  ex_logs)
        
    def get_follow_ids(self, 
                     agent_id):
        if isinstance(agent_id,str):
            agent_id = int(agent_id)

        follow_ids = self.social_member_data.loc[
            agent_id,
            "follow"]
        friend_ids = self.social_member_data.loc[
            agent_id,
            "friend"]

        return {
            "follow_ids":follow_ids,
            "friend_ids":friend_ids
        }
            
                
                
    def save_networkx_graph(self):
        pass
    
    
   

    def sample_cur_agents(self, 
                          cur_agent_ids:list = [],
                          sample_ratio:float = 0.1,
                          sample_big_name_ratio:float = 0.3):
        
        hot_agent_ids = copy.deepcopy(self.big_name_list)
        common_agent_ids = list(filter(lambda x: x not in hot_agent_ids, cur_agent_ids))
       
        common_num = min(max(int(sample_ratio*len(common_agent_ids)),5),
                         len(common_agent_ids))
        hot_num = min(max(int(sample_big_name_ratio*len(hot_agent_ids)),5),
                         len(hot_agent_ids))
        return random.sample(common_agent_ids,common_num), \
            random.sample(hot_agent_ids,hot_num)
    
    def sample_cur_agents_llmplan(self, 
                                agent_plans_map:dict ={}
                                ):
        """按照二项分布采样"""
        hot_agent_ids = []
        common_agent_ids = []
        import numpy as np
        for day, agent_ids in agent_plans_map.items():
            p = int(day)/30
            p = 1 if p > 1 else p
            sampled_list = np.array(agent_ids)[np.random.rand(len(agent_ids)) <= p]
            for agent_id in sampled_list:
                if agent_id in self.big_name_list:
                    hot_agent_ids.append(int(agent_id))
                else:
                    common_agent_ids.append(int(agent_id))
        return common_agent_ids, hot_agent_ids


    def get_user_num_followers(self, user_index:int):
        if isinstance(user_index,str):
            user_index = int(user_index)
        return self.follow_map.get(user_index,0)
    
    def update_big_name_list(self):
        """get big name list: follower count > threshold"""
        self.update_follow_map()
        threshold_ratio = self.control_profile.get("hub_rate",0.2)
        threshold = int(self.social_member_data.shape[0]*threshold_ratio)
        follower_count_list = [(agent_id, self.follow_map.get(agent_id,0))
                          for idx, agent_id
                          in enumerate(self.social_member_data.index)]
        social_member_filter = sorted(follower_count_list, key=lambda x: x[1], reverse=True)[:threshold]
        social_member_filter_ids = [x[0] for x in social_member_filter]
        self.big_name_list = social_member_filter_ids
        

    def update_follow_map(self):
        for user_index in self.social_member_data['user_index']:
            for follow_id in self.social_member_data.loc[user_index,
                                                     "follow"]:
                self.follow_map[follow_id] = self.follow_map.get(follow_id,0) + 1
            for friend_id in self.social_member_data.loc[user_index,
                                                     "friend"]:
                self.follow_map[friend_id] = self.follow_map.get(friend_id,0) + 1    
            

    def get_memory_init_kwargs(self, user_index:int):
        seen_tweets = []
        posted_tweets = []
        
        action_counts = {}
        posted_topics = []
        document_prompt = PromptTemplate.from_template("""
{tweet_idx}:
    user: {user_name}
    topic: {topic}
    tweet: {page_content}""")
        
        user_tweets = list(filter(lambda doc: doc.metadata["user_index"] == user_index,
                                  self.forum_loader.docs))
        actions = [tweet.metadata["action"] for tweet in user_tweets]
        for tweet in user_tweets:
            posted_tweets.append(document_prompt.format(**tweet.metadata,
                                                        page_content=tweet.page_content))
            posted_topics.append(tweet.metadata["topic"])
        
        topic_memory = {
            "posted_topics":posted_topics,
            "followed_topics":[]
        } 

        action_counter = Counter(actions)
        action_counts = dict(action_counter.items()) 
        return {
            "seen_tweets":seen_tweets,
            "posted_tweets":posted_tweets,
            "topic_memory":topic_memory,
            "action_counts":action_counts
        }
    
    def get_follower_description(self,user_index:str):
        if isinstance(user_index,str):
            user_index = int(user_index)
        template = """
Here's the list of people you followed:

{user_names}
"""
        followed_agent_ids = self.social_member_data.loc[user_index,
                                                     "follow"]
        followed_agent_names = [
            self.social_member_data.loc[followed_agent_id,
                                        "user_name"]
            for followed_agent_id in followed_agent_ids
        ]
        return template.format(user_names =
                               ",".join(followed_agent_names))
    
    def get_user_big_name(self,user_index:str):
        if isinstance(user_index,str):
            user_index = int(user_index)
        big_name = user_index in self.big_name_list
        return big_name
    
    def return_deleted_agent_ids(self):
        delete_ids = []
        for transitive_agent_log in self.transitive_agent_log:
            delete_ids.extend(transitive_agent_log["delete_ids"])
        return delete_ids
    
    def plot_agent_plan_distribution(self,
                                     cur_time:str,
                                     sampled_agent_ids:dict = {},
                                     agent_plans_map:dict = {}):
        data_dir = os.path.join(self.generated_data_dir,"data","plans")
        os.makedirs(data_dir,exist_ok=True)
        from LLMGraph.utils.io import writeinfo
        data_dir = os.path.join(data_dir,f"agent_plans_{cur_time}.json")
        writeinfo(data_dir,
                  {"sampled_agent_ids":sampled_agent_ids,
                   "agent_plans_map":agent_plans_map,
                   "big_name_list":self.big_name_list})
        
        ## plot fig
        # data_dir = os.path.join(data_dir,f"agent_plans_{cur_time}.pdf")
        import matplotlib.pyplot as plt
        

        # # 绘制折线图
        # plt.figure(figsize=(10, 5))  # 可以调整图像的大小
        # plt.plot(days, plan_lengths, marker='o')  # 使用'o'标记每个点

        # # 添加标题和轴标签
        # plt.ylabel('Agent Number')
        # plt.xlabel('Activity Frequency')

        # # 显示图形
        # plt.xticks(rotation=45)  # 可能需要旋转x轴标签，以防它们互相重叠
        # plt.grid(True)  # 显示网格线
        # plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
        # plt.savefig(data_dir)

    def _update_user_graph(self):
        """
        重新构建整个社交图，使其只包含 active_users 及这些用户发布的推文。
        """
        # 1) 丢掉旧的 social_graph
        self.social_graph = None
        self._idx_to_doc = {}

        # 2) 准备最新的“活跃用户”对应的 DataFrame
        df_active = self.social_member_data.loc[
            self.social_member_data["user_index"].isin(self.active_users)
        ]

        # 3) 准备最新的“活跃用户”的推文列表
        all_docs = self.forum_loader.load()
        docs_active = []
        for doc in all_docs:
            uid = int(doc.metadata.get("user_index", -1))
            if uid in self.active_users:
                docs_active.append(doc)

        # 4) 调用 build_social_graph，仅针对这两部分
        graph_data, idx_to_doc = build_social_graph(df_active, docs_active)

        # 5) 覆盖保存
        self.social_graph = graph_data
        self._idx_to_doc = idx_to_doc