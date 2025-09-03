

from typing import List, Union
import json

from loguru import logger
from LLMGraph.manager import MovieManager


from . import env_registry as EnvironmentRegistry
from .base import BaseEnvironment
import copy
import os
import random
from agentscope.message import Msg
from LLMGraph.agent.movie import MovieAgent, MovieManagerAgent
from LLMGraph.wrapper import MovieAgentWrapper
from datetime import datetime, timedelta
from agentscope.agents.rpc_agent import RpcAgentServerLauncher,RpcAgent
from LLMGraph.utils.timing import timed
from LLMGraph.wrapper.agent_group import MovieGroupAgent
 
@EnvironmentRegistry.register("movie")
class MovieEnvironment(BaseEnvironment):
    """
    A environment implementing the logic of conversation.
    
    Args:
        agents: tenant_manager
        rule: Rule for the environment
        max_turns: Maximum number of turns
        cnt_turn: Current turn number
        last_messages: Messages from last turn
        rule_params: Variables set by the rule
    """

    cur_agents:dict = {} # 存储所有agent
    
    agent_configs:dict = {} # 存储agent的config
    
    movie_rate_configs:dict = {
        "watch_plan":"SYSTEM",
        "min_rate_all": 200 ,# 总体最少做多少个评价
        }
    
    time_configs:dict ={
        "start_time": datetime.strptime("1997-01-01", '%Y-%m-%d'),
        "cur_time": datetime.strptime("1997-01-01", '%Y-%m-%d'),
        "end_time": datetime.strptime("2001-12-31", '%Y-%m-%d'),
        "movie_time_delta": timedelta(days=4*30) ,# 4个月,
        "watcher_time_delta": timedelta(days=30),
        "watcher_add": False, # 所有时间点都是 n个watcher进行rate
        # 如果是true,则按照watcher_time_delta分批进入
    }
    
    movie_group_configs: dict = {
        "group_participation_rate": 0.4,
        "group_size_range": (2, 5)
    }
    
    cur_rate:int = 0
    
    movie_agents:dict = {} # id: movie_agent 存储所有的agent 暂时没用到
    
    class Config:
        arbitrary_types_allowed = True

    @timed
    def __init__(self,
                 launcher_args:list = [],
                 **kwargs):
        to_dist = len(launcher_args) > 0
        if to_dist:
            to_dist_kwargs = {
                "to_dist":{
                    "host":launcher_args[0]["host"],
                    "port":launcher_args[0]["port"]
                }
            }
        else:
            to_dist_kwargs = {}
            
        movie_manager_configs = kwargs.pop("managers").pop("movie")
        task_path = kwargs.pop("task_path")
        config_path = kwargs.pop("config_path")
        agent_configs = kwargs.pop("agent")
        movie_data_dir = os.path.join(task_path,
                                      movie_manager_configs.pop("movie_data_dir"))
       
        link_movie_path = os.path.join(task_path,
                                          movie_manager_configs.pop("link_movie_path"))
        generated_data_dir = os.path.join(os.path.dirname(config_path),
                                          movie_manager_configs.pop("generated_data_dir"))
        
        time_configs = kwargs.pop("time_configs",{})
        time_configs["movie_time_delta"] = timedelta(days=30*time_configs["movie_time_delta"])
        time_configs["watcher_time_delta"] = timedelta(days=30*time_configs["watcher_time_delta"])
        
        manager_agent = MovieManagerAgent(
            name = "movie_manager",
            model_config_name="default",
            movie_data_dir = movie_data_dir,
            link_movie_path=link_movie_path,
            generated_data_dir=generated_data_dir,
            movie_manager_configs = movie_manager_configs,
            start_time = time_configs["start_time"].strftime("%Y-%m-%d"),
            cur_time = time_configs["cur_time"].strftime("%Y-%m-%d"),
            movie_time_delta = time_configs["movie_time_delta"].days,
            **copy.deepcopy(to_dist_kwargs)
        )
        movie_rate_configs = kwargs.pop("movie_rate_configs")
    

        call_func_msg = Msg("user",
                            role="user",
                            func = "load_history",
                            content="call_function")
        return_value = manager_agent(call_func_msg).content
        if return_value is not None:
            cur_time = return_value.get("cur_time",time_configs["cur_time"]) 
            cur_rate = return_value.get("cur_rate",0)
            if isinstance(cur_time,str):
                time_configs["cur_time"] = datetime.strptime(cur_time, '%Y-%m-%d').date()
        else:
            cur_rate = 0
            
        print(f"Generated ratings number: {cur_rate}")
        movie_group_configs = kwargs.pop("movie_group_configs", {})
        super().__init__(manager_agent = manager_agent,
                         agent_configs = agent_configs,
                         movie_rate_configs = movie_rate_configs,
                         cur_rate = cur_rate,
                         time_configs = time_configs,
                         to_dist = to_dist, 
                         launcher_args = launcher_args,
                         movie_group_configs=movie_group_configs,
                         **kwargs)
        
    @timed
    def initialize(self):
        self.update_agents()
        
    def reset(self) -> None:
        """Reset the environment"""
        self.cnt_turn = 0

    def is_done(self) -> bool:
        """Check if the environment is done"""
        """True: Done"""
        
        return self.cur_rate >= self.movie_rate_configs["min_rate_all"] \
            or self.time_configs["cur_time"] >= self.time_configs["end_time"]
   
            
    @timed
    def rate_movie(self):
        def run_parallel():
            if len(self.cur_agents)==0: return {}
            rating_msgs:List[Msg] =[]
            for agent_id,agent in list(self.cur_agents.items()):
                rate_msg = self.call_agent_func(agent, 
                                                 "rate_movie_process", 
                                        kwargs = {"cur_time":self.time_configs["cur_time"].strftime("%Y-%m-%d"),
                                                "max_retry_rate":2})
                rating_msgs.append(rate_msg)

            ratings = []
            thunk_size = len(self.launcher_args)
            if thunk_size == 0:
                thunk_size = 1
            for i in range(0,len(rating_msgs),thunk_size):
                thunk = rating_msgs[i:i+thunk_size]
                ratings_sub = [msg.content for msg in thunk]
                ratings.extend(ratings_sub)
            # ratings = [rating_msg.content for rating_msg in rating_msgs]
            return ratings
        
        return run_parallel() # 需要进行讨论的tenant
    
    @timed
    def update_time(self):
        self.time_configs["cur_time"] = self.time_configs["cur_time"] + self.time_configs["watcher_time_delta"]
        if isinstance(self.time_configs["cur_time"],datetime):
            self.time_configs["cur_time"] = self.time_configs["cur_time"].date()
        
    
    
    @timed
    def init_agent(self, 
                   name: str,
                   infos:dict,  
                   launcher_id = 0,
                   ) -> MovieAgentWrapper:
        rating_counts_id = self.call_manager_agent_func(
                "get_watcher_rating_infos",
                kwargs={
                    "watcher_id":name,
                }
            ).content
        if self.to_dist:
            launcher_arg = self.launcher_args[launcher_id]
            to_dist_kwargs = {
                "to_dist":{
                    "host":launcher_arg["host"],
                    "port":launcher_arg["port"]
                }
            }
        else:
            to_dist_kwargs = {}
        
        agent_rating_counts = self.call_manager_agent_func(
            "get_rating_counts",
            kwargs={
                "rating_counts_id": rating_counts_id
            }
        ).content
        
        agent = MovieAgent(name = name,
                           infos = infos,
                            agent_configs=self.agent_configs,
                            rating_counts=agent_rating_counts["rating_counts"],
                            ratings=agent_rating_counts["ratings"],    
                            **copy.deepcopy(to_dist_kwargs))

        wrapper_agent = MovieAgentWrapper(
                            name = name,
                            agent = agent,
                            manager_agent = self.manager_agent,
                            max_tool_iters = 2,
                            max_retrys = 3,
                            **copy.deepcopy(to_dist_kwargs))
        return wrapper_agent

    @timed
    def update_agents(self):
        """initialize the agents and plans"""
        
        agent_profiles = self.call_manager_agent_func(
                func_name="add_and_return_watcher_profiles",
                kwargs = {
        "cur_time":self.time_configs["cur_time"].strftime("%Y-%m-%d"),
        "watcher_add":self.time_configs["watcher_add"],
        "watcher_num":self.time_configs["watcher_num"]
                }).content

        agents_added = []
        if len(agent_profiles) == 0 :return

        for idx,agent_profile in enumerate(agent_profiles):
            assert agent_profile["id"] not in self.cur_agents.keys(),"error!"
            

            if self.to_dist:
                launcher_id = idx%len(self.launcher_args)
            else:
                launcher_id = 0
            agent = self.init_agent(agent_profile["id"], 
                                    agent_profile, 
                                    launcher_id=launcher_id,)
           
            agents_added.append(agent)
            self.cur_agents[agent_profile["id"]] = agent
        
        def run_parallel(task,
                         len_v:int,
                        agents,
                        requirement:str = "You should watch a lot of movies and rate them."):
            watch_msgs :List[Msg] = []
            for agent in agents:
                # system/llm control plan
                watch_msg = self.call_agent_func(agent, 
                                                 "get_movie_watch_plan", 
                                        kwargs = {"task":task,
                                                "len_v":len_v,
                                                "use_default_plan":self.movie_rate_configs.get("watch_plan",
                                                                                               "SYSTEM")== "SYSTEM",
                                                "requirement":requirement})
                
                watch_msgs.append(watch_msg)
            
            # 堵塞
            [watch_msg.content for watch_msg in watch_msgs]
        
        
        number_years = (self.time_configs["end_time"] - self.time_configs["cur_time"])\
            //timedelta(days=365)
        round_delta = self.time_configs["watcher_time_delta"]
        
        template = """You should make a plan to watch a few movies every {month_num} \
months for the next {year_num} years, \
You can allocate your time reasonably without going to see it every month.\
So you need to give your plan, and respond in the format of a vector with a length of {len_v}."""
        month_num = round_delta//timedelta(days=30)
        len_v = number_years* (12//month_num + 1)# 为了防止越界
        
        task = template.format(month_num = month_num,
                               len_v = len_v,
                               year_num = number_years)
        run_parallel(task, len_v, agents_added) # 需要进行讨论的tenant
        
        
    @timed
    def step(self):
        
        self.call_manager_agent_func(
            "add_movies",
            kwargs={
                "cur_time":self.time_configs["cur_time"].strftime("%Y-%m-%d"),
                "movie_time_delta":self.time_configs["movie_time_delta"].days,
            }
        ).content
        print("Add movie is over")
        self.call_manager_agent_func("update").content

        # self.update_agents() # 一开始加入所有agent
        # rate movie process
        all_ratings_this_turn = []
        all_agent_ids = list(self.cur_agents.keys())
        
        # get group
        groups, ungrouped_agent_ids = self.call_manager_agent_func(
            "form_interest_groups",
            kwargs={
                "all_agent_ids": all_agent_ids,
                "group_size_range": self.movie_group_configs["group_size_range"],
                "group_participation_rate": self.movie_group_configs["group_participation_rate"]
            }
        ).content
        print(f"Get Groups: {len(groups)}, solo: {len(ungrouped_agent_ids)}")
        
        # group action
        group_agents = []
        for g_id, group_ids in enumerate(groups):
            if self.to_dist:
                # 根据 g_id 进行轮询分配
                launcher_arg = self.launcher_args[g_id % len(self.launcher_args)]
                to_dist_kwargs = {
                    "to_dist": {
                        "host": launcher_arg["host"],
                        "port": launcher_arg["port"]
                    }
                }
            else:
                to_dist_kwargs = {}
            group_members = [self.cur_agents[gid] for gid in group_ids]
            group_agent = MovieGroupAgent(
                name=f"MovieGroup_{g_id}",
                agents=group_members,
                manager_agent=self.manager_agent,
                **copy.deepcopy(to_dist_kwargs)
            )
            group_agents.append(group_agent)
        logger.info(f"Groups initialized!")

        if len(group_agents)==0: return []
        all_group_results = []
        for idx in range(0, len(group_agents), 20):
            sub_group_agents = group_agents[idx:idx+20]
            sub_msgs = [
                group_agent(Msg(
                            name="user",
                            content="call_function",
                            role="assistant",
                            func="group_activity",
                            kwargs={"cur_time": self.time_configs["cur_time"].strftime("%Y-%m-%d")}
                        ))
                for group_agent in sub_group_agents
            ]

            for m in sub_msgs:
                all_group_results.extend(m.content or {})

        logger.info(f"Group comms finished!")
        # solo action
        solo_results = []
        for idx in range(0, len(ungrouped_agent_ids), 20):
            sub_ungrouped_agents = ungrouped_agent_ids[idx:idx+20]
            sub_msgs = [
                self.call_agent_func(
                    self.cur_agents[agent_id], 
                    "rate_movie_process", 
                    kwargs={"cur_time": self.time_configs["cur_time"].strftime("%Y-%m-%d")}
                )
                for agent_id in sub_ungrouped_agents
            ]

            for m in sub_msgs:
                solo_results.extend(m.content or {})

        all_ratings_this_turn = all_group_results + solo_results

        
        # update rating DB
        print(f"--- TIME: {self.time_configs['cur_time']} ---")
        print(f"Total ratings collected in this step: {len(all_ratings_this_turn)}")
        if len(all_ratings_this_turn) > 0:
            print(f"First 3 ratings: {all_ratings_this_turn[:3]}")
        self.update_movie_manager(all_ratings_this_turn)
                
        # update movie/watcher DB and Time
        self.update_time()

        ### testing
        logger.info("Step Finished! Now exiting...")
        # exit(0)
        
    @timed
    def update_movie_manager(self,
                             ratings
                             ):
        num = self.call_manager_agent_func("update_db_ratings",
                                           kwargs={
                                               "ratings":ratings,
                                            #    "agent_ids":list(self.cur_agents.keys()),
                                           }).content
        self.cur_rate += num
        logger.info(f"Current iter update db ratings num: {num}, cur_rate: {self.cur_rate}")
        
        
    
    def save(self, start_time):
        self.call_manager_agent_func(
            "save",
            kwargs={
                "cur_time":self.time_configs["cur_time"].strftime("%Y-%m-%d"),
                "start_time":start_time
            }
        )

        
    def test(self):
        """define unittest"""
        pass

    def eval(self):
        pass