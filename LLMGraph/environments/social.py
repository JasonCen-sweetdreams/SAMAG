import asyncio

from typing import List, Any, Union

from LLMGraph.agent.social import SocialAgent,SocialManagerAgent
from LLMGraph.wrapper import agent_group
from LLMGraph.wrapper.social import SocialAgentWrapper
from LLMGraph.wrapper.agent_group import SocialGroupAgent

import json
from LLMGraph.manager import SocialManager
import pandas as pd
from . import env_registry as EnvironmentRegistry
from .base import BaseEnvironment
import copy
import os
import random
from dateutil.relativedelta import relativedelta
from LLMGraph.utils.timing import timed
import traceback

from agentscope.agents.rpc_agent import RpcAgentServerLauncher
from datetime import datetime, timedelta
from agentscope.message import Msg, PlaceholderMessage

@EnvironmentRegistry.register("social")
class SocialEnvironment(BaseEnvironment):
    """social environment for twitter

    Args:
        BaseEnvironment (_type_): _description_

    Returns:
        _type_: _description_
    """
   
    cur_agents:dict = {} # 存储所有agent

    sampled_agent_ids:dict ={
        "big_name":[],
        "common":[]
    }
    
    agent_configs:dict # 存储agent的config
    
    time_configs:dict ={
        "start_time": datetime.strptime("1997-01-01", '%Y-%m-%d'),
        "cur_time": datetime.strptime("1997-01-01", '%Y-%m-%d'),
        "end_time": datetime.strptime("2001-12-31", '%Y-%m-%d'),
        "social_time_delta": timedelta(days = 1) ,# 一天
    }
    
    social_configs:dict ={
        "max_people":100,
        "add_people_rate":0.1,
        "delete_people_rate":0.1,
        "group_rate":0.3
    }
    
    class Config:
        arbitrary_types_allowed = True

    def __init__(self,
                 launcher_args=[],
                 **kwargs):
        to_dist = len(launcher_args) > 0
        social_manager_configs = kwargs.pop("managers").pop("social")
        task_path = kwargs.pop("task_path")
        config_path = kwargs.pop("config_path")
        agent_configs = kwargs.pop("agent")
        social_data_dir = os.path.join(task_path,
                                      social_manager_configs.pop("social_data_dir"))
       
        generated_data_dir = os.path.join(os.path.dirname(config_path),
                                          social_manager_configs.pop("generated_data_dir"))
        
        time_configs = kwargs.pop("time_configs",{})
        time_configs["social_time_delta"] = timedelta(days=time_configs["social_time_delta"])
        time_configs["people_add_delta"] = timedelta(days=time_configs["people_add_delta"])

       
        if to_dist:
            to_dist_kwargs = {
                "to_dist":{
                    "host":launcher_args[0]["host"],
                    "port":launcher_args[0]["port"]
                }
            }
        else:
            to_dist_kwargs = {}

        manager_agent = SocialManagerAgent( # pylint: disable=E1123
                                            name = "socialmanager",
                                           social_data_dir = social_data_dir,
                                            generated_data_dir = generated_data_dir,
                                            social_manager_configs=social_manager_configs,
                                            cur_time=time_configs["cur_time"].strftime("%Y-%m-%d"),
                                            # to_dist = {
                                            #         "host":"localhost",
                                            #         "port":"2333"
                                            #     } 
                                            **copy.deepcopy(to_dist_kwargs)
                                            )
        cur_time = manager_agent(
            Msg("user",
                content="get_start_time",
                kwargs={},
                func="get_start_time"
            )
        ).content
        
        cur_time = datetime.strptime(cur_time,"%Y-%m-%d").date()
        time_configs["cur_time"] = cur_time

        super().__init__(manager_agent = manager_agent,
                         agent_configs = agent_configs,
                         time_configs = time_configs,
                         launcher_args = launcher_args,
                         to_dist = to_dist,
                         **kwargs
                         )

    
    
        
    def reset(self) -> None:
        """Reset the environment"""
        pass

    def is_done(self) -> bool:
        """Check if the environment is done"""
        """True: Done"""
        
        return self.time_configs["cur_time"] >= self.time_configs["end_time"] or \
        len(self.cur_agents) < 1
   
    @timed
    def social_one_agent(self,
                        agent,
                        big_name:bool = False) -> Msg:
                                      
        """
        the async run parse of tenant(tenant_id) communication.
        return: the receivers, self(if continue_communication)
        """
        # assert isinstance(agent, SocialAgentWrapper)
        social_content_msg = self.call_agent_func(agent,
                                            "twitter_process",
                                            kwargs={
                                                "cur_time":self.time_configs["cur_time"].strftime("%Y-%m-%d"),
                                                "big_name": big_name
                                            })
        return social_content_msg
    
    @timed
    def social_one_group(self,
                         group_agent: SocialGroupAgent,
                         cur_time_str: str) -> Msg:
        """
        Args:
            group_agent (SocialGroupAgent): The group agent instance to communicate with.
            cur_time_str (str): The current time formatted as a string ("%Y-%m-%d").
            
        Returns:
            Msg: The message containing the results from the group communication.
        """
        msg = Msg(
            name="user",
            content="call_function",
            role="assistant",
            func="communication",
            kwargs={"cur_time": cur_time_str}
        )

        result_msg = group_agent(msg)
        return result_msg

    @timed
    def social(self,
               agent_plans_map:dict = {}):
        """按照随机采样"""
        # common_agent_ids, hot_agent_ids = self.call_manager_agent_func(
        #     "sample_cur_agents",
        #                             kwargs ={
        #                             "cur_agent_ids":list(self.cur_agents.keys()),
        #                             "sample_ratio":0.1,
        #                             "sample_big_name_ratio":0.3
        #                             }).content

        """按照llm生成的概率采样"""
        
        common_agent_ids, hot_agent_ids = self.call_manager_agent_func(
            "sample_cur_agents_llmplan",
            kwargs ={
            "agent_plans_map": agent_plans_map
            }).content
    
        self.sampled_agent_ids["big_name"] = hot_agent_ids
        self.sampled_agent_ids["common"] = common_agent_ids
        
        self.call_manager_agent_func(
            "plot_agent_plan_distribution",
            kwargs={
                "cur_time": datetime.strftime(self.time_configs["cur_time"],"%Y-%m-%d"),
                "agent_plans_map": agent_plans_map,
                "sampled_agent_ids": self.sampled_agent_ids
            }
        ).content
        
        all_active = hot_agent_ids + common_agent_ids
        print(f"Current iter active agents: {len(all_active)}")
        if len(all_active) == 0:
            return []

        proportion = self.social_configs.get("group_rate", 0.2)
        pool_size = max(int(len(all_active) * proportion), 0)
        if pool_size < 3:
            group_pool = []
            remaining = [aid for aid in all_active]
        else:
            ### 随机抽取
            group_pool = random.sample(all_active, pool_size)
            remaining = [aid for aid in all_active if aid not in group_pool]
        random.shuffle(group_pool)
        groups = []
        i = 0
        N = len(group_pool)
        while i < N:
        # 如果剩余不足 3 人，就把剩下的每个人单独成一组
            if N - i < 3:
                for aid in group_pool[i:]:
                    groups.append([aid])
                break

            # 组大小
            # size = random.randint(3, 5)
            ### testing：固定
            size = 3
            # 防止越界
            if i + size > N:
                size = N - i
                if size < 3:
                    # 不足 3，就在下一个循环把它们当成单独组
                    continue

            groups.append(group_pool[i:i + size])
            i += size
        
        for aid in remaining:
            groups.append([aid])

        group_agents = []
        for g_id, member_list in enumerate(groups):
            ### 并行参数
            if self.to_dist:
                # 简单起见，可以让所有group agent都使用第一个launcher的配置
                # 或者根据 g_id 进行轮询分配
                launcher_arg = self.launcher_args[g_id % len(self.launcher_args)]
                to_dist_kwargs = {
                    "to_dist": {
                        "host": launcher_arg["host"],
                        "port": launcher_arg["port"]
                    }
                }
            else:
                to_dist_kwargs = {}
            wrappers = [(self.cur_agents[agent_id], agent_id in hot_agent_ids) for agent_id in member_list]
            leader_idx = None
            for idx_, aid in enumerate(member_list):
                if aid in hot_agent_ids:
                    leader_idx = idx_
                    break
            if leader_idx is None:
                leader_idx = 0
            # 把 leader 放在第 0 位
            if leader_idx != 0:
                wrappers[0], wrappers[leader_idx] = wrappers[leader_idx], wrappers[0]
            group_agent_name = "Group_%d" % g_id
            grp_agent = SocialGroupAgent(
                name = group_agent_name,
                agents = wrappers,
                manager_agent = self.manager_agent,
                ### 传分布式参数
                **copy.deepcopy(to_dist_kwargs))
            group_agents.append(grp_agent)

        cur_time_str = self.time_configs["cur_time"].strftime("%Y-%m-%d")
        print(f"Total group count is: {len(group_agents)}")
        
        def run_parallel(time_out_seconds = 300):
            if len(group_agents)==0: return []

            all_group_results = []
            for idx in range(0, len(group_agents), 20):
                sub_group_agents = group_agents[idx:idx+20]
                sub_msgs = [
                    self.social_one_group(grp, cur_time_str)
                    for grp in sub_group_agents
                ]
                
                for m in sub_msgs:
                    all_group_results.extend(m.content or {})
            return all_group_results
        
        return run_parallel() # 需要进行讨论的tenant
    
    @timed
    def collect_agent_plans(self):
        def run_parallel():
            if len(self.cur_agents)==0: return []

            cur_agents_keys = list(self.cur_agents.keys())
            plans_content = []

            for idx in range(0, len(self.cur_agents), 100):
                sub_plans = []
                cur_agents_keys_sub = cur_agents_keys[idx:idx+100]
                for agent_id in cur_agents_keys_sub:
                    agent = self.cur_agents[agent_id]
                    msg = self.call_agent_func(agent,
                                     "get_acting_plan"
                                     )
                    sub_plans.append(msg)

                sub_plans_content = [plan.content for plan in sub_plans]
                plans_content.extend(sub_plans_content)

            for content in plans_content:
                assert content is not None
            return plans_content
        
        plans_all_agent = run_parallel()
        agent_plans_map = {i:[] for i in range(1, 31)}
        for plan_agent, agent_id in zip(plans_all_agent, 
                                        self.cur_agents.keys()):
            log_in_days = plan_agent.get("log_in_days", 8)
            agent_plans_map[log_in_days].append(agent_id)

        return agent_plans_map
    
    def update_time(self):
        self.time_configs["cur_time"] = self.time_configs["cur_time"] \
            + self.time_configs["social_time_delta"]
        if isinstance(self.time_configs["cur_time"], datetime):
            self.time_configs["cur_time"] = self.time_configs["cur_time"].date()

    def init_agent(self, 
                   name:Union[int, str],
                   infos:dict,  
                   launcher_id = 0) -> SocialAgentWrapper:
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

        memory_init_kwargs = self.call_manager_agent_func(
            "get_memory_init_kwargs",
            kwargs={
                "user_index":name
            }
        ).content

        agent = SocialAgent(name = name,
                            infos = infos,
                            agent_configs=self.agent_configs,
                            memory_init_kwargs = memory_init_kwargs,
                            **copy.deepcopy(to_dist_kwargs))

        wrapper_agent = SocialAgentWrapper(
                            name = name,
                            agent = agent,
                            manager_agent = self.manager_agent,
                            max_tool_iters = 1,
                            **copy.deepcopy(to_dist_kwargs))
        return wrapper_agent
    
    @timed
    def update_agents(self,
                      denote_transitive_log:bool = True):
        """initialize the agents and plans"""
        num_added = int(len(self.cur_agents) * \
            self.social_configs["add_people_rate"])
        
        num_deleted = int(len(self.cur_agents) * \
            self.social_configs["delete_people_rate"])
        
        print("\n" + "="*80)
        print(f"DEBUG: update_agents() called for date: {self.time_configs['cur_time'].strftime('%Y-%m-%d')}")
        current_agent_count = len(self.cur_agents)
        add_rate = self.social_configs["add_people_rate"]
        raw_num_added = current_agent_count * add_rate
        print(f"  - Current agent count: {current_agent_count}")
        print(f"  - Add people rate: {add_rate}")
        print(f"  - Raw calculation for num_added (count * rate): {raw_num_added:.4f}")
        print(f"  - Final num_added (after int()): {num_added}")
        print(f"  - Final num_deleted: {num_deleted}")
        if num_added == 0 and current_agent_count > 0:
            print("  - [!!!] WARNING: num_added calculated as 0. Log entry for this round will likely be SKIPPED.")
        print("="*80 + "\n")

        delete_agent_ids = self.call_manager_agent_func(
            "delete_user_profiles",                                                                 
            kwargs={
                "cur_time":self.time_configs["cur_time"].strftime("%Y-%m-%d"),
                "add_user_time_delta":self.time_configs["people_add_delta"].days,
                "num_delete":num_deleted
            }
        ).content

        for cur_agent_id in delete_agent_ids:
            if cur_agent_id in self.cur_agents.keys():
                self.cur_agents.pop(cur_agent_id)
                print(f"Agent {cur_agent_id} deleted")
        
        agent_profiles = None
        cur_time = self.time_configs["cur_time"].strftime("%Y-%m-%d")
        """add agents (num_added)"""
        if num_added == 0 and len(self.cur_agents)==0:
            agent_profiles_df_list =[]

            max_iter = 1 # social_member_size / threshold
            for i in range(max_iter):
                agent_profiles_cur = self.call_manager_agent_func(
                    "add_and_return_user_profiles",
                    kwargs={
                        "cur_time":self.time_configs["cur_time"].strftime("%Y-%m-%d"),
                        "add_user_time_delta":self.time_configs["people_add_delta"].days,
                        "num_added":num_added
                    }
                ).content
                if len(agent_profiles_cur) =={}:break
                agent_profiles_df = pd.DataFrame.from_dict(agent_profiles_cur)
                agent_profiles_df_list.append(agent_profiles_df)
                print(f"updated {agent_profiles_df.shape[0]} agent profiles")
                
            agent_profiles = pd.concat(agent_profiles_df_list)

        else:
            agent_profiles_dfs = []
            step = num_added if num_added < 20 else 20
            step = 5 if step < 5 else step
            added_num = 0
            for i in range(0,num_added,step):
                add_num_per_round = step
                assert add_num_per_round <= 20, f"error add_num_per_round:{step}, {add_num_per_round}"
                agent_profiles_ = self.call_manager_agent_func(
                    "add_and_return_user_profiles",
                    kwargs={
                        "cur_time":self.time_configs["cur_time"].strftime("%Y-%m-%d"),
                        "add_user_time_delta":self.time_configs["people_add_delta"].days,
                        "num_added":add_num_per_round
                    }
                ).content
                # cur_time = self.time_configs["cur_time"].strftime("%Y-%m-%d")
                if len(agent_profiles_) ==0:
                    print(f"Current adding step {i} not adding any users! TIME: {cur_time}")
                    break
                agent_profiles_df = pd.DataFrame.from_dict(agent_profiles_)
                added_num += agent_profiles_df.shape[0]
                agent_profiles_dfs.append(agent_profiles_df)
                if added_num > num_added:
                    break
            if len(agent_profiles_dfs)==0:
                print(f"agent_profiles_dfs dont have anything! TIME: {cur_time}")
                return
            agent_profiles = pd.concat(agent_profiles_dfs)

        self.call_manager_agent_func(
            "update_add_user_time",
            kwargs={
                "cur_time":self.time_configs["cur_time"].strftime("%Y-%m-%d")
            }
        ).content
        
        agents_added = []
        if agent_profiles is None or agent_profiles.shape[0]==0: return

        for idx, agent_profile_info in enumerate(agent_profiles.iterrows()):
            index, agent_profile = agent_profile_info
            user_index = agent_profile["user_index"]
            # assert agent_profile["user_index"] not in self.cur_agents.keys(),\
            #       f"{index}, {user_index}"
            if user_index in self.cur_agents.keys():
                continue
            if self.to_dist:
                launcher_id = idx%len(self.launcher_args)
            else:
                launcher_id = 0
            agent = self.init_agent(agent_profile["user_index"],
                                   agent_profile.to_dict(),
                                   launcher_id = launcher_id)
            agents_added.append(agent)
            self.cur_agents[agent_profile["user_index"]] = agent
            if idx % 100 == 0:
                print(f"Agent {agent_profile['user_index']} added")
        print(f"Agent number after add and delete of the current iter: {len(self.cur_agents)}. TIME: {cur_time}")
        
        if denote_transitive_log:
            self.call_manager_agent_func("denote_transitive_log",
                                     kwargs={
                                         "delete_ids":delete_agent_ids,
                                         "add_ids":agent_profiles["user_index"].to_list(),
                                         "cur_time":self.time_configs["cur_time"].strftime("%Y-%m-%d")
                                     })
        
            
        
    @timed
    def step(self):
        print(f"Current iter beginning at {datetime.now()}...\n", flush=True)
        if self.time_configs["people_add_delta"] > timedelta(days=0):
            # 暂时不往网络内添加agent
            self.update_agents()
        
        print(f"Adopting llm generated plans at {datetime.now()}...\n", flush=True)
        """adopt llm generated plans"""
        agent_plans_map = self.collect_agent_plans()
        print(f"Social at {datetime.now()}\n", flush=True)
        twitters = self.social(agent_plans_map)
        
        # update rating DB
        print(f"Updating database at {datetime.now()}...", flush=True)
        add_num = self.update_social_manager(twitters)
        self.call_manager_agent_func("update_big_name_list").content

        print(f"added {add_num} twitters for {self.time_configs['cur_time'].strftime('%Y-%m-%d')} at reality time: {datetime.now()}", flush=True)
        # update social/watcher DB and Time
        self.update_time()
        
    @timed
    def update_social_manager(self,
                             all_twitter_actions:list = []
                             ):
        print(f"Adding {len(all_twitter_actions)} twitters")
        print(f"All twitters: \n{all_twitter_actions}")
        num  = 0
        # agent_ids = [*self.sampled_agent_ids["big_name"],
        #              *self.sampled_agent_ids["common"]
        #              ]
        # zipped_agent_twitters = list(zip(twitters,agent_ids))
        
        # for idx in range(0, len(zipped_agent_twitters), 100):
        #     zipped_agent_twitters_sub = zipped_agent_twitters[idx:idx+100]
        # print(f"All zipped_agent_twitters: {zipped_agent_twitters[:-3]}")
        # print(f"total twitters length: {len(zipped_agent_twitters)}")
        # for idx in range(0, len(zipped_agent_twitters), 100):
        for idx in range(0, len(all_twitter_actions), 100):
            sub_actions = all_twitter_actions[idx:idx + 100]

            # 对当前 100 条（或不足 100 条）的子列表，先构造好一个 data_list
            data_list = []
            current_date_str = self.time_configs["cur_time"].strftime("%Y-%m-%d")
            for action_dict in sub_actions:
                if action_dict:
                    agent_id = action_dict.get("agent_id")
                    if agent_id is None:
                        print(f"Warning: Found an action without agent_id: {action_dict}")
                        continue
                    data_list.append({
                        "agent_id": agent_id,
                        "cur_time": current_date_str,
                        "twitters": action_dict
                    })

            # 如果这一轮子批次里没人发推，则跳过
            if not data_list:
                continue

            # 一次性调用批量接口 add_tweets_batch，返回这一子批次写入数
            add_msg = self.call_manager_agent_func(
                "add_tweets_batch",
                kwargs={"data_list": data_list}
            )
            sub_count = int(add_msg.content)
            num += sub_count
            print(f"  Sub-batch {idx // 100 + 1}: added {sub_count} tweets.")
        # for idx in range(0, len(zipped_agent_twitters), 100):
        #     zipped_agent_twitters_sub = zipped_agent_twitters[idx:idx+100]
        #     sub_add_msgs = []
        #     for twitters_one, agent_id in zipped_agent_twitters_sub:
        #         # 去掉空的值
        #         if twitters_one is None: continue
        #         add_msg = self.call_manager_agent_func(
        #             "add_tweets",
        #             kwargs={
        #                 "agent_id":agent_id,
        #                 "cur_time":self.time_configs["cur_time"].strftime("%Y-%m-%d"),
        #                 "twitters":twitters_one
        #             }
        #         )
        #         sub_add_msgs.append(add_msg)
        #     sub_add_num = [add_msg.content for add_msg in sub_add_msgs]
        #     num += sum(sub_add_num)        
        # self.call_manager_agent_func("update_docs").content
        return num
        
    
    def save(self, start_time):
        # self.social_manager.save_networkx_graph()
        self.call_manager_agent_func(
            "save_infos",
            kwargs={
                "cur_time": self.time_configs["cur_time"].strftime("%Y-%m-%d"),
                "start_time": start_time
            }
        ).content


    def initialize(self):
        is_rerun = self.call_manager_agent_func("rerun").content
        print(f"++++++++++++ Environment Initialize START. Rerun flag is: {is_rerun} ++++++++++++")
        if is_rerun:
            print("--> Entering RERUN logic branch.")
            self.update_agents(denote_transitive_log=False)
            delete_agent_ids = self.call_manager_agent_func("return_deleted_agent_ids").content
            for cur_agent_id in delete_agent_ids:
                if cur_agent_id in self.cur_agents.keys():
                    self.cur_agents.pop(cur_agent_id)
                    print(f"Agent {cur_agent_id} deleted")
            time_s = self.time_configs["cur_time"] - \
                                                  self.time_configs["social_time_delta"] 
            self.call_manager_agent_func("rerun_set_time",
                                         kwargs={
                                            "last_added_time": time_s.strftime("%Y-%m-%d")
                                         }).content
            # self.time_configs["cur_time"] = self.time_configs["cur_time"] - \
            #     self.time_configs["social_time_delta"]
            
        else:
            print("--> Entering NORMAL initialization logic branch.")
            self.update_agents()
                
        print("Finish Initialization")

    def test(self):
        """define unittest"""
        pass