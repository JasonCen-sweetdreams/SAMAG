from typing import Dict,Sequence, Union
from agentscope.agents import AgentBase
from .base import BaseAgentWrapper
from agentscope.message import Msg, PlaceholderMessage
import time
from Emulate.utils.str_process import remove_before_first_space
from Emulate.utils.timing import timed
from loguru import logger

from datetime import datetime


class SocialAgentWrapper(BaseAgentWrapper):
    
    def __init__(self, 
                 name: Union[str,int], 
                 agent:AgentBase,
                 manager_agent:AgentBase,
                 max_retrys: int = 3, 
                 max_tool_iters: int = 2,
                 to_dist = False, 
                 **kwargs) -> None:
        
        """tools: 需要已经to_dist后的agent"""
        
        
        super().__init__(name, 
                         agent, 
                         manager_agent, 
                         max_retrys, 
                         max_tool_iters,
                         to_dist = to_dist,
                          **kwargs)


    def reply(self, message:Msg = None) -> Msg:
        available_func_names = ["twitter_process",
                                "get_acting_plan"]
        func_name = message.get("func")
        assert func_name in available_func_names, f"{func_name} not in {available_func_names}"
        func = getattr(self,func_name)
        kwargs = message.get("kwargs",{})
        func_res = func(**kwargs)
        return func_res
        
    
    @timed
    def twitter_process(self,
                        cur_time:str,
                        big_name:bool = False,
                        group_context: list = None,
                        in_group: bool = False):
        cur_time = datetime.strptime(cur_time,"%Y-%m-%d").date()
        start_time = time.perf_counter()
        follow_content = self.call_manager_func(
                "get_follow_ids",
                kwargs={
                    "agent_id":self.name,
                }
            ).content
        get_follow_ids_end_time = time.perf_counter()
        # twitter_infos = self.get_twitter_search_batch(follow_content)
        
        if group_context:
            group_prompt = """**Group Interaction Guidelines:**
You are part of a group. The following `group context` (between "=== group context start ===" and "=== group context end ===") details recent actions by other members. Use this to inform your actions:

1.  **Analyze Group Context:** Identify overall trends, topics, and the types of actions being taken (e.g., `act_type` and `content`). Pay special attention to influential members (like leaders or initial posters who set the tone).
2.  **Aim for Cohesion & Support:**
    * Consider performing actions that are similar in type or theme to those already present in the group context.
    * Look for opportunities to offer supportive contributions, such as agreeing with, elaborating on, or positively reinforcing existing posts or topics initiated by others.

Your actions should reflect a clear understanding of and engagement with the group's ongoing activity.
"""
            ctx_lines = ["=== group context start ==="]
            for action in group_context:
                agent_id = action.get("agent_id")
                act_type = action.get("action")
                content = action.get("input", "")
                line = f"user {agent_id} did {act_type}"
                if content:
                    line += f" with content: \"{content}\""
                ctx_lines.append(line)
            ctx_lines.append("=== group context end ===")
            ctx_block = group_prompt + "\n" +"\n".join(ctx_lines)
            twitter_infos = ""
        else:
            ctx_block = ""
            twitter_infos = self.get_twitter_search_batch(follow_content, in_group)

            if twitter_infos == "":
                return Msg(
                self.name,
                content=[],
                role="assistant"
            )
        process_end_time = time.perf_counter()
        if big_name:
            forum_actions = self.forum_action_bigname(
                twitter_infos=twitter_infos,
                group_context=ctx_block
                )
        else:
            forum_actions = self.forum_action(
                twitter_infos=twitter_infos,
                group_context=ctx_block
               )
        action_end_time = time.perf_counter()
        available_actions = ["tweet","retweet","reply"]
        forum_actions_filtered = []
        for forum_action in forum_actions:
            if isinstance(forum_action,dict):
                try:
                    if forum_action.get("action","").lower() in available_actions:
                        forum_actions_filtered.append(forum_action)
                except:
                    pass
        self.call_agent_func("add_twitter_action_memory",
                    kwargs={
                        "forum_actions":forum_actions_filtered
                    }).content
        logger.info(f"Twitter_process total time consume: {action_end_time - start_time}\n\
                    Is group context: {'True' if group_context else 'False'}\n\
                    Get_follow_ids time: {get_follow_ids_end_time - start_time}\n\
                    Process/Search time: {process_end_time - get_follow_ids_end_time}\n\
                    Action time: {action_end_time - process_end_time}")
        return Msg(
            self.name,
            content=forum_actions_filtered,
            role="assistant"
        )
        
    @timed
    def forum_action(self, 
                     twitter_infos:str,
                     group_context: list = None) -> Msg:
        time_cache = {}
        time_start = time.time()
        role_description = self.call_manager_func(
            "get_user_role_description",
            kwargs={"user_index":self.name}            
        ).content
        friend_data = self.call_manager_func(
            "get_user_friend_info",
            kwargs={"user_index":self.name}            
        ).content
        
        time_cache["friend_data"] = time.time()-time_start
        time_start = time.time()

        num_followers = self.call_manager_func(
            "get_user_num_followers",
            kwargs={"user_index":self.name}            
        ).content

        time_cache["num_followers"] = time.time()-time_start
        time_start = time.time()

        forum_msg_content = self.call_agent_func("forum_action",
            kwargs={
                "role_description":role_description,
                "friend_data":friend_data,
                "twitter_data":twitter_infos,
                "group_context":group_context,
                "num_followers":num_followers
            }).content
        
        assert forum_msg_content is not None
        time_all_end = time.time()-time_start
        time_cache["forum_action"] = time_all_end
        return forum_msg_content
    
    @timed
    def forum_action_bigname(self, twitter_infos:str, group_context: list = None) -> Msg:
        time_start = time.time()
        time_start_all = time_start
        time_cache = {}

        role_description = self.call_manager_func(
            "get_user_role_description",
            kwargs={"user_index":self.name}            
        ).content
        
        friend_data = self.call_manager_func(
            "get_user_friend_info",
            kwargs={"user_index":self.name}            
        ).content
        time_cache["friend_data"] = time.time()-time_start
        time_start = time.time()
        
        num_followers = self.call_manager_func(
            "get_user_num_followers",
            kwargs={"user_index":self.name}            
        ).content
        time_cache["num_followers"] = time.time()-time_start
        time_start = time.time()

        forum_msg_content = self.call_agent_func("forum_action_bigname",
            kwargs={
                "role_description":role_description,
                "friend_data":friend_data,
                "twitter_data":twitter_infos,
                "group_context":group_context,
                "num_followers":num_followers
            }).content
        assert forum_msg_content is not None
        time_all = time.time() - time_start_all
        time_cache["forum_action_bigname"] = time_all

        return forum_msg_content
    
    @timed
    def get_twitter_search_batch(self,
                                 follow_content: dict = {},
                                 in_group = False) -> str:
        time_start_all = time.time()
        time_start = time_start_all
        time_cache = {}
        role_description = self.call_manager_func(
            "get_user_role_description",
            kwargs={"user_index":self.name}            
        ).content
        
        # update tool with follow content


        agent_msgs = self.call_agent_get_prompt(            
            "get_twitter_search_batch",
            kwargs={
                "role_description":role_description,
            }).content
        time_cache["prompt_inputs"] = time.time()-time_start
        time_start = time.time()

        interested_topics = self.call_agent_func(
            "return_interested_topics"
        ).content
        time_cache["interested_topics"] = time.time()-time_start
        time_start = time.time()

        self.call_manager_func("update",
                               kwargs={
                                    "social_follow_map":follow_content,
                                    "interested_topics":interested_topics,
                                })
        time_cache["update_tool"] = time.time()-time_start
        time_start = time.time()

        response = self.step(agent_msgs=agent_msgs,
                             use_tools=True,
                             return_tool_exec_only=True,
                             return_intermediate_steps=True)
        time_cache["step"] = time.time()-time_start
        time_start = time.time()
        intermediate_steps = response.get("intermediate_steps",[])
        if intermediate_steps== []:
            return ""

        searched_infos = ""
        searched_keywords = []
        new_action_count = 0
        for intermediate_step in intermediate_steps:
            action, observation = intermediate_step
            result = observation.get("result",{})
            keyword = action.get("kwargs",{}).get("keyword","")
            searched_keywords.append(keyword)
            try:
                searched_infos += result.get("output","")+ "\n"
            except:
                pass
            # 只进行少量的新动作
            if in_group:
                new_action_count += 1
                if new_action_count >= 5:
                    break

        self.call_agent_func(
            "add_tweets",
            kwargs={
                "tweet_content": searched_infos,
                "type":"seen"
            }
        ).content

        time_cache["add_tweets"] = time.time()-time_start
        time_start = time.time()

        self.call_agent_func(
            "add_searched_keywords",
            kwargs={
                "keywords": searched_keywords
            }
        ).content
        
        time_all = time.time()-time_start_all
        time_cache["add_tweets"] = time_all
        logger.info(f"Search Process time consume: {time_cache}")
        return searched_infos
    

    def get_agent_memory_msgs(self):
        return self.call_agent_func(
            "get_short_memory"
        )
    
    def get_acting_plan(self):
        role_description = self.call_manager_func(
            "get_user_role_description",
            kwargs={"user_index":self.name}            
        ).content

        big_name = self.call_manager_func(
            "get_user_big_name",
            kwargs={"user_index":self.name}            
        ).content

        plan = self.call_agent_func(
            "get_acting_plan",
            kwargs={
                "role_description":role_description,
                "big_name":big_name
            }
        ).content

        return Msg(            
            self.name,
            content=plan,
            role="assistant"
        )