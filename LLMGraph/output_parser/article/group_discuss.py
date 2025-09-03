from __future__ import annotations
import json
from json import JSONDecodeError
import re
from typing import Union, Any, Dict, List
from loguru import logger

from ..tool_parser import ToolParser, find_and_load_json
from .. import article_output_parser_registry
from ..base_parser import AgentOutputParser

    
@article_output_parser_registry.register("group_discuss")
class GroupDiscussParser(AgentOutputParser):
    
    def parse(self, llm_output: str):
        
        
        try:
            last_period_index = llm_output.rfind('.')
            if last_period_index != -1:
                llm_output = llm_output[:last_period_index + 1]
               
            return {"return_values":{"communication":llm_output}}
        except Exception as e:
            raise {"fail":True}
    

    
@article_output_parser_registry.register("choose_researcher")
class ChooseResearcherParser(AgentOutputParser):
    
    def parse(self, llm_output: str):
        try:
            output = llm_output.strip().split("\n")[0]
            return {"return_values":{"researcher":output}}
        except Exception as e:
            return {"fail":True}
         
@article_output_parser_registry.register("get_idea")
class GetIdeaParser(AgentOutputParser):
    
    def parse(self, llm_output: str):
        llm_output +="\n"

        try:
            regex = r"Thought\s*\d*\s*:(.*?)\nIdea.*?:(.*?)\nKeywords.*?:(.*?)\nAbstract.*?:(.*?)\nFinish.*?:(.*?)\n"
            output = re.search(regex,llm_output,re.DOTALL|re.IGNORECASE)
            finish = output.group(5)
            if "true" in finish.lower():
                finish = True
            else:
                finish = False
            return {"return_values":{"thought":output.group(1),
                                    "action":"writetopic",
                                    "idea":output.group(2),
                                    "keywords":output.group(3).split(","),
                                    "finish":finish,
                                    "abstract":output.group(4)}}

                
        except Exception as e:
            return {"fail":True}
        
   
@article_output_parser_registry.register("proposal_initialization")
class ProposalInitializationParser(ToolParser):

    @property
    def _type(self) -> str:
        return "proposal_initialization_parser"

    @staticmethod
    def parse_ai_message_to_openai_tool_action(
        raw_output: Union[dict, str],
    ) -> Dict[str, Union[List, Dict, bool]]:
        res = raw_output

        # 1) 没有工具调用时，按 proposal JSON 或正则解析
        if isinstance(res, str) or (isinstance(res, dict) and len(res.get("function", [])) == 0):
            try:
                # 尝试 JSON 加载
                data = res if isinstance(res, dict) else json.loads(res)
                q = data.get("question")
                o = data.get("objectives")
                m = data.get("method")
                h = data.get("hypotheses")
                if isinstance(q, str) and isinstance(o, list) and isinstance(m, str) and isinstance(h, list):
                    return {"return_values": data}
            except Exception:
                pass

            # 正则回退
            text = res if isinstance(res, str) else json.dumps(res)
            pattern = re.compile(
                r'\{\s*"question"\s*:\s*"(?P<question>[^"]+)"\s*,'
                r'\s*"objectives"\s*:\[(?P<objectives>(?:"[^"]+"(?:\s*,\s*"[^"]+")*))\]\s*,'
                r'\s*"method"\s*:\s*"(?P<method>[^"]+)"\s*,'
                r'\s*"hypotheses"\s*:\[(?P<hypotheses>(?:"[^"]+"(?:\s*,\s*"[^"]+")*))\]\s*\}'
                , re.DOTALL
            )
            m_obj = pattern.search(text)
            if m_obj:
                objectives = re.findall(r'"([^"]+)"', m_obj.group("objectives"))
                hypotheses = re.findall(r'"([^"]+)"', m_obj.group("hypotheses"))
                return {"return_values": {
                    "question": m_obj.group("question"),
                    "objectives": objectives,
                    "method": m_obj.group("method"),
                    "hypotheses": hypotheses
                }}

        # 2) 检测到工具调用 → 输出 actions 以触发执行
        actions: List[Dict] = []

        # 列表形式的多次调用
        if isinstance(res, list):
            for act in res:
                if all(k in act for k in ("tool", "tool_input", "log")):
                    actions.append(act)
            return {"actions": actions}

        # dict 中的 function 字段
        try:
            funcs = res["function"]
        except Exception:
            # 尝试从文本中抽取 JSON
            try:
                funcs = find_and_load_json(res, "list")
            except Exception:
                return {"fail": True}

        for func in funcs:
            name = func.get("name")
            args = func.get("arguments")
            if not isinstance(args, dict):
                try:
                    parsed = json.loads(args or "{}")
                except JSONDecodeError:
                    return {"fail": True}
                tool_input = parsed.get("__arg1", parsed)
            else:
                tool_input = args

            log = f"Invoking `{name}` with {tool_input}"
            actions.append({
                "tool": name,
                "tool_input": tool_input,
                "log": log
            })

        return {"actions": actions}


@article_output_parser_registry.register("suggestion_generation")
class SuggestionGenerationParser(ToolParser):

    @property
    def _type(self) -> str:
        return "suggestion_generation_parser"

    @staticmethod
    def parse_ai_message_to_openai_tool_action(
        raw_output: Union[dict, str],
    ) -> Dict[str, Union[List, Dict, bool]]:
        res = raw_output

        # 1) 没有工具调用时，按 suggestions JSON 或正则解析
        if isinstance(res, str) or (isinstance(res, dict) and len(res.get("function", [])) == 0):
            try:
                data = res if isinstance(res, dict) else json.loads(res)
                sug = data.get("suggestions")
                if isinstance(sug, list):
                    return {"return_values": {"suggestions": sug}}
            except Exception:
                pass

            # 正则回退
            text = res if isinstance(res, str) else json.dumps(res)
            pattern = re.compile(r'\{\s*"suggestions"\s*:\s*\[(?P<array>.+?)\]\s*\}', re.DOTALL)
            m_sug = pattern.search(text)
            if m_sug:
                arr = m_sug.group("array")
                items = re.findall(
                    r'\{\s*"area"\s*:\s*"([^"]+)"\s*,\s*"detail"\s*:\s*"([^"]+)"\s*\}',
                    arr
                )
                suggestions = [{"area": a, "detail": d} for a, d in items]
                return {"return_values": {"suggestions": suggestions}}

        # 2) 检测到工具调用 → 输出 actions
        actions: List[Dict] = []

        if isinstance(res, list):
            for act in res:
                if all(k in act for k in ("tool", "tool_input", "log")):
                    actions.append(act)
            return {"actions": actions}

        try:
            funcs = res["function"]
        except Exception:
            try:
                funcs = find_and_load_json(res, "list")
            except Exception:
                return {"fail": True}

        for func in funcs:
            name = func.get("name")
            args = func.get("arguments")
            if not isinstance(args, dict):
                try:
                    parsed = json.loads(args or "{}")
                except JSONDecodeError:
                    return {"fail": True}
                tool_input = parsed.get("__arg1", parsed)
            else:
                tool_input = args

            log = f"Invoking `{name}` with {tool_input}"
            actions.append({
                "tool": name,
                "tool_input": tool_input,
                "log": log
            })

        return {"actions": actions}

@article_output_parser_registry.register("proposal_generation")
class ProposalGenerationParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Dict[str, Any]:
        """
        支持解析两种 key:
          "final_hypotheses" 或 "hypotheses"
        """
        logger.info(f"proposal_generation llm_output: {llm_output}")

        # 1. 尝试 JSON 解析
        try:
            data = json.loads(llm_output)
            fq = data.get("final_question")
            fo = data.get("final_objectives")
            fm = data.get("final_method")
            # 如果没有 final_hypotheses，就尝试 hypotheses
            fh = data.get("final_hypotheses", data.get("hypotheses"))

            if (
                isinstance(fq, str)
                and isinstance(fo, list)
                and isinstance(fm, str)
                and isinstance(fh, list)
            ):
                # 规范化字段名，统一用 final_hypotheses
                data["final_hypotheses"] = fh
                return {"return_values": data}
        except JSONDecodeError:
            pass

        # 2. 正则回退，兼容 final_hypotheses 或 hypotheses
        text = llm_output
        pattern = re.compile(
            r'\{\s*"final_question"\s*:\s*"(?P<fq>[^"]+)"\s*,'
            r'\s*"final_objectives"\s*:\s*\[(?P<fo>(?:"[^"]+"(?:\s*,\s*"[^"]+")*))\]\s*,'
            r'\s*"final_method"\s*:\s*"(?P<fm>[^"]+)"\s*,'
            r'\s*(?:"final_hypotheses"|"hypotheses")\s*:\s*\[(?P<fh>(?:"[^"]+"(?:\s*,\s*"[^"]+")*))\]\s*\}',
            re.DOTALL,
        )
        m = pattern.search(text)
        if m:
            fo_list = re.findall(r'"([^"]+)"', m.group("fo"))
            fh_list = re.findall(r'"([^"]+)"', m.group("fh"))
            return {
                "return_values": {
                    "final_question": m.group("fq"),
                    "final_objectives": fo_list,
                    "final_method": m.group("fm"),
                    "final_hypotheses": fh_list,
                }
            }

        return {"fail": True}
# @article_output_parser_registry.register("proposal_generation")
# class ProposalGenerationParser(AgentOutputParser):
#     def parse(self, llm_output: str) -> Dict[str, Any]:
#         """
#         首先尝试解析 JSON；失败后用单条正则提取所有字段：
#         {
#           "final_question": str,
#           "final_objectives": [str,...],
#           "final_method": str,
#           "final_hypotheses": [str,...]
#         }
#         返回: {"return_values": {...}} or {"fail": True}
#         """
#         # 1. JSON 解析
#         logger.info(f"proposal_generation llm_output: {llm_output}")
#         try:
#             data = json.loads(llm_output)
#             return {"return_values": data}
#         except json.JSONDecodeError:
#             pass
#         # 2. 正则一次匹配全部字段
#         pattern = re.compile(
#             r'\{\s*"final_question"\s*:\s*"(?P<fq>[^"]+)"\s*,'
#             r'\s*"final_objectives"\s*:\s*\[(?P<fo>(?:"[^"]+"(?:\s*,\s*"[^"]+"))*)\]\s*,'
#             r'\s*"final_method"\s*:\s*"(?P<fm>[^"]+)"\s*,'
#             r'\s*"final_hypotheses"\s*:\s*\[(?P<fh>(?:"[^"]+"(?:\s*,\s*"[^"]+"))*)\]\s*\}'
#         , re.DOTALL)
#         m = pattern.search(llm_output)
#         if m:
#             final_objectives = re.findall(r'"([^"]+)"', m.group('fo'))
#             final_hypotheses = re.findall(r'"([^"]+)"', m.group('fh'))
#             return {"return_values": {
#                 "final_question": m.group('fq'),
#                 "final_objectives": final_objectives,
#                 "final_method": m.group('fm'),
#                 "final_hypotheses": final_hypotheses
#             }}
#         return {"fail": True}
