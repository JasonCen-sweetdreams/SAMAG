from .. import movie_output_parser_registry
from ..base_parser import AgentOutputParser, find_and_load_json

@movie_output_parser_registry.register("select_movies")
class SelectMoviesParser(AgentOutputParser):
    def parse(self, llm_output: str):
        try:
            parsed_json = find_and_load_json(llm_output, "dict")
            # 期望的返回结构是 {"return_values": {"selected_movies": [...]}}
            if "return_values" in parsed_json and "selected_movies" in parsed_json["return_values"]:
                 return parsed_json
            else:
                 return {"fail": True, "error": "Missing 'selected_movies' key in parsed JSON."}
        except Exception as e:
            return {"fail": True, "error": str(e)}
