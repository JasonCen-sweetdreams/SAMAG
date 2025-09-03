from .. import movie_output_parser_registry
from ..base_parser import AgentOutputParser, find_and_load_json

@movie_output_parser_registry.register("propose_movie")
class ProposeMovieParser(AgentOutputParser):
    def parse(self, llm_output: str):
        try:
            parsed_json = find_and_load_json(llm_output, "dict")
            return parsed_json
        except Exception as e:
            return {"fail": True, "error": str(e)}

@movie_output_parser_registry.register("discuss_movie")
class DiscussMovieParser(AgentOutputParser):
    def parse(self, llm_output: str):
        try:
            parsed_json = find_and_load_json(llm_output, "dict")
            return parsed_json
        except Exception as e:
            return {"fail": True, "error": str(e)}