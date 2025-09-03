from abc import abstractmethod
from typing import Dict, List, Union

from pydantic import BaseModel, Field

from LLMGraph.message import Message
from . import memory_registry


@memory_registry.register("base")
class BaseMemory(BaseModel):
    # name :str # 标记是谁的记忆
    name: Union[str, int]
    id:str= None# 标记是谁的记忆
    class Config:
        arbitrary_types_allowed = True

    def add_message(self, messages: List[Message]) -> None:
        pass


    def to_string(self) -> str:
        pass


    def reset(self) -> None:
        pass


