import json
import asyncio

from pydantic import Field
from tenacity import retry, stop_after_attempt, wait_random_exponential

from metagpt.actions.action import Action
from metagpt.actions.project_management_an import REFINED_TASK_LIST, TASK_LIST
from metagpt.actions.write_code_plan_and_change_an import REFINED_TEMPLATE
from metagpt.const import BUGFIX_FILENAME, REQUIREMENT_FILENAME
from metagpt.logs import logger
from metagpt.schema import CodingContext, Document, RunCodeResult
from metagpt.utils.common import CodeParser
from metagpt.utils.project_repo import ProjectRepo

from metagpt.roles import Role
from metagpt.schema import Message

from pathlib import Path
import re


PROMPT_TEMPLATE = """
    Write a python function that can {instruction} and provide two runable test cases.
    Return ```python your_code_here ``` with NO other texts,
    your code:
    """


class SimpleWriteCode(Action):
    
    name: str = "WriteCode"
    i_context: Document = Field(default_factory=Document)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.context.git_repo = r'E:\Python_work\LLM_MetaGPT\MetaGPT-main'
        self.PROMPT_TEMPLATE = PROMPT_TEMPLATE

    async def run(self, *args, **kwargs) -> CodingContext:
        prompt = self.PROMPT_TEMPLATE.format(instruction=kwargs['instruction'])

        rsp = await self._aask(prompt)

        code_text = SimpleWriteCode.parse_code(rsp)

        return code_text

    @staticmethod
    def parse_code(rsp):
        pattern = r'```python(.*)```'
        match = re.search(pattern, rsp, re.DOTALL)
        code_text = match.group(1) if match else rsp
        return code_text


class SimpleCoder(Role):
    name: str = "Alice"
    profile: str = "SimpleCoder"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.context.git_repo = r'E:\Python_work\LLM_MetaGPT\MetaGPT-main'
        self.set_actions([SimpleWriteCode])

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo  # todo will be SimpleWriteCode()

        msg = self.get_memories(k=1)[0]  # find the most recent messages
        code_text = await todo.run(instruction = msg.content)
        msg = Message(content=code_text, role=self.profile, cause_by=type(todo))

        return msg


async def main():
    from pathlib import Path
    msg = "write a function that calculates the sum of a list"
    role = SimpleCoder()
    logger.info(msg)
    result = await role.run(msg)
    logger.info(result)

asyncio.run(main())
