import copy
from typing import Dict, Iterator, List, Optional, Union

from qwen_agent import Agent
from qwen_agent.llm import BaseChatModel
from qwen_agent.llm.schema import DEFAULT_SYSTEM_MESSAGE, FUNCTION, Message
from qwen_agent.memory import Memory
from qwen_agent.tools import BaseTool

MAX_LLM_CALL_PER_RUN = 8


class FnCallAgent(Agent): ## 继承自 agent.py 中的agent class
    """This is a widely applicable function call agent integrated with llm and tool use ability."""

    def __init__(self,
                 function_list: Optional[List[Union[str, Dict,
                                                    BaseTool]]] = None,
                 llm: Optional[Union[Dict, BaseChatModel]] = None,
                 system_message: Optional[str] = DEFAULT_SYSTEM_MESSAGE,
                 name: Optional[str] = None,
                 description: Optional[str] = None,
                 files: Optional[List[str]] = None):
        """Initialization the agent.

        Args:
            function_list: One list of tool name, tool configuration or Tool object,
              such as 'code_interpreter', {'name': 'code_interpreter', 'timeout': 10}, or CodeInterpreter(). ### agent中要用的那些工具。
            llm: The LLM model configuration or LLM model object.
              Set the configuration as {'model': '', 'api_key': '', 'model_server': ''}.  ## 配置LLM服务的相关参数
            system_message: The specified system message for LLM chat. ##### 给LLM的初始message
            name: The name of this agent. 
            description: The description of this agent, which will be used for multi_agent.
            files: A file url list. The initialized files for the agent. ## 暂时不知道怎么使用
        """
        super().__init__(function_list=function_list,
                         llm=llm,
                         system_message=system_message,
                         name=name,
                         description=description)

        # Default to use Memory to manage files
        self.mem = Memory(llm=self.llm, files=files)
    ### 重写Agent类的虚函数，其作用是基于给定的信息message，返回一个response generator
    def _run(self,
             messages: List[Message],
             lang: str = 'en',
             **kwargs) -> Iterator[List[Message]]:
        messages = copy.deepcopy(messages)
        num_llm_calls_available = MAX_LLM_CALL_PER_RUN
        response = []
        while True and num_llm_calls_available > 0:
            num_llm_calls_available -= 1
            output_stream = self._call_llm(   ### 基于给定的messages，为agent调用LLM的接口函数，返回LLM的response generator，函数中通过调用self.llm.chat(messages, functions)实现
                messages=messages,
                functions=[
                    func.function for func in self.function_map.values()
                ])
            output: List[Message] = []
            for output in output_stream:
                if output:
                    yield response + output
            if output:
                response.extend(output)
                messages.extend(output)
            use_tool, action, action_input, _ = self._detect_tool(response[-1])
            if use_tool:  ### 经过检查，确认LLM的判断是否调用工具，若是调用工具，则进行工具调用，并拿到工具调用的结果。 dectect_tool的作用只是检查message的function_call是否是None，而这个信息是由llm.chat调用生成message时，作为message类变量写入的。
                observation = self._call_tool(action,
                                              action_input,
                                              messages=messages)
                fn_msg = Message(
                    role=FUNCTION,
                    name=action,
                    content=observation,
                )
                messages.append(fn_msg)   ## 调用工具的结果，添加到message和response列表中。
                response.append(fn_msg)
                yield response
            else:
                break

    def _call_tool(self,
                   tool_name: str,
                   tool_args: Union[str, dict] = '{}',
                   **kwargs) -> str:
        # Temporary plan: Check if it is necessary to transfer files to the tool
        # Todo: This should be changed to parameter passing, and the file URL should be determined by the model
        if self.function_map[tool_name].file_access:   ### 如果这个工具是要access文件的，需要把所有的文件作为参数传给工具
            assert 'messages' in kwargs
            files = self.mem.get_all_files_of_messages(
                kwargs['messages']) + self.mem.system_files   ### 获取所有相关的文件，一部分是memory中存放的系统文件，一部分是message中所有的文件
            return super()._call_tool(tool_name,
                                      tool_args,
                                      files=files,
                                      **kwargs)
        else:
            return super()._call_tool(tool_name, tool_args, **kwargs)  ### 调用工具，拿到工具调用结果
