import copy
import random
import time
from abc import ABC, abstractmethod
from typing import Dict, Iterator, List, Optional, Union

from qwen_agent.utils.tokenization_qwen import tokenizer
from qwen_agent.utils.utils import (get_basename_from_url, has_chinese_chars,
                                    is_image, print_traceback)

from .schema import (ASSISTANT, DEFAULT_SYSTEM_MESSAGE, FUNCTION, SYSTEM, USER,
                     ContentItem, Message)

LLM_REGISTRY = {}


def register_llm(model_type):

    def decorator(cls):
        LLM_REGISTRY[model_type] = cls
        return cls

    return decorator


class ModelServiceError(Exception):

    def __init__(self,
                 exception: Optional[Exception] = None,
                 code: Optional[str] = None,
                 message: Optional[str] = None):
        if exception is not None:
            super().__init__(exception)
        else:
            super().__init__(f'\nError code: {code}. Error message: {message}')
        self.exception = exception
        self.code = code
        self.message = message


class BaseChatModel(ABC):
    """The base class of LLM"""

    def __init__(self, cfg: Optional[Dict] = None):
        cfg = cfg or {}
        self.model = cfg.get('model', '')
        generate_cfg = copy.deepcopy(cfg.get('generate_cfg', {}))
        self.max_retries = generate_cfg.pop('max_retries', 0)
        self.generate_cfg = generate_cfg

    def chat(
        self,
        messages: List[Union[Message, Dict]],
        functions: Optional[List[Dict]] = None,
        stream: bool = True,
        delta_stream: bool = False,
    ) -> Union[List[Message], List[Dict], Iterator[List[Message]],
               Iterator[List[Dict]]]:
        """LLM chat interface.

        Args:
            messages: Inputted messages.
            functions: Inputted functions, which supports OpenAI format.
            stream: Whether to use streaming generation.
            delta_stream: Whether to return incrementally.
              (1) When False: Use full return.
              (2) When True: Use incremental return.

        Returns:
            the generated message list response by llm.
        """
        messages = copy.deepcopy(messages)

        _return_message_type = 'dict'
        new_messages = []
        for msg in messages:
            if isinstance(msg, dict):  ### 统一入参messages的格式，如果是dict，将其转换为Message类，Message类是一个BaseModelCompatibleDict，其基础也是一个dict。但为什么要用dict作为基础类型呢？
                new_messages.append(Message(**msg))
            else:
                new_messages.append(msg)
                _return_message_type = 'message'
        messages = new_messages

        if messages[0].role != SYSTEM:   ### 一定保证第一条message是系统说的话。大模型说的话是ASSISTANT这个角色说的。
            messages = [Message(role=SYSTEM, content=DEFAULT_SYSTEM_MESSAGE)
                        ] + messages

        messages = self._preprocess_messages(messages) #### 统一message中content的格式，每条message中的content都是list[ContentItem]格式。

        if functions:
            fncall_mode = True
        else:
            fncall_mode = False

        def _call_model_service():
            if fncall_mode:
                return self._chat_with_functions(
                    messages=messages,
                    functions=functions,
                    stream=stream,
                    delta_stream=delta_stream,
                )
            else:
                return self._chat(
                    messages,
                    stream=stream,
                    delta_stream=delta_stream,
                )

        if stream and delta_stream:
            # No retry for delta streaming
            output = _call_model_service()
        elif stream and (not delta_stream):
            output = retry_model_service_iterator(_call_model_service,
                                                  max_retries=self.max_retries)
        else:
            output = retry_model_service(_call_model_service,
                                         max_retries=self.max_retries)

        if isinstance(output, list):
            output = self._postprocess_messages(output,
                                                fncall_mode=fncall_mode)
            return self._convert_messages_to_target_type(
                output, _return_message_type)
        else:
            output = self._postprocess_messages_iterator(
                output, fncall_mode=fncall_mode)
            return self._convert_messages_iterator_to_target_type(
                output, _return_message_type)

    def _chat(
        self,
        messages: List[Union[Message, Dict]],
        stream: bool = True,
        delta_stream: bool = False,
    ) -> Union[List[Message], Iterator[List[Message]]]:
        if stream:
            return self._chat_stream(messages, delta_stream=delta_stream)
        else:
            return self._chat_no_stream(messages)

    @abstractmethod
    def _chat_with_functions(
        self,
        messages: List[Union[Message, Dict]],
        functions: List[Dict],
        stream: bool = True,
        delta_stream: bool = False
    ) -> Union[List[Message], Iterator[List[Message]]]:
        raise NotImplementedError

    @abstractmethod
    def _chat_stream(
        self,
        messages: List[Message],
        delta_stream: bool = False,
    ) -> Iterator[List[Message]]:
        raise NotImplementedError

    @abstractmethod
    def _chat_no_stream(
        self,
        messages: List[Message],
    ) -> List[Message]:
        raise NotImplementedError

    def _preprocess_messages(self, messages: List[Message]) -> List[Message]:
        messages = self._format_as_multimodal_messages(messages)  #### 统一message中content的格式，每条message中的content都是list[ContentItem]格式。
        return messages

    def _postprocess_messages(self, messages: List[Message],
                              fncall_mode: bool) -> List[Message]:
        messages = self._format_as_multimodal_messages(messages)
        messages = self._postprocess_stop_words(messages)
        return messages

    def _postprocess_messages_iterator(
        self,
        messages: Iterator[List[Message]],
        fncall_mode: bool,
    ) -> Iterator[List[Message]]:
        for m in messages:
            m = self._postprocess_messages(m, fncall_mode=fncall_mode)
            if m:
                yield m

    def _convert_messages_to_target_type(
            self, messages: List[Message],
            target_type: str) -> Union[List[Message], List[Dict]]:
        if target_type == 'message':
            return [
                Message(**x) if isinstance(x, dict) else x for x in messages
            ]
        elif target_type == 'dict':
            return [
                x.model_dump() if not isinstance(x, dict) else x
                for x in messages
            ]
        else:
            raise NotImplementedError

    def _convert_messages_iterator_to_target_type(
        self, messages_iter: Iterator[List[Message]], target_type: str
    ) -> Union[Iterator[List[Message]], Iterator[List[Dict]]]:
        for messages in messages_iter:
            yield self._convert_messages_to_target_type(messages, target_type)

    def _format_as_multimodal_messages(
            self, messages: List[Message]) -> List[Message]:

        multimodal_messages = []
        for msg in messages:
            assert msg.role in (USER, ASSISTANT, SYSTEM, FUNCTION)

            content = []
            if isinstance(msg.content, str):  # if text content  ## message的content变量存放message的内容，可以是str类型，也可以是List[ContentItem]类型。也就是说，一条message可以包含多个内容，但对应的角色唯一，它可以一连串说多个事情（包括上传文件），都放到一个message中。
                if msg.content:
                    content = [ContentItem(text=msg.content)]   ### 每个ContentItem有三个元素，text，image，file，存放不同类型的message的内容。此处将str类型的content转换为list[ContentItem]类型，方便后续统一处理
            elif isinstance(msg.content, list):  # if multimodal content   若message的content为list[ContentItem]类型，则其一般存放的是多模态内容
                files = []
                for item in msg.content:
                    (k, v), = item.model_dump().items()  ## item为ContentItem类型，是BaseModel的子类。model_dump()函数把整个类的信息dump到一个dict中。items()函数把dict的信息转换为可便利的(键，值)元组数组。再经过转换之后，k存放所有键值，v存放所有value值。
                    if k in ('box', 'text'):
                        content.append(ContentItem(text=v))
                    if k == 'image':
                        content.append(item)  ## 如果是image content，则直接加入到最终的content中。
                    if k in ('file', 'image'): ## 如果是image或file，全部放到files list中，后续对所有file进行处理。
                        files.append(v)
                if (msg.role in (SYSTEM, USER)) and files:
                    has_zh = has_chinese_chars(content)   ### 看所有的content中是否含有中文汉字
                    upload = []
                    for f in [get_basename_from_url(f) for f in files]:  ## 遍历所有的基础文件名
                        if is_image(f):   ### 若后缀是图片类型，例如jpg、JPEG、平、webp等
                            if has_zh:
                                upload.append(f'![图片]({f})')
                            else:
                                upload.append(f'![image]({f})')
                        else:
                            if has_zh:
                                upload.append(f'[文件]({f})')
                            else:
                                upload.append(f'[file]({f})')
                    upload = ' '.join(upload)
                    if has_zh:
                        upload = f'（上传了 {upload}）\n\n'   ### 相当于遍历了所有的文件，告诉模型，已经上传完成了
                    else:
                        upload = f'(Uploaded {upload})\n\n'
                    content = [ContentItem(text=upload)] + content  ### 所有的文件合并上传之后，作为一个统一的contentItem加入到 message的content这个list中。也就是上传文件的过程作为一个message
            else:
                raise TypeError
            ##### 用新产生的content生成一个新的message，相当于统一了message中content的格式，并添加了上传文件的操作。
            multimodal_messages.append(  
                Message(
                    role=msg.role,
                    content=content,
                    name=msg.name if msg.role == FUNCTION else None,
                    function_call=msg.function_call,
                ))

        return multimodal_messages

    def _postprocess_stop_words(self,
                                messages: List[Message]) -> List[Message]:
        messages = copy.deepcopy(messages)
        stop = self.generate_cfg.get('stop', [])

        # Make sure it stops before stop words.
        trunc_messages = []
        for msg in messages:
            truncated = False
            trunc_content = []
            for i, item in enumerate(msg.content):
                item_type, item_text = item.get_type_and_value()
                if item_type == 'text':
                    truncated, item.text = _truncate_at_stop_word(
                        text=item_text, stop=stop)
                trunc_content.append(item)
                if truncated:
                    break
            msg.content = trunc_content
            trunc_messages.append(msg)
            if truncated:
                break
        messages = trunc_messages

        # It may ends with 'Observation' when the stop word is 'Observation:'.
        partial_stop = []
        for s in stop:
            # TODO: This tokenizer is Qwen-specific.
            s = tokenizer.tokenize(s)[:-1]
            if s:
                s = tokenizer.convert_tokens_to_string(s)
                partial_stop.append(s)
        partial_stop = sorted(set(partial_stop))
        last_msg = messages[-1].content
        for i in range(len(last_msg) - 1, -1, -1):
            item_type, item_text = last_msg[i].get_type_and_value()
            if item_type == 'text':
                for s in partial_stop:
                    if item_text.endswith(s):
                        last_msg[i].text = item_text[:-len(s)]
                break

        return messages


def _truncate_at_stop_word(text: str, stop: List[str]):
    truncated = False
    for s in stop:
        k = text.find(s)
        if k >= 0:
            truncated = True
            text = text[:k]
    return truncated, text


def retry_model_service(
    fn,
    max_retries: int = 10,
    exponential_base: float = 1.0,
):
    """Retry a function with exponential backoff"""

    num_retries = 0
    delay = 2.0
    while True:
        try:
            return fn()

        except ModelServiceError as e:
            if max_retries <= 0:  # no retry
                raise e

            # If harmful input or output detected, let it fail
            if e.code == 'DataInspectionFailed':
                raise e
            if 'inappropriate content' in str(e):
                raise e

            # Retry is meaningless if the input is too long
            if 'maximum context length' in str(e):
                raise e

            print_traceback(is_error=False)

            if num_retries >= max_retries:
                raise ModelServiceError(exception=Exception(
                    f'Maximum number of retries ({max_retries}) exceeded.'))

            num_retries += 1
            delay *= exponential_base * (1.0 + random.random())
            time.sleep(delay)


def retry_model_service_iterator(
    it_fn,
    max_retries: int = 10,
    exponential_base: float = 1.0,
):
    """Retry an iterator with exponential backoff"""

    num_retries = 0
    delay = 2.0

    while True:
        try:
            for rsp in it_fn():
                yield rsp
            break

        except ModelServiceError as e:
            if max_retries <= 0:  # no retry
                raise e

            # If harmful input or output detected, let it fail
            if e.code == 'DataInspectionFailed':
                raise e
            if 'inappropriate content' in str(e):
                raise e

            # Retry is meaningless if the input is too long
            if 'maximum context length' in str(e):
                raise e

            print_traceback(is_error=False)

            if num_retries >= max_retries:
                raise ModelServiceError(exception=Exception(
                    f'Maximum number of retries ({max_retries}) exceeded.'))

            num_retries += 1
            delay *= exponential_base * (1.0 + random.random())
            time.sleep(delay)
