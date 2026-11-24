"""使用 OpenAI 风格消息格式的最小同步 ReAct 循环。"""

import json
from types import SimpleNamespace


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register(self, name, description, parameters, handler):
        """注册工具函数及其关键字参数的 JSON Schema。"""
        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": parameters,
            },
        }
        self._tools[name] = (schema, handler)

    def schemas(self):
        return [schema for schema, _ in self._tools.values()]

    def execute(self, name, arguments):
        _, handler = self._tools[name]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        return handler(**arguments)


class ReActAgent:
    def __init__(self, llm, registry, max_steps=8):
        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps

    def run(self, prompt):
        """返回最终回答或步数耗尽提示。raw 必须是可直接放入消息列表的 assistant 消息。"""
        messages = [{"role": "user", "content": prompt}]
        tools = self.registry.schemas()

        for _ in range(self.max_steps):
            resp = self.llm.chat(messages, tools)
            if not resp.tool_calls:
                return resp.content

            # 先保留助手的工具调用消息，再按调用 ID 回填每个工具的结果。
            messages.append(resp.raw)
            for call in resp.tool_calls:
                try:
                    result = self.registry.execute(call.name, call.arguments)
                    content = result if isinstance(result, str) else str(result)
                except Exception as exc:
                    content = json.dumps(
                        {"error": f"{type(exc).__name__}: {exc}"},
                        ensure_ascii=False,
                    )

                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": content,
                })

        # 生产环境应在步数耗尽时将请求转交人工处理。
        return f"步数耗尽（上限 {self.max_steps} 轮），未获得最终回答，请转人工处理。"


class FakeLLM:
    """用固定的 2 + 3 示例演示两轮对话。"""

    def chat(self, messages, tools):
        last_message = messages[-1]
        if last_message["role"] == "user":
            print("思考：调用加法工具计算 2 + 3。")
            call = SimpleNamespace(id="call_add", name="add", arguments='{"a": 2, "b": 3}')
            raw = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call.id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": call.arguments},
                }],
            }
            return SimpleNamespace(content=None, tool_calls=[call], raw=raw)

        content = f"2 + 3 = {last_message['content']}"
        raw = {"role": "assistant", "content": content}
        return SimpleNamespace(content=content, tool_calls=[], raw=raw)


if __name__ == "__main__":
    def add(a, b):
        print(f"调工具：add({a}, {b})")
        return a + b

    registry = ToolRegistry()
    registry.register(
        "add",
        "Add two integers",
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        add,
    )
    answer = ReActAgent(FakeLLM(), registry, max_steps=2).run("2 + 3 等于多少？")
    print(f"再回答：{answer}")
    print("退出")
