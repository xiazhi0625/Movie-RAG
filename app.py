import re

print("【1】开始导入 Gradio", flush=True)
import gradio as gr
print("【2】Gradio 导入完成", flush=True)

print("【3】开始导入 retriever", flush=True)
from retriever import MovieRetriever
print("【4】retriever 导入完成", flush=True)

print("【5】开始导入 llm_client", flush=True)
from llm_client import LLMClient
print("【6】llm_client 导入完成", flush=True)

print("【7】开始初始化 MovieRetriever", flush=True)
retriever = MovieRetriever()
print("【8】MovieRetriever 初始化完成", flush=True)

print("【9】开始初始化 LLMClient", flush=True)
llm = LLMClient()
print("【10】LLMClient 初始化完成", flush=True)

def is_valid_query(text):
    clean_text = text.strip()
    valid_chars = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]', clean_text)
    if len(valid_chars) / len(clean_text) < 0.3:
        return False
    return True

def normalize_history(history):
    """将 Gradio 历史记录统一转换成纯文本消息"""
    normalized = []

    for item in history or []:
        if not isinstance(item, dict):
            continue

        role = item.get("role")
        if role not in ("user", "assistant"):
            continue

        content = item.get("content")

        if isinstance(content, str):
            text = content

        elif isinstance(content, list):
            text = "\n".join(
                block["text"]
                for block in content
                if isinstance(block, dict)
                and isinstance(block.get("text"), str)
                and block.get("type") == "text"
            )
        else:
            continue

        if text.strip():
            normalized.append({
                "role": role,
                "content": text
            })

    return normalized


def respond(message, history, user_pref):
    # ① 历史记录标准化
    history = normalize_history(history)
    user_pref = user_pref or ""

    try:
        # ② 输入合法性检查
        if not message or not message.strip():
            yield "请输入有效的问题哦～", user_pref
            return

        if not is_valid_query(message):
            yield "请输入与电影相关的有效问题～", user_pref
            return

        # ③ 提取用户偏好
        if "我喜欢" in message or "我想看" in message:
            new_pref = llm.extract_preference(message)

            if new_pref:
                user_pref = new_pref
                print(f"已提取偏好: {user_pref}", flush=True)

        # ④ 结合历史记录重写查询
        search_query = message

        if history:
            search_query = llm.rewrite_query(
                history, message
            )

        print(f"检索词: {search_query}", flush=True)

        # ⑤ 检索电影数据库
        context_text, metas = retriever.hybrid_search(
            search_query,
            n_results=3
        )

        if not context_text:
            yield "抱歉，资料库中没有找到相关电影。", user_pref
            return

        # ⑥ 整理最近三轮历史对话
        history_text = ""

        for item in history[-6:]:
            speaker = (
                "用户"
                if item["role"] == "user"
                else "AI"
            )
            history_text += (
                f"{speaker}: {item['content']}\n"
            )

        # ⑦ 构建系统提示词
        system_prompt = """你是一个专业的电影推荐助手。

请根据提供的电影资料回答用户的问题。

要求：
1. 优先结合用户明确表达的电影偏好。
2. 参考历史对话，理解用户的追问。
3. 不要重复推荐历史中已经推荐过的电影。
4. 不要编造资料库没有提供的电影事实。
5. 电影名称使用加粗格式。
"""

        # ⑧ 构建用户提示词
        user_prompt = f"""【历史对话】
{history_text}

【用户偏好】
{user_pref if user_pref else "暂无"}

【电影数据库检索结果】
{context_text}

【当前问题】
{message}
"""

        # ⑨ 流式生成回答，同时写回用户偏好
        partial_message = ""

        for chunk in llm.generate_response(
            system_prompt,
            user_prompt
        ):
            partial_message += chunk

            # 第一个值：聊天内容
            # 第二个值：写回 Gradio State 的偏好
            yield partial_message, user_pref

        if not partial_message:
            yield "暂时未生成有效回答。", user_pref

    except Exception as e:
        print(f"系统异常: {e}", flush=True)
        yield "抱歉，系统暂时出现异常。", user_pref


if __name__ == "__main__":
    print("🚀 [1/3] 正在初始化 Gradio 界面...")

    with gr.Blocks() as demo:
        user_pref_state = gr.State("")

        gr.Markdown("# 🎬 专属 AI 电影推荐助手")

        chat = gr.ChatInterface(
            fn=respond,
            additional_inputs=[user_pref_state],
            additional_outputs=[user_pref_state]
        )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        inbrowser=False,
        show_error=True
    )
