import os
from openai import OpenAI, APITimeoutError, RateLimitError, APIConnectionError, BadRequestError

class LLMClient:
    def __init__(self):
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("未找到 DEEPSEEK_API_KEY 环境变量！")
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com", timeout=15.0)

    def generate_hyde_doc(self, query_text):
        """HyDE：让大模型脑补一段假电影简介"""
        system_prompt = "你是一名专业的电影编剧。请根据用户的简短提问，虚构一段200字左右的电影简介。注意：不要包含具体的电影名字，只描述剧情风格、氛围和特点。"
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": query_text}],
                temperature=0.7,
                timeout=10.0
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"⚠️ HyDE 生成失败，降级使用原句: {e}")
            return query_text # 如果生成失败，兜底用原话去搜

    def generate_response(self, system_prompt, user_prompt):
        """流式生成最终回答"""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                temperature=0.5,
                stream=True
            )
            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except APITimeoutError:
            yield "⏱️ 抱歉，大模型响应超时了，请稍后再试！"
        except RateLimitError:
            yield "🚦 抱歉，您的提问太频繁啦，请休息一下再问！"
        except APIConnectionError:
            yield "🌐 网络连接失败，请检查网络或稍后再试！"
        except BadRequestError:
            yield "🛑 请求格式错误，可能是上下文太长，请精简问题再试！"
        except Exception as e:
            print(f"⚠️ LLM API 报错: {e}")
            yield "抱歉，AI 大脑当前正在休息，请稍后再试～"

    def rewrite_query(self, history, message):
        """利用大模型把用户的代词问题补全为完整的检索句"""
        if not history:
            return message

        history_text = ""

        # 获取最近两轮对话（最多四条消息）
        for item in (history or [])[-4:]:
            if not isinstance(item, dict):
                continue

            role = item.get("role")
            content = item.get("content")

            if role not in ("user", "assistant"):
                continue

            if not isinstance(content, str):
                continue

            speaker = "用户" if role == "user" else "AI"
            history_text += f"{speaker}：{content}\n"

        prompt = f"""根据以下历史对话，把用户的最新问题补全为一个语义完整、独立的检索问题。
如果最新问题没有代词或不需要结合上下文，直接原样返回用户的最新问题。
【历史对话】：
{history_text}
【最新问题】：{message}
请只输出补全后的完整问题，不要任何解释。"""

        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                timeout=5.0
            )
            return response.choices[0].message.content.strip()
        except Exception:
            return message # 如果重写失败，兜底用原话去搜

    def extract_preference(self, message):
        """提取用户的电影偏好类型"""
        prompt = f"""用户说：“{message}”。
请提取用户喜欢的电影类型（如：科幻、喜剧、动作、恐怖等）。只输出一个词，不要任何解释。
如果无法提取出明确的电影类型，请输出“无”。"""
        try:
            response = self.client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                timeout=5.0
            )
            pref = response.choices[0].message.content.strip()
            return pref if pref != "无" else None
        except Exception:
            return None
