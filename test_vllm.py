from openai import OpenAI

client = OpenAI(api_key="EMPTY", base_url="http://192.168.141.73:8002/v1", timeout=3600)

resp = client.chat.completions.create(
    model="nanbeige4.3-33b",
    messages=[{"role": "user", "content": "你是谁，你是谁训练的"}],
    temperature=0, top_p=0.95, stream=True,
    extra_body={"chat_template_kwargs": {"enable_thinking": False}},
)
for chunk in resp:
    if chunk.choices and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)