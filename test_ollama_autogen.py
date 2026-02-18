import autogen

config_list = [{
    "model": "mistral",
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama",
    "temperature": 0.0,
    "max_tokens": 3,
}]

assistant = autogen.AssistantAgent(
    "assistant",
    llm_config={"config_list": config_list},
    code_execution_config=False,
    system_message="Output exactly the requested token(s) and nothing else."
)


user = autogen.UserProxyAgent(
    "user",
    human_input_mode="NEVER",
    code_execution_config=False
)

assistant.initiate_chat(user, message="Output exactly: PONG")
