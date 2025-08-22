"""

@File       utils.py
@Author     小明
@Date       2025/8/22 09:56
@Version    V0.0.1
"""
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr


def get_model():
    return ChatOpenAI(
        base_url='https://api.deepseek.com/v1',
        api_key=SecretStr('sk-f76e425189e14ee4a538bb044a798b10'),
        model='deepseek-chat',
    )


def get_gpt_model():
    return ChatOpenAI(
        base_url='https://api.openai-hk.com/v1',
        api_key=SecretStr('hk-u0fe491000056163a172ba46ed3fa78f9fa750e5654edc4f'),
        # model='gpt-4o',
        model='deepseek-r1',
    )


def get_embedding():
    return OpenAIEmbeddings(
        base_url='https://open.bigmodel.cn/api/paas/v4',
        api_key=SecretStr('65365a45b72e4b3185678ef3c7726830.YEHGBdvyKtDR6U4W'),
        model='embedding-3',
    )
