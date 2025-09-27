import requests
import json

def get_embedding(text: str, api_token: str) -> list:
    """
    使用Siliconflow API将给定的文本向量化。

    Args:
        text (str): 需要向量化的文本。
        api_token (str): 用于Siliconflow API认证的Bearer Token。

    Returns:
        list: 文本的嵌入向量（4096维度），如果请求失败则返回空列表。
    """
    url = "https://api.siliconflow.cn/v1/embeddings"
    
    # 维度设置为4096
    payload = {
        "model": "Qwen/Qwen3-Embedding-8B", 
        "input": text
    }
    
    headers = {
        "Authorization": f"Bearer {api_token}",
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # 如果请求失败（例如4xx或5xx错误），则抛出HTTPError

        result = response.json()
        if "data" in result and len(result["data"]) > 0 and "embedding" in result["data"][0]:
            return result["data"][0]["embedding"]
        else:
            print(f"警告: API响应中未找到嵌入数据或数据格式不正确: {result}")
            return []
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP错误发生: {http_err} - 响应内容: {response.text}")
        return []
    except requests.exceptions.ConnectionError as conn_err:
        print(f"连接错误发生: {conn_err}")
        return []
    except requests.exceptions.Timeout as timeout_err:
        print(f"请求超时: {timeout_err}")
        return []
    except requests.exceptions.RequestException as req_err:
        print(f"请求过程中发生错误: {req_err}")
        return []
    except json.JSONDecodeError as json_err:
        print(f"JSON解码错误: {json_err} - 响应内容: {response.text}")
        return []

if __name__ == "__main__":
    # 请替换为你的实际API Token
    my_api_token = "" 

    # 示例用法
    text_to_embed = "Silicon flow embedding online: fast, affordable, and high-quality embedding services. come try it out!"
    embedding = get_embedding(text_to_embed, my_api_token)

    if embedding:
        print(f"文本: \"{text_to_embed}\"")
        print(f"嵌入向量维度: {len(embedding)}")
        print(f"嵌入向量（前5个元素）: {embedding[:5]}...")
    else:
        print("未能获取嵌入向量。")


