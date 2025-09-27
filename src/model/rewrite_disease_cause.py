import requests
import json
import re
from .config import MODELS, DEFAULT_MODEL
from .prompt import DISEASE_CAUSE_REWRITE_PROMPT

def rewrite_disease_cause(raw_cause: str, disease_name: str = "", model_name: str = DEFAULT_MODEL) -> str:
    """
    将详细的病因描述简化为有助于诊断的关键要点
    
    Args:
        raw_cause: 原始完整的病因描述
        disease_name: 疾病名称（可选，提供上下文）
        model_name: 使用的模型名称
    
    Returns:
        str: 简化后的病因描述
    """
    if not raw_cause or not raw_cause.strip():
        return ""
    
    try:
        # 获取模型配置
        model_config = MODELS.get(model_name, MODELS[DEFAULT_MODEL])
        
        # 格式化提示词
        prompt = DISEASE_CAUSE_REWRITE_PROMPT.format(
            disease_name=disease_name,
            raw_cause=raw_cause
        )
        
        # 构建请求
        headers = {
            "Authorization": f"Bearer {model_config['api_key']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": model_config["model_name"],
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 200
        }

        response = requests.post(
            f"{model_config['base_url']}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        response_text = result["choices"][0]["message"]["content"]
        
        # 提取简化后的病因
        simplified_cause = extract_simplified_cause(response_text)
        
        return simplified_cause if simplified_cause else raw_cause[:50]  # 失败时返回截断版本
        
    except Exception as e:
        print(f"病因简化出错: {str(e)}")
        return raw_cause[:50]  # 出错时返回截断版本

def extract_simplified_cause(response_text: str) -> str:
    """
    从模型响应中提取简化的病因描述
    
    Args:
        response_text: 模型的完整响应文本
    
    Returns:
        str: 提取到的简化病因，如果提取失败返回空字符串
    """
    try:
        # 查找<simplified_cause>标签
        pattern = r'<simplified_cause>\s*(.*?)\s*</simplified_cause>'
        match = re.search(pattern, response_text, re.DOTALL)
        
        if match:
            simplified_text = match.group(1).strip()
            return simplified_text
        
        # 如果没有标签，尝试提取主要内容
        lines = response_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('-'):
                return line[:50]  # 限制长度
        
        return ""
        
    except Exception as e:
        print(f"提取简化病因时出错: {str(e)}")
        return ""
