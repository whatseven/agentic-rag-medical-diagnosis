from openai import OpenAI
from src.model.config import MODELS, DEFAULT_MODEL
from src.model.prompt import SYSTEM_PROMPT
from src.utils.extract_diagnosis import extract_diagnosis_result

def analyze_diagnosis(user_input, disease_results, model_name=None):
    """
    分析是否能够基于向量库结果完成诊断
    
    Args:
        user_input: 用户原始症状描述
        disease_results: 向量库搜索返回的疾病列表
        model_name: 使用的模型名称，默认使用DEFAULT_MODEL
    
    Returns:
        dict: {"need_more_info": bool, "diseases": []}
    """
    if model_name is None:
        model_name = DEFAULT_MODEL
    
    try:
        # 获取模型配置
        model_config = MODELS[model_name]
        
        # 初始化客户端
        client = OpenAI(
            api_key=model_config["api_key"],
            base_url=model_config["base_url"]
        )
        
        # 格式化疾病信息
        disease_info = ""
        for i, disease in enumerate(disease_results, 1):
            disease_info += f"{i}. {disease['name']}\n"
            disease_info += f"   描述：{disease['desc']}\n"
            disease_info += f"   症状：{disease['symptom']}\n"
            disease_info += f"   相似度：{disease['similarity_score']:.3f}\n\n"
        
        # 手动替换占位符来避免与JSON格式冲突
        system_prompt = SYSTEM_PROMPT.replace("{disease_results}", disease_info)
        
        # 调用大模型
        response = client.chat.completions.create(
            model=model_config["model_name"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            stream=False
        )
        
        # 提取结果
        content = response.choices[0].message.content
        return extract_diagnosis_result(content)
        
    except Exception as e:
        return {"error": f"分析失败: {str(e)}"}


