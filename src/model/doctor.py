import requests
import json
import os
from src.model.config import MODELS, DEFAULT_MODEL
from src.model.prompt import DOCTOR_SYSTEM_PROMPT

def load_disease_list(file_path: str = None) -> str:
    """
    加载疾病列表并格式化为字符串
    
    Args:
        file_path: 疾病列表文件路径
    
    Returns:
        格式化的疾病列表字符串，如果文件不存在或为空则返回空字符串
    """
    if not file_path or not os.path.exists(file_path):
        return ""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                return ""
            
            # 尝试解析为Python列表格式
            try:
                import ast
                disease_list = ast.literal_eval(content)
                if isinstance(disease_list, list):
                    formatted_list = ", ".join(disease_list)
                    return f"可选疾病列表：{formatted_list}\n\n"
                else:
                    return ""
            except:
                # 如果不是列表格式，按行读取
                lines = content.split('\n')
                diseases = [line.strip() for line in lines if line.strip()]
                if diseases:
                    formatted_list = ", ".join(diseases)
                    return f"可选疾病列表：{formatted_list}\n\n"
                return ""
                
    except Exception as e:
        print(f"读取疾病列表文件出错: {str(e)}")
        return ""

def diagnose(user_input: str, vector_results: list, graph_data: dict, model_name: str = DEFAULT_MODEL, disease_list_file: str = None, diagnostic_suggestions: dict = None) -> str:
    """
    进行最终医疗诊断
    
    Args:
        user_input: 用户症状描述
        vector_results: 过滤后的向量库搜索结果
        graph_data: 图数据库查询的详细医学资料
        model_name: 使用的模型名称
        disease_list_file: 疾病列表文件路径，可选
        diagnostic_suggestions: 诊断建议，可选
    
    Returns:
        诊断结果文本
    """
    # 格式化向量库结果
    vector_info = ""
    for i, result in enumerate(vector_results, 1):
        vector_info += f"{i}. 疾病：{result.get('name', '')}\n"
        vector_info += f"   描述：{result.get('desc', '')}\n"
        vector_info += f"   症状：{result.get('symptom', '')}\n"
        vector_info += f"   相似度：{result.get('similarity_score', 0):.4f}\n\n"
    
    # 格式化图数据库结果
    graph_info = ""
    if graph_data:
        for key, value in graph_data.items():
            graph_info += f"{key}：{value}\n\n"
    
    # 加载疾病列表
    disease_list_info = load_disease_list(disease_list_file)
    
    # 格式化诊断建议
    suggestions_info = ""
    if diagnostic_suggestions:
        recommended_diseases = diagnostic_suggestions.get("recommended_diseases", [])
        reason = diagnostic_suggestions.get("reason", "")
        if recommended_diseases:
            suggestions_info = f"推荐疾病：{', '.join(recommended_diseases)}\n"
            if reason:
                suggestions_info += f"建议原因：{reason}\n"
            suggestions_info += "请参考以上建议进行诊断。"
        else:
            suggestions_info = "暂无特殊建议"
    else:
        suggestions_info = "暂无特殊建议"
    
    # 手动替换占位符来避免与JSON格式冲突
    system_prompt = DOCTOR_SYSTEM_PROMPT.replace("{vector_results}", vector_info)
    system_prompt = system_prompt.replace("{disease_list}", disease_list_info)
    system_prompt = system_prompt.replace("{graph_data}", graph_info)
    system_prompt = system_prompt.replace("{diagnostic_suggestions}", suggestions_info)
    
    # 获取模型配置
    model_config = MODELS.get(model_name, MODELS[DEFAULT_MODEL])
    
    # 构建请求
    headers = {
        "Authorization": f"Bearer {model_config['api_key']}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model_config["model_name"],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        "temperature": 0.7,
        "max_tokens": 500
    }

    try:
        response = requests.post(
            f"{model_config['base_url']}/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        diagnosis_text = result["choices"][0]["message"]["content"]
        
        return diagnosis_text
        
    except Exception as e:
        return f"诊断过程中发生错误: {str(e)}"
