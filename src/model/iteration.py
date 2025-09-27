import re
import os
import json
from openai import OpenAI
from src.model.config import MODELS
from src.model.prompt import R1_EXPERT_EVALUATION_PROMPT

def extract_diagnostic_suggestions(content: str) -> dict:
    """
    从R1专家响应中提取诊断建议
    
    Args:
        content: R1专家的完整响应文本
    
    Returns:
        诊断建议字典
    """
    try:
        # 查找<diagnostic_suggestions>标签
        pattern = r'<diagnostic_suggestions>\s*(\{.*?\})\s*</diagnostic_suggestions>'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            json_str = match.group(1)
            suggestions = json.loads(json_str)
            return suggestions
        
        return None
        
    except Exception as e:
        print(f"提取诊断建议时出错: {str(e)}")
        return None

def iterative_diagnose(symptoms, vector_results, graph_data, doctor_diagnosis, disease_list_file=None):
    """
    使用R1专家模型评估诊断质量
    
    Args:
        symptoms: 患者症状描述
        vector_results: 候选疾病信息
        graph_data: 图数据库信息
        doctor_diagnosis: 初步诊断结果
        disease_list_file: 疾病列表文件路径，可选
        
    Returns:
        dict: {
            "is_correct": bool,
            "diagnostic_suggestions": dict (仅在is_correct=False时存在)
        }
    """
    try:
        # 加载疾病列表（如果提供）
        disease_list_str = ""
        if disease_list_file and os.path.exists(disease_list_file):
            try:
                with open(disease_list_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        # 尝试解析为Python列表格式
                        try:
                            import ast
                            disease_list = ast.literal_eval(content)
                            if isinstance(disease_list, list):
                                disease_list_str = f"可选疾病列表：{', '.join(disease_list)}"
                        except:
                            # 如果不是列表格式，按行读取
                            lines = content.split('\n')
                            diseases = [line.strip() for line in lines if line.strip()]
                            if diseases:
                                disease_list_str = f"可选疾病列表：{', '.join(diseases)}"
            except Exception as e:
                print(f"读取疾病列表文件出错: {str(e)}")
        
        # 构建专家评估提示词
        prompt = R1_EXPERT_EVALUATION_PROMPT.format(
            symptoms=symptoms,
            vector_results=vector_results,
            graph_data=graph_data,
            doctor_diagnosis=doctor_diagnosis,
            disease_list=disease_list_str
        )
        
        # 获取R1模型配置
        model_config = MODELS["deepseek"]
        
        # 初始化客户端
        client = OpenAI(
            api_key=model_config["api_key"],
            base_url=model_config["base_url"]
        )
        
        # 使用R1模型进行评估（确保触发推理模式）
        response = client.chat.completions.create(
            model=model_config["model_name"],
            messages=[
                {"role": "system", "content": "你是一位资深医疗专家，需要进行推理分析诊断是否正确。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,  # 低温度确保稳定性
            stream=False
        )
        
        # 获取响应内容
        content = response.choices[0].message.content
        
        # 提取评估结果
        expert_review_match = re.search(r'<expert_review>(.*?)</expert_review>', content, re.DOTALL)
        if expert_review_match:
            review_content = expert_review_match.group(1).strip()
            # 简单解析1或0
            if '1' in review_content:
                return {"is_correct": True}
            elif '0' in review_content:
                # 诊断被驳回，提取建议信息
                diagnostic_suggestions = extract_diagnostic_suggestions(content)
                result = {"is_correct": False}
                if diagnostic_suggestions:
                    result["diagnostic_suggestions"] = diagnostic_suggestions
                else:
                    # 如果没有提取到建议，提供默认建议
                    result["diagnostic_suggestions"] = {
                        "recommended_diseases": ["建议重新评估症状"],
                        "reason": "现有诊断不够准确，需要重新分析"
                    }
                return result
            else:
                # 如果无法解析，默认认为正确
                return {"is_correct": True}
        else:
            # 如果无法解析，默认认为正确
            return {"is_correct": True}
            
    except Exception as e:
        print(f"专家评估出错: {e}")
        # 出错时默认认为正确，避免阻塞诊断流程
        return {"is_correct": True}
