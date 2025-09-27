import json
import re

def extract_diagnosis_result(content):
    """
    从大模型返回内容中提取诊断结果
    
    Args:
        content: 大模型返回的完整内容
    
    Returns:
        dict: {"need_more_info": bool, "diseases": []}
    """
    try:
        # 使用正则表达式提取<diagnose>标签中的内容
        pattern = r'<diagnose>(.*?)</diagnose>'
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            json_str = match.group(1).strip()
            result = json.loads(json_str)
            return result
        else:
            return {"error": "未找到有效的诊断结果格式"}
            
    except json.JSONDecodeError:
        return {"error": "JSON格式解析失败"}
    except Exception as e:
        return {"error": f"结果提取失败: {str(e)}"}
