import json
import re

def extract_diagnosis_result(content):

    try:
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
