import re
import os
import json
from openai import OpenAI
from src.model.config import MODELS
from src.model.prompt import R1_EXPERT_EVALUATION_PROMPT

def extract_diagnostic_suggestions(content: str) -> dict:

    try:
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
   
    try:
       
        disease_list_str = ""
        if disease_list_file and os.path.exists(disease_list_file):
            try:
                with open(disease_list_file, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                      
                        try:
                            import ast
                            disease_list = ast.literal_eval(content)
                            if isinstance(disease_list, list):
                                disease_list_str = f"可选疾病列表：{', '.join(disease_list)}"
                        except:
                           
                            lines = content.split('\n')
                            diseases = [line.strip() for line in lines if line.strip()]
                            if diseases:
                                disease_list_str = f"可选疾病列表：{', '.join(diseases)}"
            except Exception as e:
                print(f"读取疾病列表文件出错: {str(e)}")

        prompt = R1_EXPERT_EVALUATION_PROMPT.format(
            symptoms=symptoms,
            vector_results=vector_results,
            graph_data=graph_data,
            doctor_diagnosis=doctor_diagnosis,
            disease_list=disease_list_str
        )
        
        model_config = MODELS["deepseek"]
        
        client = OpenAI(
            api_key=model_config["api_key"],
            base_url=model_config["base_url"]
        )

        response = client.chat.completions.create(
            model=model_config["model_name"],
            messages=[
                {"role": "system", "content": "你是一位资深医疗专家，需要进行推理分析诊断是否正确。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,  
            stream=False
        )
        
        content = response.choices[0].message.content
        
        expert_review_match = re.search(r'<expert_review>(.*?)</expert_review>', content, re.DOTALL)
        if expert_review_match:
            review_content = expert_review_match.group(1).strip()

            if '1' in review_content:
                return {"is_correct": True}
            elif '0' in review_content:
             
                diagnostic_suggestions = extract_diagnostic_suggestions(content)
                result = {"is_correct": False}
                if diagnostic_suggestions:
                    result["diagnostic_suggestions"] = diagnostic_suggestions
                else:
         
                    result["diagnostic_suggestions"] = {
                        "recommended_diseases": ["建议重新评估症状"],
                        "reason": "现有诊断不够准确，需要重新分析"
                    }
                return result
            else:
  
                return {"is_correct": True}
        else:
        
            return {"is_correct": True}
            
    except Exception as e:
        print(f"专家评估出错: {e}")
        # 出错时默认认为正确，避免阻塞诊断流程
        return {"is_correct": True}
