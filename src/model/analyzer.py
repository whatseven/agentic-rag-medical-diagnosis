from openai import OpenAI
from src.model.config import MODELS, DEFAULT_MODEL
from src.model.prompt import SYSTEM_PROMPT
from src.utils.extract_diagnosis import extract_diagnosis_result

def analyze_diagnosis(user_input, disease_results, model_name=None):

    if model_name is None:
        model_name = DEFAULT_MODEL
    
    try:
      
        model_config = MODELS[model_name]
        
       
        client = OpenAI(
            api_key=model_config["api_key"],
            base_url=model_config["base_url"]
        )
       
        disease_info = ""
        for i, disease in enumerate(disease_results, 1):
            disease_info += f"{i}. {disease['name']}\n"
            disease_info += f"   描述：{disease['desc']}\n"
            disease_info += f"   症状：{disease['symptom']}\n"
            disease_info += f"   相似度：{disease['similarity_score']:.3f}\n\n"
        
       
        system_prompt = SYSTEM_PROMPT.replace("{disease_results}", disease_info)
        
        response = client.chat.completions.create(
            model=model_config["model_name"],
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            stream=False
        )
        
       
        content = response.choices[0].message.content
        return extract_diagnosis_result(content)
        
    except Exception as e:
        return {"error": f"分析失败: {str(e)}"}


