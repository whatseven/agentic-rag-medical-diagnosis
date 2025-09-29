import requests
import json
import re
from .config import MODELS, DEFAULT_MODEL
from .prompt import DISEASE_CAUSE_REWRITE_PROMPT

def rewrite_disease_cause(raw_cause: str, disease_name: str = "", model_name: str = DEFAULT_MODEL) -> str:

    if not raw_cause or not raw_cause.strip():
        return ""
    
    try:
    
        model_config = MODELS.get(model_name, MODELS[DEFAULT_MODEL])

        prompt = DISEASE_CAUSE_REWRITE_PROMPT.format(
            disease_name=disease_name,
            raw_cause=raw_cause
        )

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
        

        simplified_cause = extract_simplified_cause(response_text)
        
        return simplified_cause if simplified_cause else raw_cause[:50] 
        
    except Exception as e:
        print(f"病因简化出错: {str(e)}")
        return raw_cause[:50] 

def extract_simplified_cause(response_text: str) -> str:

    try:
       
        pattern = r'<simplified_cause>\s*(.*?)\s*</simplified_cause>'
        match = re.search(pattern, response_text, re.DOTALL)
        
        if match:
            simplified_text = match.group(1).strip()
            return simplified_text

        lines = response_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('-'):
                return line[:50]  
        
        return ""
        
    except Exception as e:
        print(f"提取简化病因时出错: {str(e)}")
        return ""
