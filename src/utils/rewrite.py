import re
import json
import openai
from ..model.config import MODELS, DEFAULT_MODEL
from ..model.prompt import SYMPTOM_REWRITE_PROMPT

def call_symptom_api(dialog_text, model_name=None):

    if model_name is None:
        model_name = DEFAULT_MODEL
    
    if model_name not in MODELS:
        raise ValueError(f"不支持的模型: {model_name}")
    
    config = MODELS[model_name]
    
    client = openai.OpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"]
    )
    
    response = client.chat.completions.create(
        model=config["model_name"],
        messages=[
            {"role": "system", "content": SYMPTOM_REWRITE_PROMPT},
            {"role": "user", "content": dialog_text}
        ],
        temperature=0.1,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

def extract_symptoms_from_response(response_text):
  
    pattern = r'<symptom>(.*?)</symptom>'
    match = re.search(pattern, response_text, re.DOTALL)
    
    if match:
        try:
            symptom_json = match.group(1).strip()
            symptom_data = json.loads(symptom_json)
            return symptom_data.get("symptom", [])
        except json.JSONDecodeError:
            print(f"JSON解析失败: {match.group(1)}")
            return []
    else:
        print("未找到<symptom>标签")
        return []
