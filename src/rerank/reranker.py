import requests
import json

def rerank_diseases(query_symptom, milvus_results):

    if not milvus_results:
        return []
    

    documents = []
    for result in milvus_results:
        symptom = result.get('symptom', '[]')
        desc = result.get('desc', '')

        try:
            symptom_list = json.loads(symptom)
            symptom_text = ','.join(symptom_list)
        except:
            symptom_text = symptom
        
        document = f"症状：{symptom_text} 描述：{desc}"
        documents.append(document)

    url = "https://api.siliconflow.cn/v1/rerank"
    payload = {
        "model": "Qwen/Qwen3-Reranker-8B",
        "query": query_symptom,
        "documents": documents
    }
    headers = {
        "Authorization": "Bearer <>",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        rerank_result = response.json()

        reranked_diseases = []
        for item in rerank_result['results']:
            original_index = item['index']
            disease_data = milvus_results[original_index].copy()
            disease_data['relevance_score'] = item['relevance_score']
            reranked_diseases.append(disease_data)
        
        return reranked_diseases
        
    except Exception as e:
        print(f"Rerank API调用失败: {e}")
        return milvus_results 

def rerank_diseases_with_topk(query_symptom, milvus_results, top_k=None):

    reranked_results = rerank_diseases(query_symptom, milvus_results)

    if top_k is not None and len(reranked_results) > top_k:
        return reranked_results[:top_k]
    
    return reranked_results
