import requests
import json

def rerank_diseases(query_symptom, milvus_results):
    """
    对Milvus检索结果进行重排序
    
    Args:
        query_symptom: 大模型改写后的症状，如"腹痛"
        milvus_results: Milvus返回的疾病列表
    
    Returns:
        重排序后的疾病列表，按相关度从高到低
    """
    if not milvus_results:
        return []
    
    # 构建documents - 组合症状和描述
    documents = []
    for result in milvus_results:
        symptom = result.get('symptom', '[]')
        desc = result.get('desc', '')
        # 解析症状JSON字符串
        try:
            symptom_list = json.loads(symptom)
            symptom_text = ','.join(symptom_list)
        except:
            symptom_text = symptom
        
        document = f"症状：{symptom_text} 描述：{desc}"
        documents.append(document)
    
    # 调用rerank API
    url = "https://api.siliconflow.cn/v1/rerank"
    payload = {
        "model": "Qwen/Qwen3-Reranker-8B",
        "query": query_symptom,
        "documents": documents
    }
    headers = {
        "Authorization": "Bearer sk-uqdmjvjhiyggiznhihznibdgsnpjdwdscqfqrywpgolismns",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()
        
        rerank_result = response.json()
        
        # 根据rerank结果重新排序原始数据
        reranked_diseases = []
        for item in rerank_result['results']:
            original_index = item['index']
            disease_data = milvus_results[original_index].copy()
            disease_data['relevance_score'] = item['relevance_score']
            reranked_diseases.append(disease_data)
        
        return reranked_diseases
        
    except Exception as e:
        print(f"Rerank API调用失败: {e}")
        return milvus_results  # 失败时返回原始结果

def rerank_diseases_with_topk(query_symptom, milvus_results, top_k=None):
    """
    对Milvus检索结果进行重排序并支持top_k截断
    
    Args:
        query_symptom: 大模型改写后的症状，如"腹痛"
        milvus_results: Milvus返回的疾病列表
        top_k: 返回前top_k个结果，None表示返回全部
    
    Returns:
        重排序后的疾病列表，按相关度从高到低，截断到top_k个
    """
    # 调用原有的重排序函数
    reranked_results = rerank_diseases(query_symptom, milvus_results)
    
    # 如果指定了top_k，则截断结果
    if top_k is not None and len(reranked_results) > top_k:
        return reranked_results[:top_k]
    
    return reranked_results
