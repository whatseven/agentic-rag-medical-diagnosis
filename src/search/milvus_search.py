import sys
import os
from typing import List, Dict, Any
from pymilvus import connections, db, Collection, AnnSearchRequest, WeightedRanker, MilvusClient

# 添加embedding模块路径,双向量字段搜索
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'embedding'))
from embedding import get_embedding

def search_similar_diseases(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    在Milvus中搜索相似疾病（使用混合搜索symptom_vector和desc_vector）
    
    Args:
        query: 用户查询文本
        top_k: 返回前k个结果，默认10
        
    Returns:
        List[Dict]: 包含疾病信息的字典列表，不包含向量字段
    """
    # 配置信息
    host = "localhost"
    port = "19530"
    api_token = "sk-uqdmjvjhiyggiznhihznibdgsnpjdwdscqfqrywpgolismns"
    database_name = "llm_medication"
    collection_name = "medication2"  # 修改为medication2
    partition_name = "knowledge_base"
    dimension = 4096
    
    try:
        # 使用MilvusClient进行连接
        client = MilvusClient(uri=f"http://{host}:{port}")
        
        # 切换数据库
        client.using_database(database_name)
        
        # 向量化查询
        query_vector = get_embedding(query, api_token)
        if not query_vector or len(query_vector) != dimension:
            return []
        
        # 创建第一个搜索请求：症状向量搜索
        search_param_1 = {
            "data": [query_vector],
            "anns_field": "symptom_vector",
            "param": {"nprobe": 16},
            "limit": top_k * 2  # 扩大搜索范围以获得更好的混合结果
        }
        request_1 = AnnSearchRequest(**search_param_1)
        
        # 创建第二个搜索请求：描述向量搜索
        search_param_2 = {
            "data": [query_vector],
            "anns_field": "desc_vector", 
            "param": {"nprobe": 16},
            "limit": top_k * 2  # 扩大搜索范围以获得更好的混合结果
        }
        request_2 = AnnSearchRequest(**search_param_2)
        
        # 创建加权重排序器：症状权重0.6，描述权重0.4
        ranker = WeightedRanker(0.6, 0.4)
        
        # 执行混合搜索
        results = client.hybrid_search(
            collection_name=collection_name,
            reqs=[request_1, request_2],
            ranker=ranker,
            limit=top_k,
            output_fields=["oid", "name", "desc", "symptom"],
            partition_names=[partition_name]
        )
        
        # 处理结果
        if not results or len(results[0]) == 0:
            return []
        
        search_results = []
        for hit in results[0]:
            result_dict = {
                'oid': hit.entity.get('oid'),
                'name': hit.entity.get('name'),
                'desc': hit.entity.get('desc'),
                'symptom': hit.entity.get('symptom'),
                'similarity_score': float(hit.distance)
            }
            search_results.append(result_dict)
        
        return search_results
        
    except Exception as e:
        print(f"混合搜索错误: {e}")
        return []
