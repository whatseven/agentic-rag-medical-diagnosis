import sys
import os
from typing import List, Dict, Any
from pymilvus import connections, db, Collection, AnnSearchRequest, WeightedRanker, MilvusClient

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'embedding'))
from embedding import get_embedding

def search_similar_diseases(query: str, top_k: int = 5) -> List[Dict[str, Any]]:

    host = "localhost"
    port = "19530"
    api_token = ""
    database_name = "llm_medication"
    collection_name = "medication2" 
    partition_name = "knowledge_base"
    dimension = 4096
    
    try:

        client = MilvusClient(uri=f"http://{host}:{port}")

        client.using_database(database_name)

        query_vector = get_embedding(query, api_token)
        if not query_vector or len(query_vector) != dimension:
            return []

        search_param_1 = {
            "data": [query_vector],
            "anns_field": "symptom_vector",
            "param": {"nprobe": 16},
            "limit": top_k * 2  
        }
        request_1 = AnnSearchRequest(**search_param_1)

        search_param_2 = {
            "data": [query_vector],
            "anns_field": "desc_vector", 
            "param": {"nprobe": 16},
            "limit": top_k * 2  
        }
        request_2 = AnnSearchRequest(**search_param_2)

        ranker = WeightedRanker(0.6, 0.4)

        results = client.hybrid_search(
            collection_name=collection_name,
            reqs=[request_1, request_2],
            ranker=ranker,
            limit=top_k,
            output_fields=["oid", "name", "desc", "symptom"],
            partition_names=[partition_name]
        )

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
