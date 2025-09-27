import sys
import os
from typing import List, Dict, Any
from pymilvus import connections, db, Collection, AnnSearchRequest, WeightedRanker, MilvusClient

# Add embedding module path, dual-vector field search
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'embedding'))
from embedding import get_embedding

def search_similar_diseases(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for similar diseases in Milvus (using hybrid search for symptom_vector and desc_vector)
    
    Args:
        query: User query text in English
        top_k: Return top k results, default 5
        
    Returns:
        List[Dict]: List of dictionaries containing disease information, excluding vector fields
    """
    # Configuration
    host = "localhost"
    port = "19530"
    api_token = "sk-uqdmjvjhiyggiznhihznibdgsnpjdwdscqfqrywpgolismns"
    database_name = "llm_medication"
    collection_name = "medication2_en"  # 英文版本collection
    partition_name = "knowledge_base_en"
    dimension = 4096
    
    try:
        # Connect using MilvusClient
        client = MilvusClient(uri=f"http://{host}:{port}")
        
        # Switch database
        client.using_database(database_name)
        
        # Vectorize query
        query_vector = get_embedding(query, api_token)
        if not query_vector or len(query_vector) != dimension:
            return []
        
        # Create first search request: symptom vector search
        search_param_1 = {
            "data": [query_vector],
            "anns_field": "symptom_vector",
            "param": {"nprobe": 16},
            "limit": top_k * 2  # Expand search range for better hybrid results
        }
        request_1 = AnnSearchRequest(**search_param_1)
        
        # Create second search request: description vector search
        search_param_2 = {
            "data": [query_vector],
            "anns_field": "desc_vector", 
            "param": {"nprobe": 16},
            "limit": top_k * 2  # Expand search range for better hybrid results
        }
        request_2 = AnnSearchRequest(**search_param_2)
        
        # Create weighted ranker: symptom weight 0.6, description weight 0.4
        ranker = WeightedRanker(0.6, 0.4)
        
        # Execute hybrid search
        results = client.hybrid_search(
            collection_name=collection_name,
            reqs=[request_1, request_2],
            ranker=ranker,
            limit=top_k,
            output_fields=["oid", "name", "desc", "symptom"],
            partition_names=[partition_name]
        )
        
        # Process results
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
        print(f"Hybrid search error: {e}")
        return []
