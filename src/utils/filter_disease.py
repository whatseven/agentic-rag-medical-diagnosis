def filter_diseases_by_name(vector_results: list, target_diseases: list) -> list:
    """
    根据目标疾病名称列表过滤向量库搜索结果
    
    Args:
        vector_results: 向量库搜索结果列表，每个元素包含name字段
        target_diseases: 目标疾病名称列表
    
    Returns:
        过滤后的向量库结果列表，只包含匹配的疾病
    """
    if not vector_results or not target_diseases:
        return []
    
    # 创建目标疾病名称集合用于快速查找
    target_set = set(target_diseases)
    
    # 过滤匹配的疾病
    filtered_results = []
    for result in vector_results:
        if result.get('name') in target_set:
            filtered_results.append(result)
    
    return filtered_results
