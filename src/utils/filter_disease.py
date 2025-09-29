def filter_diseases_by_name(vector_results: list, target_diseases: list) -> list:

    if not vector_results or not target_diseases:
        return []

    target_set = set(target_diseases)

    filtered_results = []
    for result in vector_results:
        if result.get('name') in target_set:
            filtered_results.append(result)
    
    return filtered_results
