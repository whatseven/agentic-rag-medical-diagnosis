import py2neo

def neo4j_diagnosis_search(disease_name: str) -> str:
    """
    Search for diagnosis-related core information based on disease name from English knowledge graph
    
    Args:
        disease_name: Disease name in English
        
    Returns:
        str: Formatted diagnosis-related information text (including complete etiology, treatment departments, complications)
    """
    try:
        # Connect to Neo4j database
        client = py2neo.Graph("bolt://localhost:7687", user="neo4j", password="neo4j123", name="neo4j")
        
        # Query disease basic information (disease cause)
        disease_query = f"""
        MATCH (n:Disease{{name:'{disease_name}'}})
        RETURN n.cause AS cause
        """
        disease_result = client.run(disease_query).data()
        
        if not disease_result:
            return ""
        
        disease_info = disease_result[0]
        
        # Query treatment departments
        department_query = f"""
        MATCH (d:Disease{{name:'{disease_name}'}})-[:DISEASE_BELONGS_TO_DEPARTMENT]->(dept:Department)
        RETURN dept.name AS department_name
        """
        department_result = client.run(department_query).data()
        departments = [record['department_name'] for record in department_result]
        
        # Query complications
        complication_query = f"""
        MATCH (d:Disease{{name:'{disease_name}'}})-[:DISEASE_COMPLICATION]->(comp:Disease)
        RETURN comp.name AS complication_disease
        """
        complication_result = client.run(complication_query).data()
        complications = [record['complication_disease'] for record in complication_result]
        
        # Format output
        result_text = f"Disease Name: {disease_name}\n\n"
        
        # Disease cause (complete version, to be simplified later by LLM)
        cause = disease_info.get('cause', '')
        if cause:
            result_text += f"Disease Cause: {cause}\n\n"
        
        # Treatment departments
        if departments:
            result_text += f"Treatment Departments: {' '.join(departments)}\n\n"
        
        # Complications
        if complications:
            result_text += f"Complications: {' '.join(complications)}\n\n"
        
        return result_text.strip()
        
    except Exception as e:
        print(f"Neo4j diagnosis query error: {e}")
        return ""
