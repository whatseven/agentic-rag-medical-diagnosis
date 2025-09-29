import py2neo

def neo4j_diagnosis_search(disease_name: str) -> str:

    try:

        client = py2neo.Graph("bolt://localhost:7687", user="neo4j", password="neo4j123", name="neo4j")
        
        disease_query = f"""
        MATCH (n:疾病{{名称:'{disease_name}'}})
        RETURN n.疾病病因 AS 病因
        """
        disease_result = client.run(disease_query).data()
        
        if not disease_result:
            return ""
        
        disease_info = disease_result[0]

        department_query = f"""
        MATCH (d:疾病{{名称:'{disease_name}'}})-[:疾病所属科目]->(dept:科目)
        RETURN dept.名称 AS 科室名称
        """
        department_result = client.run(department_query).data()
        departments = [record['科室名称'] for record in department_result]

        complication_query = f"""
        MATCH (d:疾病{{名称:'{disease_name}'}})-[:疾病并发疾病]->(comp:疾病)
        RETURN comp.名称 AS 并发疾病
        """
        complication_result = client.run(complication_query).data()
        complications = [record['并发疾病'] for record in complication_result]

        result_text = f"疾病名称：{disease_name}\n\n"

        cause = disease_info.get('病因', '')
        if cause:
            result_text += f"疾病病因：{cause}\n\n"

        if departments:
            result_text += f"治疗科室：{' '.join(departments)}\n\n"

        if complications:
            result_text += f"并发症：{' '.join(complications)}\n\n"
        
        return result_text.strip()
        
    except Exception as e:
        print(f"Neo4j诊断查询错误: {e}")
        return ""
