import py2neo

def neo4j_diagnosis_search(disease_name: str) -> str:
    """
    根据疾病名称搜索诊断相关的核心信息
    
    Args:
        disease_name: 疾病名称
        
    Returns:
        str: 格式化的诊断相关信息文本（包含完整病因、治疗科室、并发症）
    """
    try:
        # 连接Neo4j数据库
        client = py2neo.Graph("bolt://localhost:7687", user="neo4j", password="neo4j123", name="neo4j")
        
        # 查询疾病基本信息（疾病病因）
        disease_query = f"""
        MATCH (n:疾病{{名称:'{disease_name}'}})
        RETURN n.疾病病因 AS 病因
        """
        disease_result = client.run(disease_query).data()
        
        if not disease_result:
            return ""
        
        disease_info = disease_result[0]
        
        # 查询治疗科室
        department_query = f"""
        MATCH (d:疾病{{名称:'{disease_name}'}})-[:疾病所属科目]->(dept:科目)
        RETURN dept.名称 AS 科室名称
        """
        department_result = client.run(department_query).data()
        departments = [record['科室名称'] for record in department_result]
        
        # 查询并发症
        complication_query = f"""
        MATCH (d:疾病{{名称:'{disease_name}'}})-[:疾病并发疾病]->(comp:疾病)
        RETURN comp.名称 AS 并发疾病
        """
        complication_result = client.run(complication_query).data()
        complications = [record['并发疾病'] for record in complication_result]
        
        # 格式化输出
        result_text = f"疾病名称：{disease_name}\n\n"
        
        # 疾病病因（完整版本，后续通过大模型精简）
        cause = disease_info.get('病因', '')
        if cause:
            result_text += f"疾病病因：{cause}\n\n"
        
        # 治疗科室
        if departments:
            result_text += f"治疗科室：{' '.join(departments)}\n\n"
        
        # 并发症
        if complications:
            result_text += f"并发症：{' '.join(complications)}\n\n"
        
        return result_text.strip()
        
    except Exception as e:
        print(f"Neo4j诊断查询错误: {e}")
        return ""
