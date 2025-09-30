import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model.rewrite_query import process_dialog_symptoms
from src.search.milvus_search import search_similar_diseases
from src.rerank.reranker import rerank_diseases_with_topk
from src.model.analyzer import analyze_diagnosis
from src.search.neo4j_diagnose import neo4j_diagnosis_search
from src.model.doctor import diagnose
from src.model.rewrite_disease_cause import rewrite_disease_cause
from src.model.iteration import iterative_diagnose

def parse_neo4j_result(neo4j_text: str) -> dict:
    result = {
        'disease_name': '',
        'cause': '',
        'department': '',
        'complications': ''
    }
    
    try:
        lines = neo4j_text.strip().split('\n')
        current_field = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith('疾病名称：'):
                result['disease_name'] = line.replace('疾病名称：', '').strip()
            elif line.startswith('疾病病因：'):
                current_field = 'cause'
                result['cause'] = line.replace('疾病病因：', '').strip()
            elif line.startswith('治疗科室：'):
                current_field = 'department'
                result['department'] = line.replace('治疗科室：', '').strip()
            elif line.startswith('并发症：'):
                current_field = 'complications'
                result['complications'] = line.replace('并发症：', '').strip()
            elif current_field and line:
                result[current_field] += line           
    except Exception as e:
        print(f"解析图数据库结果出错: {str(e)}")
    return result

def process_graph_data_with_simplified_cause(disease_name: str, neo4j_text: str, model_name: str = None) -> str:
    try:
        parsed_data = parse_neo4j_result(neo4j_text)
        if not parsed_data['cause']:
            print(f"警告: 疾病 {disease_name} 没有病因信息")
            return ""
        simplified_cause = rewrite_disease_cause(
            raw_cause=parsed_data['cause'],
            disease_name=disease_name,
            model_name=model_name
        )
        if not simplified_cause:
            print(f"警告: 疾病 {disease_name} 病因简化失败，跳过该疾病")
            return ""
        result_text = f"疾病名称：{disease_name}\n\n"
        result_text += f"疾病病因：{simplified_cause}\n\n"
        if parsed_data['department']:
            result_text += f"治疗科室：{parsed_data['department']}\n\n"
        if parsed_data['complications']:
            result_text += f"并发症：{parsed_data['complications']}\n\n"
        return result_text.strip()
    except Exception as e:
        print(f"处理疾病 {disease_name} 的图数据库信息出错: {str(e)}，跳过该疾病")
        return ""

def get_initial_diagnosis_data(user_input: str, model_name: str = None, top_k: int = 10, silent_mode: bool = False) -> dict:
    try:
        if not silent_mode:
            print("获取初始诊断数据...")
            print(f"用户输入: {user_input}")
        if not silent_mode:
            print(f"\n步骤1: 向量搜索(top_k={top_k})...")
        milvus_results = search_similar_diseases(user_input, top_k=top_k)
        if not silent_mode:
            print(f"搜索到 {len(milvus_results)} 个疾病")
        if not milvus_results:
            return {
                "vector_results": [],
                "graph_data": {},
                "success": False,
                "error": "未找到相关疾病信息，请咨询专业医生。"
            }
        rerank_top_k = 5  
        if not silent_mode:
            print(f"\n步骤2: 重排序并截断到top{rerank_top_k}...")
        reranked_results = rerank_diseases_with_topk(user_input, milvus_results, top_k=rerank_top_k)
        if not silent_mode:
            print(f"重排序完成，从{len(milvus_results)}个筛选到{len(reranked_results)}个结果")
        if not silent_mode:
            print("\n步骤3: 分析诊断...")
        analysis_result = analyze_diagnosis(user_input, reranked_results, model_name)
        if not silent_mode:
            print(f"分析结果: {analysis_result}")
        if 'error' in analysis_result:
            return {
                "vector_results": reranked_results,
                "graph_data": {},
                "success": False,
                "error": analysis_result['error']
            }
        need_more_info = analysis_result.get('need_more_info', False)
        target_diseases = analysis_result.get('diseases', [])
        if need_more_info and target_diseases:
            if not silent_mode:
                print(f"\n需要更多信息，目标疾病: {target_diseases}")
            if not silent_mode:
                print("\n步骤4: 图数据库查询和病因简化...")
            graph_data = {}
            for disease_name in target_diseases:
                if not silent_mode:
                    print(f"查询疾病: {disease_name}")
                disease_info = neo4j_diagnosis_search(disease_name)
                if disease_info:
                    processed_info = process_graph_data_with_simplified_cause(
                        disease_name, disease_info, model_name
                    )
                    if processed_info:
                        graph_data[disease_name] = processed_info
                        if not silent_mode:
                            print(f"✓ 疾病 {disease_name} 信息处理完成")
                    else:
                        if not silent_mode:
                            print(f"✗ 疾病 {disease_name} 信息处理失败，跳过")
                else:
                    if not silent_mode:
                        print(f"✗ 疾病 {disease_name} 未找到图数据库信息")
            filtered_results = reranked_results
            if not silent_mode:
                print(f"保留所有向量库结果: {len(filtered_results)} 个")
                print(f"获取图数据库信息的疾病: {len(graph_data)} 个") 
        else:
            if not silent_mode:
                print("\n无需更多信息，直接使用重排序结果")
            filtered_results = reranked_results
            graph_data = {}
        if not silent_mode:
            print("\n初始数据获取完成!")
        return {
            "vector_results": filtered_results,
            "graph_data": graph_data,
            "success": True
        }
    except Exception as e:
        error_msg = f"获取初始诊断数据出错: {str(e)}"
        if not silent_mode:
            print(error_msg)
        return {
            "vector_results": [],
            "graph_data": {},
            "success": False,
            "error": error_msg
        }
        
def medical_diagnosis_pipeline(user_input: str, model_name: str = None, disease_list_file: str = None, silent_mode: bool = False) -> str:
    max_retries = 3
    rejection_count = 0  
    previous_suggestions = None  
    if not silent_mode:
        print("=== 开始医疗诊断流程===")
    if not silent_mode:
        print(f"\n{'='*60}")
        print("获取诊断所需的基础数据...")
        print(f"{'='*60}")
    initial_data = get_initial_diagnosis_data(
        user_input=user_input,
        model_name=model_name,
        top_k=5,  
        silent_mode=silent_mode
    )
    if not initial_data["success"]:
        return initial_data.get("error", "获取诊断数据失败")
    vector_results_str = ""
    for i, disease in enumerate(initial_data["vector_results"], 1):
        vector_results_str += f"{i}. {disease.get('name', 'Unknown')}\n"
        vector_results_str += f"   描述：{disease.get('desc', 'No description')}\n"
        vector_results_str += f"   症状：{disease.get('symptom', 'No symptoms')}\n"
        vector_results_str += f"   相似度：{disease.get('similarity_score', 0):.3f}\n\n"
    graph_data_str = ""
    for disease_name, disease_info in initial_data["graph_data"].items():
        graph_data_str += f"{disease_info}\n\n"
    symptoms_str = user_input  
    if not silent_mode:
        print("基础数据获取完成，开始迭代诊断...")
    for attempt in range(max_retries):
        if not silent_mode:
            print(f"\n{'='*60}")
            print(f"第 {attempt + 1} 次诊断尝试")
            if previous_suggestions and not silent_mode:
                print(f"使用上轮建议：{previous_suggestions.get('recommended_diseases', [])}")
            print(f"{'='*60}")
        try:
            
            if not silent_mode:
                print("调用doctor模块进行诊断...")
            
            diagnosis_result = diagnose(
                user_input, 
                initial_data["vector_results"], 
                initial_data["graph_data"], 
                model_name, 
                disease_list_file, 
                previous_suggestions  
            )
            
            if not silent_mode:
                print(f"诊断完成: {diagnosis_result[:100]}...")
            if not silent_mode:
                print(f"\n{'='*40}")
                print("R1专家评估诊断质量...")
                print(f"{'='*40}")
            
            expert_review = iterative_diagnose(
                symptoms=symptoms_str,
                vector_results=vector_results_str,
                graph_data=graph_data_str,
                doctor_diagnosis=diagnosis_result,
                disease_list_file=disease_list_file
            )
            
            if not silent_mode:
                print(f"评估结果: {'通过' if expert_review['is_correct'] else '驳回'}")
            
            if expert_review["is_correct"]:
                if not silent_mode:
                    print(f"\n{'='*60}")
                    print("诊断正确，流程结束")
                    print(f"{'='*60}")
                return diagnosis_result
            else:
                rejection_count += 1
               
                previous_suggestions = expert_review.get("diagnostic_suggestions")
                if not silent_mode and previous_suggestions:
                    print(f"建议：{previous_suggestions.get('recommended_diseases', [])}")
                
                if attempt == max_retries - 1:  
                    if not silent_mode:
                        print("诊断有误，已达到最大重试次数...")
                    break  # 跳出循环
                else:
                    if not silent_mode:
                        print(f"诊断有误，准备第 {attempt + 2} 次重试...")
                
        except Exception as e:
            if not silent_mode:
                print(f"第 {attempt + 1} 次诊断过程出错: {str(e)}")
            continue
    
    
    if not silent_mode:
        print(f"\n{'='*60}")
        print(f"迭代诊断已达到最大重试次数 (共被驳回{rejection_count}次)，使用doctor模块进行最终诊断")
        print(f"{'='*60}")
    
    try:
        
        if not silent_mode:
            print("调用doctor模块进行最终诊断...")
        
        final_diagnosis = diagnose(
            user_input, 
            initial_data["vector_results"], 
            initial_data["graph_data"], 
            model_name, 
            disease_list_file, 
            previous_suggestions  
        )
        
        if not silent_mode:
            print("doctor模块最终诊断完成")
        
        return final_diagnosis
        
    except Exception as e:
        return f"doctor模块最终诊断失败: {str(e)}"

if __name__ == "__main__":
    # 示例调用
    test_input = ""
    result = medical_diagnosis_pipeline(test_input)
    print(f"\n最终诊断结果:\n{result}")
