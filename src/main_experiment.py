import json
import sys
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import re


project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')
sys.path.insert(0, project_root)


import importlib.util
main_file_path = os.path.join(project_root, 'main_rerank copy_simple_iteration copy.py')
spec = importlib.util.spec_from_file_location("main_rerank copy_simple_iteration copy", main_file_path)
main_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(main_module)
medical_diagnosis_pipeline = main_module.medical_diagnosis_pipeline

def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """
    add id
    
    Args:
        file_path
    
    Returns:
        list_with_id
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if line:
                item = json.loads(line)
                item['id'] = idx  
                data.append(item)
    return data

def preprocess_dialog(dialog: List[str]) -> str:
    """
    
    
    Args:
        dialog: dialogue
    
    Returns:
        string
    """
    return '\n'.join(dialog)

def extract_diseases_from_diagnosis(diagnosis_text: str) -> List[str]:
    
    try:
       
        pattern = r'<final_diagnosis>\s*(\{.*?\})\s*</final_diagnosis>'
        match = re.search(pattern, diagnosis_text, re.DOTALL)
        
        if match:
            json_str = match.group(1)
            diagnosis_data = json.loads(json_str)
            diseases = diagnosis_data.get('diseases', [])
            return diseases if isinstance(diseases, list) else [diseases]
        
        
        disease_patterns = [
            r'诊断[：:]\s*([^。\n]+)',
            r'可能的疾病[：:]\s*([^。\n]+)',
            r'初步诊断[：:]\s*([^。\n]+)',
            r'考虑[：:]?\s*([^。\n，,]+)',
        ]
        
        for pattern in disease_patterns:
            matches = re.findall(pattern, diagnosis_text)
            if matches:
                return [match.strip() for match in matches]
        
        return ["未能提取疾病信息"]
        
    except Exception as e:
        return [f"提取错误: {str(e)}"]

def process_single_item(item: Dict[str, Any], disease_list_file: str = None) -> Dict[str, Any]:
    
    try:
        
        dialog_text = preprocess_dialog(item['original_dialog'])
        
      
        start_time = time.time()
        diagnosis_result = medical_diagnosis_pipeline(dialog_text, disease_list_file=disease_list_file, silent_mode=True)
        end_time = time.time()
        
       
        predicted_diseases = extract_diseases_from_diagnosis(diagnosis_result)
        
        result = {
            'id': item['id'],
            'ground_truth_disease': item['disease'],
            'ground_truth_label': item['label'],
            'input_dialog': dialog_text,
            'raw_diagnosis': diagnosis_result,
            'predicted_diseases': predicted_diseases,
            'processing_time': round(end_time - start_time, 2),
            'status': 'success'
        }
        
        print(f"✓ 完成ID {item['id']}: {len(dialog_text[:50])}... -> {predicted_diseases}")
        return result
        
    except Exception as e:
        print(f"✗ ID {item['id']} 处理失败: {str(e)}")
        return {
            'id': item['id'],
            'ground_truth_disease': item['disease'],
            'ground_truth_label': item['label'],
            'input_dialog': preprocess_dialog(item['original_dialog']),
            'raw_diagnosis': f"处理错误: {str(e)}",
            'predicted_diseases': ["处理失败"],
            'processing_time': 0,
            'status': 'error'
        }

def evaluate_dataset(input_file: str, output_file: str, max_workers: int = 100, limit: int = None, disease_list_file: str = None):
    """
    评估整个数据集
    
    Args:
        input_file: 输入数据集文件路径
        output_file: 输出结果文件路径
        max_workers: 并发线程数
        limit: 限制处理的数据条数，None表示处理全部
        disease_list_file: 疾病列表文件路径，可选
    """
    print(f"开始评估数据集: {input_file}")
    print(f"并发线程数: {max_workers}")
    
    if disease_list_file:
        print(f"使用疾病列表约束: {disease_list_file}")
    else:
        print("不使用疾病列表约束")
    
    # 加载数据集
    print("加载数据集...")
    dataset = load_dataset(input_file)
    
    if limit:
        dataset = dataset[:limit]
        print(f"限制处理前 {limit} 条数据")
    
    print(f"总共 {len(dataset)} 条数据")
    
    # 并发处理
    print("\n开始并发处理...")
    start_time = time.time()
    
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务，传入疾病列表文件
        future_to_item = {executor.submit(process_single_item, item, disease_list_file): item for item in dataset}
        
        # 收集结果
        for future in as_completed(future_to_item):
            result = future.result()
            results.append(result)
    
    # 按id排序确保顺序正确
    results.sort(key=lambda x: x['id'])
    
    end_time = time.time()
    total_time = round(end_time - start_time, 2)
    
    print(f"\n处理完成! 总耗时: {total_time}秒")
    print(f"平均每条耗时: {round(total_time/len(dataset), 2)}秒")
    
    # 统计成功失败数
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = len(results) - success_count
    print(f"成功: {success_count}, 失败: {error_count}")
    
    # 保存结果
    print(f"\n保存结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print("评估完成!")
    return results



if __name__ == "__main__":
    # 配置文件路径
    input_file = ""
    output_dir = ""
    output_file = os.path.join(output_dir, "")
    

    disease_list_file = ""  
     disease_list_file = ""  
    
    os.makedirs(output_dir, exist_ok=True)
    
    
    print("=== DiaMed evaluation===")
    choice = input("mode:\n1. 测试模式(前10条)\n2. 小批量(前50条)\n3. 全量评估\n请选择(1/2/3): ").strip()
    
    if choice == '1':
        limit = 10
        max_workers = 5
    elif choice == '2':
        limit = 50
        max_workers = 50
    elif choice == '3':
        limit = None
        max_workers = 20
    else:
        print("无效选择，使用测试模式")
        limit = 10
        max_workers = 3
    
    # 执行评估
    results = evaluate_dataset(input_file, output_file, max_workers, limit, disease_list_file)
    

    
    print(f"\n结果已保存到: {output_file}")
