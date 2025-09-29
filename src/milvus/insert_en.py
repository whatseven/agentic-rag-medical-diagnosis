import json
import sys
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pymilvus import connections, db, Collection, FieldSchema, CollectionSchema, DataType, utility
import time


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'embedding'))
from embedding import get_embedding

class MilvusInserter:
    def __init__(self, host="localhost", port="19530"):
        self.host = host
        self.port = port
        self.api_token = "" # embedding model api token
        self.database_name = "llm_medication"
        self.collection_name = "medication2_en"  
        self.partition_name = "knowledge_base_en"
        self.dimension = 4096
        self.batch_size = 20
        self.failed_oids = []  
        
    def connect_milvus(self):
        
        print("正在连接Milvus...")
        connections.connect("default", host=self.host, port=self.port)
        print("Milvus连接成功!")
        
    def create_database(self):
        
        try:
            db.create_database(self.database_name)
            print(f"数据库 '{self.database_name}' 创建成功")
        except Exception:
            print(f"数据库 '{self.database_name}' 已存在")
        
        # 使用数据库
        db.using_database(self.database_name)
        
    def create_collection_schema(self):
        
        fields = [
            FieldSchema(name="oid", dtype=DataType.VARCHAR, max_length=50, is_primary=True),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=1000),
            FieldSchema(name="desc", dtype=DataType.VARCHAR, max_length=30000),  
            FieldSchema(name="symptom", dtype=DataType.VARCHAR, max_length=5000),  
            FieldSchema(name="symptom_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),  
            FieldSchema(name="desc_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)  
        ]
        
        schema = CollectionSchema(fields, "English Medical Knowledge Base - Support dual-vector retrieval for symptom and desc")
        return schema
        
    def create_collection(self):
        
        if utility.has_collection(self.collection_name):
            print(f"Collection '{self.collection_name}' 已存在")
            collection = Collection(self.collection_name)
        else:
            schema = self.create_collection_schema()
            collection = Collection(self.collection_name, schema)
            print(f"Collection '{self.collection_name}' 创建成功")
            
        
        if not collection.has_partition(self.partition_name):
            collection.create_partition(self.partition_name)
            print(f"分区 '{self.partition_name}' 创建成功")
        else:
            print(f"分区 '{self.partition_name}' 已存在")
            
        return collection
        
    def vectorize_symptoms(self, symptoms_list):
       
        if not symptoms_list:
            return [0.0] * self.dimension
            
        
        symptoms_text = " ".join(symptoms_list)
        vector = get_embedding(symptoms_text, self.api_token)
        
        if not vector or len(vector) != self.dimension:
            return [0.0] * self.dimension
            
        return vector

    def vectorize_desc(self, desc_text):
        
        if not desc_text or not desc_text.strip():
            return [0.0] * self.dimension
            
        vector = get_embedding(desc_text, self.api_token)
        
        if not vector or len(vector) != self.dimension:
            return [0.0] * self.dimension
            
        return vector
        
    def truncate_text(self, text, max_length):
        
        if not text:
            return ""
        if len(text) > max_length:
            print(f"警告: 文本被截断，原长度: {len(text)}, 截断后: {max_length}")
            return text[:max_length]
        return text

    def process_record(self, record):
        
        try:
            oid = record.get("_id", {}).get("$oid", "")
            if not oid:
                return None
                
            
            symptoms = record.get("symptom", [])
            symptom_vector = self.vectorize_symptoms(symptoms)
            
            
            desc_text = record.get("desc", "")
            desc_vector = self.vectorize_desc(desc_text)
            
            
            symptom_failed = (symptom_vector == [0.0] * self.dimension and symptoms)
            desc_failed = (desc_vector == [0.0] * self.dimension and desc_text.strip())
            
           
            if symptom_failed or desc_failed:
                self.failed_oids.append({
                    "oid": oid,
                    "symptom_failed": symptom_failed,
                    "desc_failed": desc_failed
                })
                
            
            processed_record = {
                "oid": self.truncate_text(oid, 50),  
                "name": self.truncate_text(record.get("name", ""), 1000),
                "desc": self.truncate_text(desc_text, 30000),  
                "symptom": self.truncate_text(json.dumps(symptoms, ensure_ascii=False), 5000), 
                "symptom_vector": symptom_vector,  
                "desc_vector": desc_vector  
            }
            
            return processed_record
            
        except Exception as e:
            print(f"处理记录失败 {record.get('_id', {}).get('$oid', 'unknown')}: {e}")
            return None
            
    def load_data(self, file_path):
        
        print(f"正在加载数据文件: {file_path}")
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if len(line) < 3:  
                    continue
                try:
                    
                    record = json.loads(line)
                    data.append(record)
                except json.JSONDecodeError as e:
                    print(f"警告: 第{line_num}行JSON解析失败: {e}")
                    continue
        print(f"数据加载完成，共 {len(data)} 条记录")
        return data
        
    def validate_batch_data(self, batch_data):
        
        for i, record in enumerate(batch_data):
            
            if len(record.get("oid", "")) > 50:
                print(f"错误: 记录{i} oid长度超限: {len(record['oid'])}")
                return False
            if len(record.get("name", "")) > 500:
                print(f"错误: 记录{i} name长度超限: {len(record['name'])}")
                return False
            if len(record.get("desc", "")) > 30000:
                print(f"错误: 记录{i} desc长度超限: {len(record['desc'])}")
                return False
            if len(record.get("symptom", "")) > 5000:
                print(f"错误: 记录{i} symptom长度超限: {len(record['symptom'])}")
                return False
        return True

    def insert_data_batch(self, collection, batch_data):
       
        if not batch_data:
            return
            
        
        if not self.validate_batch_data(batch_data):
            print("批次数据验证失败，跳过插入")
            return
            
        
        insert_data = []
        for field in ["oid", "name", "desc", "symptom", "symptom_vector", "desc_vector"]:
            insert_data.append([record[field] for record in batch_data])
            
        collection.insert(insert_data, partition_name=self.partition_name)
        
    def run(self, file_path):
        
        try:
           
            self.connect_milvus()
            
            
            self.create_database()
            
         
            collection = self.create_collection()
            
        
            raw_data = self.load_data(file_path)
            
           
            print(f"开始处理数据，批量大小: {self.batch_size}")
            print("将同时向量化symptom和desc字段...")
            batches = [raw_data[i:i+self.batch_size] for i in range(0, len(raw_data), self.batch_size)]
            
            processed_count = 0
            for batch_idx, batch in enumerate(tqdm(batches, desc="处理批次")):
                processed_batch = []
                
                
                for record in tqdm(batch, desc=f"批次 {batch_idx+1}/{len(batches)}", leave=False):
                    processed_record = self.process_record(record)
                    if processed_record:
                        processed_batch.append(processed_record)
                        processed_count += 1
                        
                
                if processed_batch:
                    self.insert_data_batch(collection, processed_batch)
                    
               
                time.sleep(0.5)
                
            
            print("正在为symptom_vector创建向量索引...")
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            collection.create_index("symptom_vector", index_params)
            
            print("正在为desc_vector创建向量索引...")
            collection.create_index("desc_vector", index_params)
            
            
            collection.load()
            
            print(f"\n✅ 数据插入完成!")
            print(f"总记录数: {len(raw_data)}")
            print(f"成功处理: {processed_count}")
            print(f"向量化失败的记录数: {len(self.failed_oids)}")
            
            if self.failed_oids:
                print(f"向量化失败的详情（前10个）:")
                for i, failed_info in enumerate(self.failed_oids[:10]):
                    oid = failed_info["oid"]
                    failures = []
                    if failed_info["symptom_failed"]:
                        failures.append("symptom")
                    if failed_info["desc_failed"]:
                        failures.append("desc")
                    print(f"  {i+1}. OID: {oid}, 失败字段: {', '.join(failures)}")
                
                if len(self.failed_oids) > 10:
                    print(f"  ... 还有 {len(self.failed_oids) - 10} 个失败记录")
                
        except Exception as e:
            print(f"执行过程中发生错误: {e}")
            raise

if __name__ == "__main__":
    file_path = ""
    
    inserter = MilvusInserter()
    inserter.run(file_path)
