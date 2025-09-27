import json
import sys
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pymilvus import connections, db, Collection, FieldSchema, CollectionSchema, DataType, utility
import time

# 添加embedding模块路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'embedding'))
from embedding import get_embedding

class MilvusInserter:
    def __init__(self, host="localhost", port="19530"):
        self.host = host
        self.port = port
        self.api_token = "" # embedding model api token
        self.database_name = "llm_medication"
        self.collection_name = "medication"  # 修改collection名称
        self.partition_name = "knowledge_base"
        self.dimension = 4096
        self.batch_size = 20
        self.failed_oids = []  # 记录向量化失败的oid
        
    def connect_milvus(self):
        """连接Milvus数据库"""
        print("正在连接Milvus...")
        connections.connect("default", host=self.host, port=self.port)
        print("Milvus连接成功!")
        
    def create_database(self):
        """创建数据库（如果不存在）"""
        try:
            db.create_database(self.database_name)
            print(f"数据库 '{self.database_name}' 创建成功")
        except Exception:
            print(f"数据库 '{self.database_name}' 已存在")
        
        # 使用数据库
        db.using_database(self.database_name)
        
    def create_collection_schema(self):
        """创建Collection的Schema"""
        fields = [
            FieldSchema(name="oid", dtype=DataType.VARCHAR, max_length=50, is_primary=True),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="desc", dtype=DataType.VARCHAR, max_length=30000),  # 原始desc数据
            FieldSchema(name="symptom", dtype=DataType.VARCHAR, max_length=5000),  # 原始症状数组的JSON字符串
            FieldSchema(name="symptom_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),  # symptom向量
            FieldSchema(name="desc_vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension)  # desc向量
        ]
        
        schema = CollectionSchema(fields, "医疗知识库 - 支持symptom和desc双向量检索")
        return schema
        
    def create_collection(self):
        """创建Collection（如果不存在）"""
        if utility.has_collection(self.collection_name):
            print(f"Collection '{self.collection_name}' 已存在")
            collection = Collection(self.collection_name)
        else:
            schema = self.create_collection_schema()
            collection = Collection(self.collection_name, schema)
            print(f"Collection '{self.collection_name}' 创建成功")
            
        # 创建分区（如果不存在）
        if not collection.has_partition(self.partition_name):
            collection.create_partition(self.partition_name)
            print(f"分区 '{self.partition_name}' 创建成功")
        else:
            print(f"分区 '{self.partition_name}' 已存在")
            
        return collection
        
    def vectorize_symptoms(self, symptoms_list):
        """将症状列表转换为向量"""
        if not symptoms_list:
            return [0.0] * self.dimension
            
        # 将症状数组拼接成字符串
        symptoms_text = " ".join(symptoms_list)
        vector = get_embedding(symptoms_text, self.api_token)
        
        if not vector or len(vector) != self.dimension:
            return [0.0] * self.dimension
            
        return vector

    def vectorize_desc(self, desc_text):
        """将描述文本转换为向量"""
        if not desc_text or not desc_text.strip():
            return [0.0] * self.dimension
            
        vector = get_embedding(desc_text, self.api_token)
        
        if not vector or len(vector) != self.dimension:
            return [0.0] * self.dimension
            
        return vector
        
    def truncate_text(self, text, max_length):
        """截断文本到指定长度"""
        if not text:
            return ""
        return text[:max_length] if len(text) > max_length else text

    def process_record(self, record):
        """处理单条记录"""
        try:
            oid = record.get("_id", {}).get("$oid", "")
            if not oid:
                return None
                
            # 向量化症状
            symptoms = record.get("symptom", [])
            symptom_vector = self.vectorize_symptoms(symptoms)
            
            # 向量化描述
            desc_text = record.get("desc", "")
            desc_vector = self.vectorize_desc(desc_text)
            
            # 检查向量化是否失败
            symptom_failed = (symptom_vector == [0.0] * self.dimension and symptoms)
            desc_failed = (desc_vector == [0.0] * self.dimension and desc_text.strip())
            
            # 如果任一向量化失败，记录oid
            if symptom_failed or desc_failed:
                self.failed_oids.append({
                    "oid": oid,
                    "symptom_failed": symptom_failed,
                    "desc_failed": desc_failed
                })
                
            # 准备数据（包含所有字段）
            processed_record = {
                "oid": oid,
                "name": self.truncate_text(record.get("name", ""), 500),
                "desc": self.truncate_text(desc_text, 30000),  # 保存原始desc数据
                "symptom": self.truncate_text(json.dumps(symptoms, ensure_ascii=False), 5000),  # 保存原始症状数组
                "symptom_vector": symptom_vector,  # symptom向量
                "desc_vector": desc_vector  # desc向量
            }
            
            return processed_record
            
        except Exception as e:
            print(f"处理记录失败 {record.get('_id', {}).get('$oid', 'unknown')}: {e}")
            return None
            
    def load_data(self, file_path):
        """加载数据"""
        print(f"正在加载数据文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"数据加载完成，共 {len(data)} 条记录")
        return data
        
    def insert_data_batch(self, collection, batch_data):
        """批量插入数据"""
        if not batch_data:
            return
            
        # 组织数据为列格式（包含新的desc_vector字段）
        insert_data = []
        for field in ["oid", "name", "desc", "symptom", "symptom_vector", "desc_vector"]:
            insert_data.append([record[field] for record in batch_data])
            
        collection.insert(insert_data, partition_name=self.partition_name)
        
    def run(self, file_path):
        """主运行函数"""
        try:
            # 连接Milvus
            self.connect_milvus()
            
            # 创建数据库
            self.create_database()
            
            # 创建Collection
            collection = self.create_collection()
            
            # 加载数据
            raw_data = self.load_data(file_path)
            
            # 分批处理数据
            print(f"开始处理数据，批量大小: {self.batch_size}")
            print("将同时向量化symptom和desc字段...")
            batches = [raw_data[i:i+self.batch_size] for i in range(0, len(raw_data), self.batch_size)]
            
            processed_count = 0
            for batch_idx, batch in enumerate(tqdm(batches, desc="处理批次")):
                processed_batch = []
                
                # 处理当前批次
                for record in tqdm(batch, desc=f"批次 {batch_idx+1}/{len(batches)}", leave=False):
                    processed_record = self.process_record(record)
                    if processed_record:
                        processed_batch.append(processed_record)
                        processed_count += 1
                        
                # 插入当前批次
                if processed_batch:
                    self.insert_data_batch(collection, processed_batch)
                    
                # 避免API调用过于频繁
                time.sleep(0.5)
                
            # 创建向量索引（为两个向量字段都创建索引）
            print("正在为symptom_vector创建向量索引...")
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            collection.create_index("symptom_vector", index_params)
            
            print("正在为desc_vector创建向量索引...")
            collection.create_index("desc_vector", index_params)
            
            # 加载Collection
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