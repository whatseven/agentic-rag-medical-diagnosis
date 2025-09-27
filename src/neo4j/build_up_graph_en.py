import os
import re
import py2neo
from tqdm import tqdm
import argparse
import json


#导入普通实体
def import_entity(client,type,entity):
    def create_node(client,type,name):
        # 使用参数化查询避免特殊字符问题
        order = f"CREATE (n:{type} {{name: $name}})"
        client.run(order, name=name)

    print(f'正在导入{type}类数据')
    for en in tqdm(entity):
        create_node(client,type,en)
#导入疾病类实体
def import_disease_data(client,type,entity):
    print(f'正在导入{type}类数据')
    for disease in tqdm(entity):
        # 只保留简单的字符串类型属性，避免复杂嵌套结构
        node_props = {}
        for key, value in disease.items():
            if isinstance(value, str):
                node_props[key] = value
            elif isinstance(value, (int, float)):
                node_props[key] = str(value)
            elif isinstance(value, list) and all(isinstance(item, str) for item in value):
                # 如果是字符串列表，转换为用分号分隔的字符串
                node_props[key] = "; ".join(value)
            # 跳过复杂的嵌套结构
        
        node = py2neo.Node(type, **node_props)
        client.create(node)

def create_all_relationship(client,all_relationship):
    def create_relationship(client,type1, name1,relation, type2,name2):
        # 使用参数化查询避免特殊字符问题
        order = f"MATCH (a:{type1} {{name: $name1}}), (b:{type2} {{name: $name2}}) CREATE (a)-[r:{relation}]->(b)"
        client.run(order, name1=name1, name2=name2)
    print("正在导入关系.....")
    for type1, name1,relation, type2,name2  in tqdm(all_relationship):
        create_relationship(client,type1, name1,relation, type2,name2)

if __name__ == "__main__":
    #连接数据库的一些参数
    parser = argparse.ArgumentParser(description="通过medical.json文件,创建一个知识图谱")
    parser.add_argument('--website', type=str, default='bolt://localhost:7687', help='neo4j的连接网站')
    parser.add_argument('--user', type=str, default='neo4j', help='neo4j的用户名')
    parser.add_argument('--password', type=str, default='neo4j123', help='neo4j的密码')
    parser.add_argument('--dbname', type=str, default='neo4j', help='数据库名称')
    args = parser.parse_args()

    #连接...
    client = py2neo.Graph(args.website, user=args.user, password=args.password, name=args.dbname)

    #将数据库中的内容删光
    #is_delete = input('注意:是否删除neo4j上的所有实体 (y/n):')
    #if is_delete=='y':
    #    client.run("match (n) detach delete (n)")

    with open('llm_medication/src/data/medical_new_2_en.json','r',encoding='utf-8') as f:
        all_data = f.read().split('\n')
    
    #所有实体
    all_entity = {
        "Disease": [],
        "Drug": [],
        "Food": [],
        "CheckItem":[],
        "Department":[],
        "Symptom":[],
        "Treatment":[],
        "Manufacturer":[],
    }
    
    # 实体间的关系
    relationship = []
    for i,data in enumerate(all_data):
        if (len(data) < 3):
            continue
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            continue

        disease_name = data.get("name","")
        all_entity["Disease"].append({
            "name":disease_name,
            "desc": data.get("desc", ""),
            "cause": data.get("cause", ""),
            "prevent": data.get("prevent", ""),
            "cure_lasttime":data.get("cure_lasttime",""),
            "cured_prob": data.get("cured_prob", ""),
            "easy_get": data.get("easy_get", ""),
        })

        drugs = data.get("common_drug", []) + data.get("recommand_drug", [])
        all_entity["Drug"].extend(drugs)  # 添加药品实体
        if drugs:
            relationship.extend([("Disease", disease_name, "DISEASE_USES_DRUG", "Drug",durg)for durg in drugs])

        do_eat = data.get("do_eat",[])+data.get("recommand_eat",[])
        no_eat = data.get("not_eat",[])
        all_entity["Food"].extend(do_eat+no_eat)
        if do_eat:
            relationship.extend([("Disease", disease_name,"DISEASE_RECOMMENDED_FOOD","Food",f) for f in do_eat])
        if no_eat:
            relationship.extend([("Disease", disease_name, "DISEASE_AVOIDED_FOOD", "Food", f) for f in no_eat])

        check = data.get("check", [])
        all_entity["CheckItem"].extend(check)
        if check:
            relationship.extend([("Disease", disease_name, "DISEASE_REQUIRES_CHECK", "CheckItem",ch) for ch in check])

        cure_department=data.get("cure_department", [])
        all_entity["Department"].extend(cure_department)
        if cure_department:
            relationship.append(("Disease", disease_name, "DISEASE_BELONGS_TO_DEPARTMENT", "Department",cure_department[-1]))

        symptom = data.get("symptom",[])
        for i,sy in enumerate(symptom):
            if symptom[i].endswith('...'):
                symptom[i] = symptom[i][:-3]
        all_entity["Symptom"].extend(symptom)
        if symptom:
            relationship.extend([("Disease", disease_name, "DISEASE_HAS_SYMPTOM", "Symptom",sy )for sy in symptom])

        cure_way = data.get("cure_way", [])
        if cure_way:
            for i,cure_w in enumerate(cure_way):
                if(isinstance(cure_way[i], list)):
                    cure_way[i] = cure_way[i][0] #glm处理数据集偶尔有格式错误
            cure_way = [s for s in cure_way if len(s) >= 2]
            all_entity["Treatment"].extend(cure_way)
            relationship.extend([("Disease", disease_name, "DISEASE_TREATED_BY", "Treatment", cure_w) for cure_w in cure_way])
            

        acompany_with = data.get("acompany", [])
        if acompany_with:
            relationship.extend([("Disease", disease_name, "DISEASE_COMPLICATION", "Disease", disease) for disease in acompany_with])

        drug_detail = data.get("drug_detail",[])
        for detail in drug_detail:
            lis = detail.split(',')
            if(len(lis)!=2):
                continue
            p,d = lis[0],lis[1]
            all_entity["Manufacturer"].append(d)
            all_entity["Drug"].append(p)
            relationship.append(('Manufacturer',d,"MANUFACTURES","Drug",p))
    for i in range(len(relationship)):
        if len(relationship[i])!=5:
            print(relationship[i])
    relationship = list(set(relationship))
    all_entity = {k:(list(set(v)) if k!="Disease" else v)for k,v in all_entity.items()}
    
    # 保存关系 放到data下
    with open("llm_medication/src/data/rel_aug_en.txt",'w',encoding='utf-8') as f:
        for rel in relationship:
            f.write(" ".join(rel))
            f.write('\n')

    if not os.path.exists('llm_medication/src/data/ent_aug_en'):
        os.mkdir('llm_medication/src/data/ent_aug_en')
    for k,v in all_entity.items():
        with open(f'llm_medication/src/data/ent_aug_en/{k}.txt','w',encoding='utf8') as f:
            if(k!='Disease'):
                for i,ent in enumerate(v):
                    f.write(ent+('\n' if i != len(v)-1 else ''))
            else:
                for i,ent in enumerate(v):
                    f.write(ent['name']+('\n' if i != len(v)-1 else ''))

    #将属性和实体导入到neo4j上,注:只有疾病有属性，特判
    for k in all_entity:
        if k!="Disease":
            import_entity(client,k,all_entity[k])
        else:
            
            import_disease_data(client,k,all_entity[k])
    create_all_relationship(client,relationship)

    

    

    