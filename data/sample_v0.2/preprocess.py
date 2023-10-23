import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig
import os, copy, json
from typing import List
import pandas as pd
from datasets import load_dataset, Dataset

# prompt_table = "表格数据信息以字典的形式如下所示:\n{data_describe}\n其中的key为表名,value为单张表包含的数据信息.根据表格数据信息以及用户的输入生成符合json格式输出的指令.\n\n"
# prompt_table_dsl = "数据库的信息如下所示:\n{data_describe}\n'db_info'中的key为表名; 'numeric_info'表示连续列的信息; 'categorical_info'表示离散列的信息,'date_cols_info'表示时间列的信息. 'foreign_keys'表示数据库多表之间的连接关系. 根据数据库信息以及用户的输入生成符合json格式输出的指令.\n\n"
prompt_table_sql = "数据库的信息如下所示:\n{data_describe}\n'schema'和'detail'表示数据库的内容; 'foreign_keys'表示数据库多表之间的连接关系. 根据数据库信息以及用户的输入生成符合json格式输出的指令.\n\n"
prompt_cot = "The table data information is in the form of a dictionary as follows:\n{data_describe}\nwhere key is the table name and value is the data information contained in a single table. Based on table data information and user input, please solve the input task step by step.\n\n"


baichuan_prompt = """
用户在与一个客观的助手对话。助手会尊重找到的材料，给出全面专业的解释，但不会过度演绎。同时回答中不会暴露引用的材料：\n
```
引用材料
```
用户："""


query = """
original input:
帮我把从3210到3525的行数据提取出来
cot input:
step1:帮我把从3210到3525的行数据提取出来
"""


def convert_data_llama_efficient(input_data_path, output_data_path, cot_input=True,cot_output=False):
    # 数据转换成llama-efficient格式
    with open(input_data_path) as f:
        datas = json.load(f)
    data_news = []
    for data in datas:
        data_new = {}
        data_describe = data["table_infos"]

        data_describe = json.dumps(data_describe, ensure_ascii=False)
        system = prompt_table_sql.format(data_describe=data_describe)

        data_new["system"] = system

        convs = data["conversations"]
        # query,response
        if convs[-2]["from"] == "human":
            query = convs[-2]["value"]
            if cot_input:
                cot = data["chain_of_thought"]
                cot_query = ""
                for step,content in cot.items():
                    cot_query = cot_query+step+":"+content+"\n"
                query = "original input:"+"\n"+query+"\n"+"cot input:"+"\n"+cot_query
        else:
            raise ValueError("-2不是human")
        if convs[-1]["from"] == "gpt":
            #output放在最后，明天跟command相关
            values = convs[-1]["value"]
            for value in values:
                output = value["output"]
                del value["output"]
                value["output"]=output
            response = json.dumps(values, ensure_ascii=False)
        else:
            raise ValueError("-1不是gpt")

        # history
        history = []

        data_new["prompt"] = query
        data_new["query"] = ""
        data_new["response"] = response
        data_new["history"] = history
        data_news.append(data_new)

    with open(output_data_path, "w") as f:
        json.dump(data_news, f, ensure_ascii=False)


def convert_data_cot(input_data_path, output_data_path, cot_input=True,cot_output=False):
    with open(input_data_path) as f:
        datas = json.load(f)
    data_news = []
    for data in datas:
        data_new = {}
        # data_new["id"] = data["id"]

        # system
        head = data["head"]
        df_info = data["df_info"]
        data_describe = {}
        for key, value in head.items():
            value.update(df_info[key])
            data_describe[key] = value
        for name,db in data_describe.items():
            del db["target"]
            del db["task_type"]

        data_describe = json.dumps(data_describe, ensure_ascii=False)
        system = prompt_cot.format(data_describe=data_describe)

        data_new["system"] = system

        convs = data["conversations"]
        # query,response
        if convs[-2]["from"] == "human":
            query = convs[-2]["value"]

        else:
            raise ValueError("-2不是human")

        cot = data["chain_of_thought"]
        response = json.dumps(cot, ensure_ascii=False)

        # history
        history = []

        data_new["prompt"] = query
        data_new["query"] = ""
        data_new["response"] = response
        data_new["history"] = history
        data_news.append(data_new)

    with open(output_data_path, "w") as f:
        json.dump(data_news, f, ensure_ascii=False)

if __name__ == "__main__":
    convert_data_llama_efficient("train_dsl.json","train_dsl_convert.json")
    # convert_data_cot("samplev0.2_val.json","samplev0.2_val_cot.json")
    