# -*- coding: utf-8 -*-
"""
@Time ： 2023/10/18 10:57
@Auth ： zhaliangyu
@File ：preprocess.py
@IDE ：PyCharm
"""

import json
from copy import deepcopy

def get_command_args_values(dic, output):
    for key, value in dic.items():
        if isinstance(value, dict):
            output = get_command_args_values(value, output)
        else:
            output += str(value) + " "
    return output

processed_data_path = "processed_train_data.json"
data_path = "dsl_train_post.json"
new_datas = []
with open(data_path, "r") as f:
    ori_datas = json.load(f)

for data in ori_datas:
    new_data = {}
    dsls = data["conversations"]
    new_data["sql"] = deepcopy(data["query"])
    new_data["dsl"] = str(deepcopy(data["conversations"]))
    new_data["dsl_guidance"] = ""
    for dsl in dsls:
        command_args = dsl["command_args"]
        output = ""
        dsl_type = dsl["command"]
        if dsl_type != "Bool":
            dsl_guidance = str(dsl["input"]) + " " + str(dsl["output"]) + \
                                   " " + str(dsl["command"]) + " " +  \
                           get_command_args_values(command_args, output)
        else:
            dsl_guidance = str(dsl["input"]) + " " + str(dsl["output"]) + \
                                   " " + str(dsl["command"]) + " " +  \
                           str(command_args)
        new_data["dsl_guidance"] += dsl_guidance
    new_datas.append(new_data)

with open(processed_data_path, "w") as f:
    json.dump(new_datas, f, ensure_ascii=False)
