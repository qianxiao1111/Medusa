# -*- coding: utf-8 -*-
"""
@Time ： 2023/10/26 11:02
@Auth ： zhaliangyu
@File ：inference_speed_compare.py
@IDE ：PyCharm
"""

#%%

import torch
import os
import transformers
from medusa.model.medusa_model import MedusaModel
import time
from contextlib import contextmanager
import numpy as np
from medusa.model.utils import *
from medusa.model.kv_cache import *
from medusa.model.medusa_choices import mc_sim_7b_63

print("finish import")

#%%
# define time calculate
@contextmanager
def timed(wall_times, key):
    start = time.time()
    torch.cuda.synchronize()
    yield
    torch.cuda.synchronize()
    end = time.time()
    elapsed_time = end - start
    wall_times[key].append(elapsed_time)

def medusa_forward(input_ids, model, tokenizer, medusa_choices, temperature, posterior_threshold, posterior_alpha, max_steps = 512):
    wall_times = {'medusa': [], 'tree': [], 'posterior': [], 'update': [], 'init': []}

    with timed(wall_times, 'init'):
        if hasattr(model, "medusa_choices") and model.medusa_choices == medusa_choices:
            # Load the cached medusa buffer
            medusa_buffers = model.medusa_buffers
        else:
            # Initialize the medusa buffer
            medusa_buffers = generate_medusa_buffers(
                medusa_choices, device=model.base_model.device
            )
        model.medusa_buffers = medusa_buffers
        model.medusa_choices = medusa_choices

        # Initialize the past key and value states
        if hasattr(model, "past_key_values"):
            past_key_values = model.past_key_values
            past_key_values_data = model.past_key_values_data
            current_length_data = model.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(model.base_model)
            model.past_key_values = past_key_values
            model.past_key_values_data = past_key_values_data
            model.current_length_data = current_length_data

        input_len = len(input_ids)
        reset_medusa_mode(model)
        medusa_logits, logits = initialize_medusa(
                input_ids, model, medusa_buffers["medusa_attn_mask"], past_key_values
        )
    new_token = 0

    for idx in range(max_steps):
        with timed(wall_times, 'medusa'):
            candidates, tree_candidates = generate_candidates(
                    medusa_logits,
                    logits,
                    medusa_buffers["tree_indices"],
                    medusa_buffers["retrieve_indices"],
                )

        with timed(wall_times, 'tree'):
            medusa_logits, logits, outputs = tree_decoding(
                    model,
                    tree_candidates,
                    past_key_values,
                    medusa_buffers["medusa_position_ids"],
                    input_ids,
                    medusa_buffers["retrieve_indices"],
                )

        with timed(wall_times, 'posterior'):
            best_candidate, accept_length = evaluate_posterior(
                    logits, candidates, temperature, posterior_threshold, posterior_alpha
                )

        with timed(wall_times, 'update'):
            input_ids, logits, medusa_logits, new_token = update_inference_inputs(
                    input_ids,
                    candidates,
                    best_candidate,
                    accept_length,
                    medusa_buffers["retrieve_indices"],
                    outputs,
                    logits,
                    medusa_logits,
                    new_token,
                    past_key_values_data,
                    current_length_data,
                )

        if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
            break

    return input_ids, new_token, idx, wall_times

#%%
# load medusa model and set hyper-params
from transformers import BitsAndBytesConfig
base_model_path = "/home/qyhuang/weights/dsl_weights/wizardlm-10-16-dsl/checkpoint-100"
medusa_path = "/home/qyhuang/weights/wizardlm13b_medusa"
cache_dir = ""
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "</s>"

config = transformers.AutoConfig.from_pretrained(
    base_model_path,
    cache_dir=cache_dir
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)
#
# base_model = transformers.AutoModelForCausalLM.from_pretrained(
#     base_model_path,
#     config=config,
#     cache_dir=cache_dir,
#     low_cpu_mem_usage=True,
#     torch_dtype=torch.bfloat16,
#     quantization_config=quantization_config,
#     load_in_4bit=True,
# )
# print("finish loading base model")
#%%
model = MedusaModel.from_pretrained(
    base_model=base_model_path,
    medusa_head_name_or_path=medusa_path
)

tokenizer = transformers.AutoTokenizer.from_pretrained(
    base_model_path,
    cache_dir=cache_dir,
    model_max_length=4096,
    padding_side="right",
    use_fast=False,
)

# tokenizer = model.get_tokenizer()
tokenizer.add_special_tokens(
    {
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
        "pad_token": DEFAULT_PAD_TOKEN
    }
)
temperature = 0.
posterior_threshold = 0.09
posterior_alpha = 0.3

#%%

PROMPT = "数据库的信息如下所示：{db_info},\n schema'和'detail'表示数据库的内容; 'foreign_keys'表示数据库多表之间的连接关系. " \
         "根据数据库信息以及用户的输入生成符合json格式输出的指令. \n\nUSER: {user_query}\nASSISTANT:"

db_info = """{"db_name": "Twenty-Five_Cents", "db_info": {"Twenty-Five_Cents": {"numeric_info": {"Year": [2006, 2007, 2007], "Issue price": [24.95, 24.95, 24.95]}, "categorical_info": {"Theme": ["Calgary Flames", "Edmonton Oilers", "Montreal Canadiens", "Ottawa Senators", "Toronto Maple Leafs", "Vancouver Canucks"], "Artist": ["N/A"], "Mintage": ["1264", "1634", "2213", "2952", "3527", "832", "N/A"]}, "date_cols_info": {}}}, "foreign_keys": []}"""
db_info = db_info.replace("{","{{").replace("}", "}}")
user_query = "不同主题的平均发行价格是多少？"
PROMPT = PROMPT.format(db_info=db_info,
                       user_query=user_query)
medusa_choices = mc_sim_7b_63

with torch.inference_mode():
    input_ids = tokenizer([PROMPT]).input_ids
    output_ids, new_token, idx, wall_time = medusa_forward(
                    torch.as_tensor(input_ids).cuda(),
                    model,
                    tokenizer,
                    medusa_choices,
                    temperature,
                    posterior_threshold,
                    posterior_alpha,
                )
    print("Output length:", output_ids.size(-1))
    print("Compression ratio:", new_token / idx)

output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
print(output)