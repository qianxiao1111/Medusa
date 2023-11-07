# -*- coding: utf-8 -*-
"""
@Time ： 2023/11/6 18:52
@Auth ： zhaliangyu
@File ：inference_acc_compare_1.py
@IDE ：PyCharm
"""
# -*- coding: utf-8 -*-
"""
@Time ： 2023/11/3 10:11
@Auth ： zhaliangyu
@File ：inference_acc_compare.py
@IDE ：PyCharm
"""
import copy

import transformers, os
import shutil
from medusa.model.medusa_model import MedusaModel
import time
from contextlib import contextmanager
import numpy as np
from medusa.model.utils import *
from medusa.model.kv_cache import *
from medusa.model.medusa_choices import mc_sim_7b_63



# define test function
import pandas as pd
import copy
from numbers import Number
from tqdm import tqdm
from medusa.templates import get_conversation_template
from transformers import GenerationConfig, LogitsProcessorList, \
    InfNanRemoveLogitsProcessor
import json
from fastchat.serve.cli import SimpleChatIO, RichChatIO, ProgrammaticChatIO
print("finish import!!")

def get_logits_processor() -> LogitsProcessorList:
    logits_processor = LogitsProcessorList()
    logits_processor.append(InfNanRemoveLogitsProcessor())
    return logits_processor


def compare_dicts(dict1, dict2):
    # 检查字典的键是否相等
    if set(dict1.keys()) != set(dict2.keys()):
        return False

    # 递归地比较字典的值
    for key in dict1.keys():
        value1 = dict1[key]
        value2 = dict2[key]

        # 如果值是字典，递归调用函数
        if isinstance(value1, dict) and isinstance(value2, dict):
            if not compare_dicts(value1, value2):
                return False

        # 如果list
        elif isinstance(value1, list) and isinstance(value2, list):
            if len(value1) != len(value2):
                return False
            if len(value1):
                # list嵌套dict
                if isinstance(value1[0], dict) and isinstance(value2[0], dict):
                    value1 = sorted(value1, key=lambda x: str(x))
                    value2 = sorted(value2, key=lambda x: str(x))
                    for i in range(len(value1)):
                        if not compare_dicts(value1[i], value2[i]):
                            return False

                # list嵌套list
                elif isinstance(value1[0], list) and isinstance(value2[0],
                                                                list):
                    value1 = sorted(value1, key=lambda x: str(x))
                    value2 = sorted(value2, key=lambda x: str(x))
                    for i in range(len(value1)):
                        if set(value1[i]) != set(value2[i]):
                            return False
                else:
                    # list里面是str或number
                    if set(value1) != set(value2):
                        return False

        # 字符串、数字、bool
        elif value1 != value2:
            return False

        else:
            if isinstance(value1, Number) and isinstance(value2, Number):
                value1 = float(value1)
                value2 = float(value2)
            if type(value1) != type(value2):
                return False

    # 如果所有键和对应的值都相等，返回True
    return True


def chat_loop(model, data_dir, save_dir, tokenizer, template_name, device,
              model_type="original"):
    rst_all = []

    chatio = SimpleChatIO(True)
    # for filename in os.listdir(data_dir):
    #     print(filename)
    em_count = 0
    rst = []
    with open(os.path.join(data_dir, data_dir), "rb") as f1:
        json_contexts = json.load(f1)
    pbar = tqdm(json_contexts)
    conv = get_conversation_template(template_name)
    for json_context in pbar:
        rst_line = {}
        db_infos = json_context["table_infos"]
        db_infos = str(db_infos).replace("{", "{{").replace("}", "}}")
        target = json_context["conversations"][1]["value"]
        query = json_context["conversations"][0]["value"]
        chain_of_thought = json_context["chain_of_thought"]
        cot = ""
        for key, value in chain_of_thought.items():
            temp_step = key + ":" + value + "\n"
            cot += temp_step
        conv_copy = copy.deepcopy(conv)
        inp = "original input: \n" + query + "\n" + \
              "cot_input: \n" + cot

        conv_copy.system_message = conv_copy.system_message.format(
            db_info=db_infos
        )
        conv_copy.append_message(conv_copy.roles[0], inp)
        conv_copy.append_message(conv_copy.roles[1], None)
        chatio.prompt_for_output(conv_copy.roles[1])
        prompt = conv_copy.get_prompt()
        input_ids = tokenizer.encode(
            prompt, return_tensors="pt"
        ).to(device)
        if model_type == "original":
            gen_kwargs = {}
            gen_kwargs["do_sample"] = True
            gen_kwargs["max_new_tokens"] = 512
            gen_kwargs["temperature"] = 0.1
            gen_kwargs["eos_token_id"] = tokenizer.eos_token_id
            gen_kwargs["pad_token_id"] = tokenizer.pad_token_id
            g_configs = dict(
                input_ids=input_ids,
                generation_config=GenerationConfig(**gen_kwargs),
                logits_processor=get_logits_processor()
            )
            output_ids = model.generate(**g_configs)
            input_len = input_ids.shape[1]
            output_ = output_ids.tolist()[0][input_len:]
            outputs = tokenizer.decode(output_, skip_special_tokens=True)
        elif model_type == "medusa":
            gen_kwargs = {}
            gen_kwargs["max_steps"] = 512
            gen_kwargs["temperature"] = 0.1
            gen_kwargs["posterior_threshold"] = 0.09
            gen_kwargs["posterior_alpha"] = 0.3

            outputs_ = model.medusa_generate(
                input_ids,
                medusa_choices=mc_sim_7b_63,
                **gen_kwargs
            )
            for output in outputs_:
                outputs = output["text"]
        else:
            raise ValueError()

        print("output text:", outputs)
        try:
            outputs = json.loads(outputs)
            if len(target) != len(outputs):
                rst_line["match"] = False
            else:
                rst_line["match"] = True
                for i, t in enumerate(target):
                    output = outputs[i]
                    ret = compare_dicts(t, output)
                    if ret == False:
                        rst_line["match"] = False
            if rst_line["match"] == True:
                em_count += 1

            rst_line["input"] = inp
            rst_line["target"] = target
            rst_line["pred"] = outputs
            rst.append(rst_line)

        # 生成的格式都不对
        except:
            rst.append(
                {
                    "output data error": [
                        inp,
                        json.dumps(target, ensure_ascii=False),
                        outputs,
                    ]
                }
            )
        pbar.update(int(1 / len(json_contexts)))

    rst_all.append(
        {
            "DataCategory": save_dir.split(".")[0],
            "DataNumbers": len(json_contexts),
            "EM_acc": round(em_count / len(json_contexts), 4),
        }
    )
    print(rst_all)
    with open(
            os.path.join(save_dir, data_dir.split(".")[0] + "_ret.json"),
            "w", encoding="utf-8") as f2:

        try:
            # 保存模型推测结果
            f2.write(json.dumps(rst, ensure_ascii=False, indent=4))
        except Exception as e:
            print(e)

    with open(os.path.join(save_dir, "ret_all.json"), "w",
              encoding="utf-8") as f3:
        f3.write(json.dumps(rst_all, ensure_ascii=False, indent=4))


def main(eval_data, save_dir, model, device, model_type):
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.mkdir(save_dir)
    chat_loop(model, eval_data, save_dir, tokenizer,
              template_name="wizardlmfordsl", device=device,
              model_type=model_type)


if __name__ == "__main__":
    # 测试原始模型
    # load model and tokenizer
    from transformers import BitsAndBytesConfig

    base_model_path = "/home/qyhuang/weights/dsl_weights/wizardlm-10-16-dsl/checkpoint-100"
    medusa_model_path = "/home/qyhuang/zhaliangyu/Medusa/medusa_weights/wizardlm13b_medusa/checkpoint-5600"
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

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        base_model_path,
        cache_dir=cache_dir,
        max_new_tokens=512,
        padding_side="right",
        use_fast=False,
    )

    dataset_path = "/home/qyhuang/zhaliangyu/Medusa/data/sample_v0.2/val_dsl.json"
    eval_result_dir = "/home/qyhuang/zhaliangyu/Medusa/data/eval_results/original_results"

    device_map = {"": int(os.environ.get("LOCAL_RANK", "1"))}
    # os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    medusa_model = MedusaModel.from_pretrained(
        base_model=base_model_path,
        from_check_point=True,
        medusa_head_name_or_path=medusa_model_path,
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        medusa_num_heads=3
    )
    medusa_model.cuda()

    main(eval_data=dataset_path, save_dir=eval_result_dir, model=medusa_model,
         device=medusa_model.base_model.device, model_type="medusa")


