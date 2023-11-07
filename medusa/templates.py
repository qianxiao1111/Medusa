# -*- coding: utf-8 -*-
"""
@Time ： 2023/11/3 10:13
@Auth ： zhaliangyu
@File ：templates.py
@IDE ：PyCharm
"""
from fastchat.conversation import Conversation, SeparatorStyle
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

def get_conversation_template(name: str):
    return register_templates[name]


wizard_dsl_conv = Conversation(
        name="wizardlm for dsl",
        system_message="数据库的信息如下所示：\n{db_info},\n schema'和'detail'表示数据库的内容; 'foreign_keys'表示数据库多表之间的连接关系. "
                       "根据数据库信息以及用户的输入生成符合json格式输出的指令. \n\n",
        roles=["USER", "ASSISTANT"],
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )

register_templates = {
    "wizardlmfordsl": wizard_dsl_conv
}

