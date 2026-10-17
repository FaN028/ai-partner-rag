# -*- coding: utf-8 -*-
# 下载中文 embedding 模型 bge-small-zh-v1.5。
#
# 走 ModelScope（阿里）镜像而非 HuggingFace：国内网络直连 HuggingFace 不可达。
# 产物目录 models/bge-small-zh-v1.5/ 与 config.MODEL_PATH 一致，rag.py 直接加载。
#
# 前置：pip install modelscope
# 执行：python download_model.py

from modelscope import snapshot_download

model_dir = snapshot_download(
    "AI-ModelScope/bge-small-zh-v1.5",       # 模型在 ModelScope 上的名字
    local_dir="./models/bge-small-zh-v1.5",  # 下载到项目里的哪个文件夹
)
print("下载完成，模型在：", model_dir)
