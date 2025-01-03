#!/bin/bash

# 安装 modelscope
pip install modelscope

# 使用 Python 代码下载模型
python -c "
from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('LLM-Research/Llama-3.2-1B', cache_dir='/home/trx/workplace/trx/Llama-3.2-1B',ignore_file_pattern='*.bin')
print(f'Model downloaded to: {model_dir}')
"