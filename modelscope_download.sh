#!/bin/bash

# 安装 modelscope
pip install modelscope

# 使用 Python 代码下载模型
python -c "
from modelscope.hub.snapshot_download import snapshot_download
model_dir = snapshot_download('Wojtek/flan-t5-large', cache_dir='/home/trx/workplace/trx/t5-large',ignore_file_pattern='*.bin')
print(f'Model downloaded to: {model_dir}')
"