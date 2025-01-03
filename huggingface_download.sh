export HF_ENDPOINT=https://hf-mirror.com
source ~/.bashrc
pip install --upgrade huggingface-hub
# huggingface-cli download --resume-download t5-small --local-dir ~/workplace/trx --local-dir-use-symlinks False
# huggingface-cli download --resume-download gpt2 --local-dir ~/workplace/trx/gpt2 --local-dir-use-symlinks False
# huggingface-cli download --resume-download meta-llama/Llama-3.2-1B --local-dir ~/workplace/trx/Llama-3.2-1B --local-dir-use-symlinks False
