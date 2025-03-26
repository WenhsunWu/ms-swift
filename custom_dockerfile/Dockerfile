FROM modelscope-registry.cn-hangzhou.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.1.0-py310-torch2.3.1-tf2.16.1-1.23.1
RUN pip install --no-cache-dir math_verify && \
    pip install --no-cache-dir swanlab && \
    pip install git+https://github.com/huggingface/trl.git && \
    pip install --no-cache-dir vllm==0.7.3
