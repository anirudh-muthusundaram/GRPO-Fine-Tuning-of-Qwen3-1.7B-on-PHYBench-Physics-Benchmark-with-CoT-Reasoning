# Use a CUDA-enabled base image for GPU support
# FROM nvidia/cuda:11.0.3-base
#FROM nvidia/cuda:11.8.0-base-ubuntu22.04
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04


ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# Fix CUDA repository GPG key issue
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

# Install curl, gnupg2, and add NVIDIA Docker GPG key
RUN apt-get update --allow-releaseinfo-change && \
    apt-get install -y curl gnupg2 && \
    curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | \
    gpg --dearmor -o /etc/apt/trusted.gpg.d/nvidia-docker.gpg && \
    rm -rf /var/lib/apt/lists/*





# Install Python 3.8 explicitly, with distutils and build-essential
# Install Python 3.10
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-distutils curl git && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    make \
    libstdc++-12-dev \
    libgcc-12-dev \
    python3 \
    python3-pip \
    git \
    curl \
    gnupg2 \
    pkg-config \
    ca-certificates && \
    rm -rf /var/lib/apt/lists/*


# Set the CC environment variable so Triton can find the C compiler
ENV CC=/usr/bin/gcc

# Upgrade pip
RUN pip3 install --upgrade pip
# Install torch first
RUN pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121 && \
    pip install flash-attn deepspeed

WORKDIR /workspace

# Install Python packages and unsloth from the main branch
RUN pip3 install torch transformers peft jupyter

# Clone and patch Unsloth
# Install Unsloth and vLLM (no dependency issues)
# Inside your Dockerfile
RUN pip install torch transformers peft jupyter

# Fix & install unsloth manually
RUN git clone https://github.com/unslothai/unsloth.git /tmp/unsloth && \
    sed -i '/^license =/d' /tmp/unsloth/pyproject.toml && \
    sed -i '/^\[project\]/a license = {text = "Apache-2.0"}' /tmp/unsloth/pyproject.toml && \
    pip install /tmp/unsloth

# Install vllm and patched transformers
RUN pip install vllm && \
    pip install --no-deps git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3

# Expose Jupyter Notebook port
EXPOSE 8888

# Run Jupyter Notebook at container startup
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser"]