FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /usr/src/app
RUN apt-get update
RUN apt-get -y upgrade
RUN apt-get install -y curl
RUN curl -L https://huggingface.co/MaziyarPanahi/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf --output model.gguf
RUN curl -L https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.f32.gguf --output embed.gguf
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.12 python3.12-dev python3.12-venv python3-pip git
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1
SHELL ["/bin/bash", "-c"]
COPY . .
RUN python3 -m venv env
RUN source env/bin/activate
ENV CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=all-major"
RUN python3 -m ensurepip --upgrade
RUN pip3 install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "ui.py", "--server.headless", "true"]
