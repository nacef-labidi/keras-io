FROM ubuntu:20.04 as venv-image

# Speed up the build, and avoid unnecessary writes to disk
ENV TZ=Europe/Paris
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PATH="/opt/venv/bin:$PATH"

RUN apt-get update && apt-get install -y libcairo2-dev python3-tk graphviz python3-pip python3-venv && \
    rm -rf /var/lib/{apt,dpkg,cache,log} && \
    python3 -m venv /opt/venv

COPY requirements.txt .

RUN pip install --upgrade --quiet pip setuptools && \
    pip install -r ./requirements.txt


FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04  as runtime-image

ENV TZ=Europe/Paris
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV APP_FOLDER=/app PATH="/opt/venv/bin:$PATH"
#ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.0/targets/x86_64-linux/lib:/usr/local/cuda-11.8/compat

RUN apt update && apt install -y python3 && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/local/bin/python

COPY --from=venv-image /opt/venv/. /opt/venv/

WORKDIR $APP_FOLDER

COPY src/ .
