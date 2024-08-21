FROM ubuntu:22.04 AS venv-image

# Speed up the build, and avoid unnecessary writes to disk
ENV TZ=Europe/Paris
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV PIPENV_VENV_IN_PROJECT=true PIP_NO_CACHE_DIR=false PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PATH="/opt/venv/bin:$PATH"

RUN rm -f /var/cache/apt/archives/*.deb /var/cache/apt/archives/partial/*.deb /var/cache/apt/*.bin && \
    apt-get update && apt-get install -y python3-tk graphviz git python3-pip python3-venv && \
    rm -rf /var/lib/{apt,dpkg,cache,log} && \
    python3 -m venv /opt/venv

COPY requirements.txt .

# installing engine in separate call as gitlab runner is unable to correctly handle two extra urls in one single pip call
RUN --mount=type=cache,target=/root/.cache/pip --mount=type=secret,id=_env,dst=/etc/secrets/.env . /etc/secrets/.env \
    && pip install --upgrade setuptools==65.5.0 wheel \
    && pip install -r ./requirements.txt --cache-dir /root/.cache/pip

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 AS runtime-image

ENV TZ=Europe/Paris
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV APP_FOLDER=/app PATH="/opt/venv/bin:$PATH"

# Headless JRE needed for running Freerouting
RUN apt-get update && apt-get install -y \
    python3 python3-distutils python3-apt curl \
    lsb-release wget software-properties-common gnupg && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/local/bin/python


COPY --from=venv-image /opt/venv/. /opt/venv/

COPY . .
