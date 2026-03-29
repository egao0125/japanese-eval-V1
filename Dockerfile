FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3-pip git \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app
COPY pyproject.toml .
COPY src/ src/
COPY tasks/ tasks/
COPY corpora/ corpora/

RUN pip install --no-cache-dir -e ".[gpu,research,lenient]"

ENTRYPOINT ["jaeval"]
CMD ["--help"]
