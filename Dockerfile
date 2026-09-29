# RunPod Serverless image for ESM-2 protein embeddings.
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_HOME=/models/torch

# Which ESM-2 checkpoint to serve, e.g. esm2_t6_8M_UR50D, esm2_t12_35M_UR50D,
# esm2_t30_150M_UR50D, esm2_t33_650M_UR50D, esm2_t36_3B_UR50D.
ARG ESM_MODEL=esm2_t33_650M_UR50D
ENV ESM_MODEL=${ESM_MODEL}

WORKDIR /app

RUN pip install runpod

COPY . /app
RUN pip install .

# Download the weights at build time so workers don't fetch them on every cold start.
RUN python -c "import esm, os; esm.pretrained.load_model_and_alphabet(os.environ['ESM_MODEL'])"

CMD ["python", "-u", "runpod_handler.py"]
