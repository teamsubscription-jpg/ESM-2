# RunPod Serverless handler for ESM-2 (queue-based endpoints).
# For load-balancing endpoints, see server.py, which reuses predict() below.
#
# Example job input:
# {
#   "input": {
#     "sequences": ["MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"],
#     "include": ["mean"],          # any of "mean", "per_tok", "contacts"
#     "repr_layers": [-1]           # optional, defaults to the last layer
#   }
# }
# "sequences" may also be a list of {"id": ..., "sequence": ...} objects.

import os

import runpod
import torch

import esm

MODEL_NAME = os.environ.get("ESM_MODEL", "esm2_t33_650M_UR50D")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model, alphabet = esm.pretrained.load_model_and_alphabet(MODEL_NAME)
model = model.eval().to(DEVICE)
batch_converter = alphabet.get_batch_converter()


def predict(job_input):
    raw = job_input.get("sequences")
    if not raw:
        return {"error": "'sequences' must be a non-empty list"}

    data = [
        (s.get("id", str(i)), s["sequence"]) if isinstance(s, dict) else (str(i), s)
        for i, s in enumerate(raw)
    ]
    include = set(job_input.get("include", ["mean"]))
    repr_layers = [
        (l + model.num_layers + 1) % (model.num_layers + 1)
        for l in job_input.get("repr_layers", [-1])
    ]

    labels, strs, tokens = batch_converter(data)
    tokens = tokens.to(DEVICE)
    with torch.no_grad():
        out = model(tokens, repr_layers=repr_layers, return_contacts="contacts" in include)

    results = []
    for i, (label, seq) in enumerate(zip(labels, strs)):
        n = len(seq)
        item = {"id": label, "length": n}
        for layer in repr_layers:
            rep = out["representations"][layer][i, 1 : n + 1]
            if "mean" in include:
                item.setdefault("mean_representations", {})[layer] = rep.mean(0).tolist()
            if "per_tok" in include:
                item.setdefault("representations", {})[layer] = rep.tolist()
        if "contacts" in include:
            item["contacts"] = out["contacts"][i, :n, :n].tolist()
        results.append(item)

    return {"model": MODEL_NAME, "results": results}


def handler(job):
    return predict(job["input"])


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
