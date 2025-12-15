#!/usr/bin/env python
# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

import json
import sys

from autoconf.min_gpu_recommender import min_gpu_recommender

path_output = sys.argv[1]
model_name = sys.argv[2]
method = sys.argv[3]
gpu_model = sys.argv[4]
tokens_per_sample = int(sys.argv[5])
batch_size = int(sys.argv[6])
model_version = sys.argv[7]


output = {
    "workers": 0,
    "gpus": 0,
    "error": "",
}

configuration = {
    "model_name": model_name,
    "method": method,
    "gpu_model": gpu_model,
    "tokens_per_sample": tokens_per_sample,
    "batch_size": batch_size,
    "model_version": model_version,
}

try:
    measured_properties = min_gpu_recommender(**configuration)
except Exception as e:
    output["error"] = str(e)
else:
    output["workers"] = measured_properties.get("workers")
    output["gpus"] = measured_properties.get("gpus")

with open(path_output, "w") as f:
    json.dump(output, f)
