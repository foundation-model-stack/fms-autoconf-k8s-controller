#!/usr/bin/env bash
              #--default-gpu-model="NVIDIA-A100-80GB-PCIe" \

./bin/manager --done-label-key=kueue.x-k8s.io/queue-name \
              --done-label-value=default-queue  \
              --namespaces "tuning" \
              --enable-appwrapper=true \
              --enable-pytorchjob=false \
              --unsuspend-derived-jobs=false \
              --url-ado https://ado-api-discovery-dev.apps.morrigan.accelerated-discovery.res.ibm.com