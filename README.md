
# Autoconf Resource Requirements for AI Jobs on Kubernetes

This Kubernetes controller integrates with the [ADO **autoconf** custom experiment](https://github.com/IBM/ado/tree/main/plugins/custom_experiments/autoconf) to automatically set resource requirements for AI tuning jobs that use [fms-hf-tuning](https://github.com/foundation-model-stack/fms-hf-tuning).

---

## What the controller does

The controller inspects the command line of your AI jobs (either `PyTorchJob` objects or `PyTorchJob` objects wrapped inside an `AppWrapper`) to extract details such as:
- Model name
- Tuning method
- Effective batch size (i.e., `per_device_batch_size * NUM_GPUS`)
- Maximum sequence length

It combines these with the target GPU model to request recommendations from the `autoconf` experiment and then:
- **Patches** the resource requests/limits of `PyTorchJob` objects, or
- **Creates** a new `AppWrapper` (when direct patching of a nested `PyTorchJob` is not supported by `AppWrapper`).

We wish to use this controller to **enhance the execution of AI workloads on Kubernetes clusters** such that they use the right number of GPUs so as to avoid going out of GPU memory.
The design of the controller enables us to explore different algorithms for resource recommendation which we plan to explore in the future.

### Kueue collaboration (design work in progress)
We are working with the Kueue maintainers on a Kubernetes Enhancement Proposal (KEP) to improve how external Kubernetes controllers interact with jobs managed by Kueue (including AI workloads). The design discussion is tracked here: <https://github.com/kubernetes-sigs/kueue/issues/6915>.

Until that work lands, this controller demonstrates a way to interact with Kueue-managed jobs while operating within current Kueue capabilities.

---

## Examples

### Test locally (no Docker image)
You can run the controller as a local process while it manages one or more namespaces on your cluster.

**Assumptions for the example below**
- You use a Kueue-managed namespace called `tuning` to run `AppWrapper` workloads.
- These workloads:
  1. use the Kueue `LocalQueue` named `default-queue`,
  2. wrap a `PyTorchJob` that uses [fms-hf-tuning](https://github.com/foundation-model-stack/fms-hf-tuning),
  3. request one or more NVIDIA GPUs of the same model (e.g., `NVIDIA-A100-SXM4-80GB`),
  4. are subject to Kyverno policies requiring the `kueue.x-k8s.io/queue-name` label on `AppWrapper` objects.

**Steps**
1. Create and activate a Python virtual environment, then install the ADO autoconf client:
   ```bash
   python3 -m venv .venv && source .venv/bin/activate
   pip install ado-autoconf==1.5.0 ipython
   ```
2. Log in to your cluster (via `kubectl` or `oc`).
3. Build the controller locally:
   ```bash
   make
   ```
4. Start the controller with flags appropriate for the scenario:
   ```bash
   ./bin/manager \
     --done-label-key=kueue.x-k8s.io/queue-name \
     --done-label-value=default-queue \
     --namespaces "tuning" \
     --enable-appwrapper=true \
     --enable-pytorchjob=true \
     --unsuspend-patched-jobs=false \
     --default-gpu-model=NVIDIA-A100-SXM4-80GB \
     --path-wrapper-script=./cmd/wrapper_autoconf.py
   ```
5. Create an `AppWrapper` or `PyTorchJob` workload with the following labels:
   ```yaml
   # This setup both satisfies Kyverno (requires a queue-name) and
   # allows Kueue to temporarily ignore the job until the controller updates it.
   kueue.x-k8s.io/queue-name: fake
   autoconf-plugin-name: resource-requirements-appwrapper
   ```

Example `AppWrapper` and `PyTorchJob` manifests are available under [`examples`](./examples).

---

### Deploy to the cluster (high-level)
If you prefer to run the controller in-cluster (e.g., as a `Deployment`), the high-level process is:

1. **Build an image** for the controller.
2. **Create RBAC**: ServiceAccount, Role/ClusterRole, and bindings that permit reading/patching the resources you plan to manage (i.e. `AppWrapper` and/or `PyTorchJob`).
3. **Deploy a `Deployment`** for the controller, setting the desired command-line flags (see **Configuration** below). Enable leader election if you run multiple replicas.
4. **Optionally expose metrics/webhooks** via a `Service` if you enable those endpoints.
5. **Label workloads** so the controller can discover them (see `--watch-label-key` / `--watch-label-value`), then create your `AppWrapper`/`PyTorchJob` objects.
6. **Observe logs and job status** to confirm resources are being recommended and applied as expected.


---

## Important Configuration flags

Below are the controller’s command-line options:

### Core behavior
- `--default-autoconf-model-version string` — Default autoconf model version to use (default `3.1.0`).
- `--default-gpu-model string` — Default GPU model if not specified in the job.
- `--patch-cpu-request` — Set job CPU request/limit to `max(1, 2 * NUM_GPUS)` (default `true`).
- `--unsuspend-patched-jobs` — Unsuspend jobs after patching.
- `--path-wrapper-script string` — Path to the local Python wrapper for running models. **Mutually exclusive** with `--url-ado`. Exactly one of these must be set.
- `--url-ado string` — URL of the ADO REST API serving the models. **Mutually exclusive** with `--path-wrapper-script`. Exactly one of these must be set.

### Discovery & scope
- `--namespaces string` — Comma-separated list of namespaces to watch.
- `--watch-label-key string` — Limit monitoring to objects labeled `key=value` (default key `autoconf-plugin-name`).
- `--watch-label-value string` — Label value used with `--watch-label-key` (default `resource-requirements-appwrapper`).
- `--enable-appwrapper` — Watch and patch `AppWrapper` objects.
- `--enable-pytorchjob` — Watch and patch `PyTorchJob` objects.

### Completion labeling
- `--done-label-key string` — Label key inserted when patching is complete (default `autoconf-plugin-done`).
- `--done-label-value string` — Label value inserted when patching is complete (default `yes`).
- `--waiting-for-ado-request-id-label string` — Label used to mark jobs waiting for an ADO request ID (default `waiting-for-ado-request-id`).
- `--recommendation-annotation-key string` — Annotation key to store recommendation results in JSON format (default `ado-autoconf.ibm.com/recommendation`).

### Recommendation Annotations

The controller adds an annotation to each processed object containing the recommendation result in JSON format:

**Successful Recommendation:**
```json
{
  "recommendation": {
    "workers": 2,
    "gpus": 4
  }
}
```

**No Recommendation Available:**
```json
{
  "error": "No recommendation"
}
```

When the recommendation engine cannot generate recommendations (e.g., `can_recommend != 1` from the ado experiment), the controller:
1. Adds the error annotation to the object
2. Sets the done label to mark it as processed
3. Allows the object to proceed through standard Kubernetes workflows without modifications
4. Emits a warning event indicating no recommendation was available

This ensures objects are not stuck in a retry loop when recommendations cannot be generated.

### Logging
- `--zap-devel` — Development mode defaults (console encoder, debug log level, warn stack traces). Production mode defaults (JSON encoder, info log level, error stack traces). Default `true`.
- `--zap-encoder [json|console]` — Zap log encoding.
- `--zap-log-level value` — Log verbosity (`debug`, `info`, `error`, `panic`, or integer > 0 for custom levels).
- `--zap-stacktrace-level [info|error|panic]` — Level at and above which stack traces are captured.
- `--zap-time-encoding [epoch|millis|nano|iso8601|rfc3339|rfc3339nano]` — Time encoding (default `epoch`).

