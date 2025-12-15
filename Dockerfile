# Copyright (c) IBM Corporation
# SPDX-License-Identifier: MIT

# Build the manager binary
FROM mirror.gcr.io/golang:1.25 AS builder
ARG TARGETOS
ARG TARGETARCH

WORKDIR /workspace
# Copy the Go Modules manifests
COPY go.mod go.mod
COPY go.sum go.sum
# cache deps before building and copying source so that we don't need to re-download as much
# and so that source changes don't invalidate our downloaded layer
RUN go mod download

# Copy the Go source (relies on .dockerignore to filter)
COPY . .

# Build
# the GOARCH has no default value to allow the binary to be built according to the host where the command
# was called. For example, if we call make docker-build in a local env which has the Apple Silicon M1 SO
# the docker BUILDPLATFORM arg will be linux/arm64 when for Apple x86 it will be linux/amd64. Therefore,
# by leaving it empty we can ensure that the container and binary shipped on it will have the same platform.
RUN CGO_ENABLED=0 GOOS=${TARGETOS:-linux} GOARCH=${TARGETARCH} go build -a -o manager cmd/main.go

# VV: Running the autoconf custom_experiment in the same container as the operator
# works best with python 3.12.7.
# If you plan to solely use the operator to request the recommendeations off of an 
# REST API endpoint then you can use a distroless image here as well as skip installing
# ado and its autoconf custom_experiment
FROM mirror.gcr.io/python:3.12.7-slim
WORKDIR /

RUN mkdir /workspace
COPY --from=builder /workspace/manager /workspace

WORKDIR /workspace

# VV: Use these lines to experiment with custom versionf of autoconf
# You'll have to git clone ado here first
# COPY ado /workspace/ado
# RUN pip install --no-cache-dir ado ado/plugins/custom_experiments/autoconf/

# VV: To run the autoconf custom_experiment code locally, use the wrapper_autoconf.py 
# script by providing the following CLI argument to manager:
# --path-wrapper-script=/workspace/wrapper_autoconf.py
COPY cmd/wrapper_autoconf.py /workspace/wrapper_autoconf.py
RUN pip install --no-cache-dir ado-autoconf

USER 65532:65532

ENTRYPOINT ["/workspace/manager"]
