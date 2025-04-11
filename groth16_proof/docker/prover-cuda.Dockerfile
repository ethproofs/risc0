# Use nvidia's base image for Ubuntu
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04 AS cuda-base

# Install a bunch of stuff we might need (audit this list later?)
RUN apt-get update && apt-get install -y \
    build-essential \
    clang \
    git \
    cmake \
    curl \
    ca-certificates \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

# Install Go
ENV GOLANG_VERSION=1.23.0
RUN curl -L https://go.dev/dl/go${GOLANG_VERSION}.linux-amd64.tar.gz | tar -xz -C /usr/local
ENV PATH="/usr/local/go/bin:${PATH}"
RUN go version

# Copy the prover code we are going to build
COPY ./circom-compat /circom-compat
WORKDIR /circom-compat

# Update the mod file and download all the dependencies
RUN go mod tidy

# Compile icicle-gnark the slow way
#RUN go get github.com/ingonyama-zk/icicle-gnark/v3; \
#    cd $(go env GOMODCACHE)/github.com/ingonyama-zk/icicle-gnark/v3@v3.2.2/wrappers/golang; \
#    /bin/bash build.sh -curve=bn254;

# Or... cache it and reuse. Faster for development, but we probably shouldn't
# ship this - use the normal way above, instead.
RUN --mount=type=cache,target=/root/.cache/icicle-gnark \
    go get github.com/ingonyama-zk/icicle-gnark/v3; \
    ICICLE_DIR=$(go env GOMODCACHE)/github.com/ingonyama-zk/icicle-gnark/v3@v3.2.2; \
    if [ -d "/root/.cache/icicle-gnark/libs" ]; then \
        echo "Using cached icicle-gnark libraries"; \
        mkdir -p /usr/local/lib /usr/local/lib/backend/bn254/cuda /usr/local/lib/backend/cuda; \
        cp -r /root/.cache/icicle-gnark/libs/* /usr/local/lib/; \
    else \
        cd $ICICLE_DIR/wrappers/golang; \
        /bin/bash build.sh -curve=bn254; \
        mkdir -p /root/.cache/icicle-gnark/libs; \
        cp -r /usr/local/lib/libicicle_* /root/.cache/icicle-gnark/libs/; \
        mkdir -p /root/.cache/icicle-gnark/libs/backend/; \
        cp -r /usr/local/lib/backend/* /root/.cache/icicle-gnark/libs/backend/; \
    fi;

# Tell the linker where icicle put all of its libraries:
# /usr/local/lib/libicicle_curve_bn254.so
# /usr/local/lib/libicicle_field_bn254.so
# /usr/local/lib/libicicle_device.so
# /usr/local/lib/backend/bn254/cuda/libicicle_backend_cuda_curve_bn254.so
# /usr/local/lib/backend/bn254/cuda/libicicle_backend_cuda_field_bn254.so
# /usr/local/lib/backend/cuda/libicicle_backend_cuda_device.so
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CGO_LDFLAGS="-L/usr/local/lib"
ENV CGO_LDFLAGS="${CGO_LDFLAGS} -L/usr/local/lib/backend/bn254/cuda"
ENV CGO_LDFLAGS="${CGO_LDFLAGS} -L/usr/local/lib/backend/cuda"
RUN ldconfig
# Icicle will need to know at runtime where to find its libraries
ENV ICICLE_BACKEND_INSTALL_DIR=/usr/local/lib/backend

# Compile the prover
RUN go build -tags=icicle /circom-compat/cmd/prover



