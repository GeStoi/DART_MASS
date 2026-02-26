# ==============================================================================
# MASS (Muscle-Actuated Skeletal System) Docker Environment
# Base: Ubuntu 22.04 | DART 6.8.5 | Python 3.10 | PyTorch (CPU)
# ==============================================================================

FROM --platform=linux/amd64 ubuntu:22.04

# Prevent interactive prompts during apt-get
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# ==============================================================================
# Stage 1: System dependencies
# ==============================================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    cmake \
    pkg-config \
    git \
    ca-certificates \
    wget \
    curl \
    # DART required dependencies
    libeigen3-dev \
    libassimp-dev \
    libccd-dev \
    libfcl-dev \
    libboost-regex-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    # DART optional: Bullet collision detector (required by MASS)
    libbullet-dev \
    # DART optional: GLUT GUI (required by MASS render module)
    libxi-dev \
    libxmu-dev \
    freeglut3-dev \
    # DART optional: OpenSceneGraph
    libopenscenegraph-dev \
    # DART optional: URDF/SDF parsers
    libtinyxml-dev \
    libtinyxml2-dev \
    liburdfdom-dev \
    # DART optional: NLopt
    libnlopt-cxx-dev \
    # DART optional: ODE
    # DART optional: ODE
    libode-dev \
    # NOTE: Do NOT install liboctomap-dev — DART 6.8 is incompatible with octomap >= 1.9.0
    # MASS Python dependencies
    python3-dev \
    python3-pip \
    python3-numpy \
    python3-tk \
    # pybind11
    pybind11-dev \
    # OpenMP support
    libomp-dev \
    # OpenGL / X11 (for headless rendering & forwarding)
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    mesa-utils \
    xvfb \
    x11-utils \
    && rm -rf /var/lib/apt/lists/*

# ==============================================================================
# Stage 2: Build and install DART 6.8.5 from source
# ==============================================================================
WORKDIR /tmp

RUN git clone --depth 1 --branch v6.8.5 https://github.com/dartsim/dart.git && \
    cd dart && \
    mkdir build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DDART_BUILD_EXAMPLES=OFF \
        -DDART_BUILD_TUTORIALS=OFF \
        -DDART_BUILD_UNITTESTS=OFF \
        -DHAVE_OCTOMAP=OFF \
        -DCMAKE_CXX_STANDARD=14 \
        -DCMAKE_CXX_FLAGS="-w" \
    && make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd /tmp && rm -rf dart

# ==============================================================================
# Stage 3: Python dependencies (PyTorch CPU, numpy, matplotlib)
# ==============================================================================
RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu && \
    pip3 install --no-cache-dir \
    numpy \
    matplotlib \
    ipython

# ==============================================================================
# Stage 4: Copy and build MASS project
# ==============================================================================
WORKDIR /workspace/MASS
COPY MASS/ /workspace/MASS/

RUN mkdir -p build && cd build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CXX_STANDARD=14 \
        -DCMAKE_CXX_FLAGS="-w -std=gnu++14" \
        -DPYTHON_EXECUTABLE=$(which python3) \
    && make -j$(nproc)

# Create nn output directory
RUN mkdir -p /workspace/MASS/nn

# ==============================================================================
# Stage 5: Runtime setup
# ==============================================================================
WORKDIR /workspace/MASS

# Set up virtual display for headless rendering
ENV DISPLAY=:99
ENV PYTHONPATH=/workspace/MASS/python:$PYTHONPATH
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# Entrypoint script to start Xvfb and keep container running
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["bash"]
