FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0
LABEL authors="greg"
LABEL description="Base ML l4t jetpack forr jetson"

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3-pip \
        libhdf5-dev \
        liblapack-dev \
        libblas-dev \
        gfortran \
        libfreetype6-dev \
        pkg-config && \
    rm -rf /var/lib/apt/lists/*


COPY ./requirements.txt /app/requirements.txt
RUN python3 -m pip install --upgrade pip && \
    pip3 install -r /app/requirements.txt

# CuPy for CUDA-accelerated preprocessing (cupyx.scipy.ndimage).
RUN pip3 install cupy-cuda12x

ENV MPLBACKEND=Agg

CMD ["bash"]
