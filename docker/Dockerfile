FROM ubuntu:bionic

# Install base utilities
RUN apt-get update \
    && apt-get install -y -qq --no-install-recommends \
        build-essential ca-certificates wget git nano less

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
RUN /bin/bash /tmp/miniconda.sh -b -p ${CONDA_DIR}

# Put conda in path so we can use conda activate
ENV PATH=$PATH:$CONDA_DIR/bin

# Install NEURON using the scrit install_NEURON7.sh
# copy install script into tmmp folder and run it with bash
COPY install_NEURON7.sh /tmp/install_NEURON7.sh
RUN chmod +x /tmp/install_NEURON7.sh \
    && /bin/bash /tmp/install_NEURON7.sh
