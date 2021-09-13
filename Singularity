Bootstrap:docker
From:pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

%files
    cli.sh /cli.sh
    requirements.txt /requirements.txt

%environment
    export SINGULARITY=true
    PATH=$PATH:/GloVe/build

%runscript
    exec /bin/bash /cli.sh "$@"

%post
    chmod u+x /cli.sh

    # Update
    apt update
    apt install -y wget build-essential git graphviz zip unzip curl vim libexpat1-dev cmake

    # Install Glove
    cd /
    git clone https://github.com/stanfordnlp/GloVe.git
    cd GloVe
    make

    # Install cppcheck
    cd /
    curl -L https://github.com/danmar/cppcheck/archive/refs/tags/2.5.tar.gz > cppcheck2.5.tar.gz    
    mkdir cppcheck
    mv cppcheck2.5.tar.gz cppcheck
    cd cppcheck
    tar -xzvf cppcheck2.5.tar.gz
    cd cppcheck-2.5
    mkdir build
    cd build
    cmake ..
    cmake --build .
    make install

    # Install Joern
    cd /
    apt install -y openjdk-8-jdk git curl gnupg bash unzip sudo wget 
    wget https://github.com/ShiftLeftSecurity/joern/releases/latest/download/joern-install.sh
    chmod +x ./joern-install.sh
    printf 'Y\n/bin/joern\ny\n/usr/local/bin\n\n' | sudo ./joern-install.sh --interactive

    # Install miniconda
    cd /
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b

    # Install RATS
    cd /
    curl -L https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/rough-auditing-tool-for-security/rats-2.4.tgz > rats-2.4.tgz
    tar -xzvf rats-2.4.tgz
    cd rats-2.4
    ./configure && make && make install

    # Install flawfinder
    pip install flawfinder

    # Install python dependencies here
    cat /requirements.txt | xargs -n 1 pip install
    pip install dgl-cu102 -f https://data.dgl.ai/wheels/repo.html
    conda install -y pygraphviz
    pip install nltk
    python -c 'import nltk; nltk.download("punkt")'
