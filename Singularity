Bootstrap:docker
From:pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

%files
    cli.sh /cli.sh
    requirements.txt /requirements.txt

%runscript
    exec /bin/bash /cli.sh "$@"

%post
    chmod u+x /cli.sh

    # Update
    apt update
    apt install -y wget build-essential git graphviz zip unzip curl vim libexpat1-dev

    # Install miniconda
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash Miniconda3-latest-Linux-x86_64.sh -b

    # Install RATS
    curl -L https://storage.googleapis.com/google-code-archive-downloads/v2/code.google.com/rough-auditing-tool-for-security/rats-2.4.tgz > rats-2.4.tgz
    tar -xzvf rats-2.4.tgz
    cd rats-2.4
    ./configure && make && make install

    # Install cppcheck
    apt install -y cppcheck

    # Install flawfinder
    pip install flawfinder

    # Install python dependencies here
    pip install -r /requirements.txt
    conda install -y pygraphviz
    python -c 'import nltk; nltk.download("punkt")'
