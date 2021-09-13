#!/bin/bash
# Get Google Drive files here
# To view drive file, go to the link:
# https://drive.google.com/file/d/<file_id>

if [[ -n "${SINGSTORAGE}" ]]; then
    cd $SINGSTORAGE
fi

if [[ -d storage/external ]]; then
    echo "storage exists, starting download"
else
    mkdir --parents storage/external
fi

cd storage/external

if [[ ! -f "MSR_data_cleaned.csv" ]]; then
    gdown https://drive.google.com/uc\?id\=1-0VhnHBp9IGh90s2wCNjeCMuy70HPl8X
    unzip MSR_data_cleaned.zip
    rm MSR_data_cleaned.zip
else
    echo "Already downloaded bigvul data"
fi

if [[ ! -f "MSR_data_cleaned_SAMPLE.csv" ]]; then
    gdown https://drive.google.com/uc\?id\=1SmUfm1ibLjSg6QAA5-jK7u0RL-noLGIF
else
    echo "Already downloaded bigvul sample data"
fi

if [[ ! -d joern-cli ]]; then
    gdown https://drive.google.com/uc\?id\=1LwZcKkHqFBWUBOLfoC8mB6oFesV5iGbX
    unzip joern-cli.zip
    rm joern-cli.zip
else
    echo "Already downloaded Joern"
fi
