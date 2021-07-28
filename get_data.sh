#!/bin/bash
# Get Google Drive files here
# To view drive file, go to the link:
# https://drive.google.com/file/d/<file_id>

if [[ -d storage/external ]]; then
    echo "storage exists, starting download"
else
    mkdir --parents storage/external
fi

cd storage/external

if [[ ! -f "bigvul2020.csv.gzip" ]]; then
    gdown https://drive.google.com/uc\?id\=1zu2-olVEO0aTu4utJiHTlCCpsX0NKP8U
else
    echo "Already downloaded bigvul data"
fi
