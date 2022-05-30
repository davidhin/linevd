#!/bin/bash

function checkit()
{
    dir="storage/processed/bigvul/old/before"
    cut -c 1-12 "$dir/$1" | grep overlays &>/dev/null
    if [ $? -ne 0 ]
    then
        echo $1 >> missing_overlay2.txt
    fi
    echo $1
}
export -f checkit

cat to_check_old.txt | parallel checkit {} | tqdm --total "$(cat to_check_old.txt | wc -l)" >> /dev/null
