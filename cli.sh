#!/bin/bash

echo "Welcome to the CLI! \
Please use '-p initialise' if it's your first time. \
This CLI must be run in the directory folder of the cloned repository."
usage() {
    echo "Usage: $0 \
[-h help] \
[-t run tests] \
[-p run program <initialise|path_to_file>] \
    [-a arguments]" 1>&2
    exit 1
}

while getopts ":hp:a:t" opt; do
    case ${opt} in
    h)
        usage
        ;;
    p)
        p=${OPTARG}
        ;;
    a)
        a+=("${OPTARG}")
        ;;
    t)
        pytest tests/
        ;;
    \?)
        echo "Invalid option"
        usage
        ;;
    esac
done
shift $((OPTIND - 1))

# Download data and install main code
if [[ "initialise" == "${p}" ]]; then
    pip install -e .
    bash get_data.sh
    exit 0
fi

# Run Python Program
if [[ -z "${p}" ]]; then
    usage
else
    python3 -u "${p}" "${a[@]}"
fi
