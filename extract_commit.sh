#!/bin/bash
# https://stackoverflow.com/a/64086857/8999671

# php php-src c351b47ce85a3a147cfa801fa9f0149ab4160834

git_user="$1"
git_repo="$2"
commit_or_branch="$3"
dir_or_file="path/to/dir-or-file"

archive_url="https://github.com/${git_user}/${git_repo}/archive/${commit_or_branch}.tar.gz"

wget -O - ${archive_url} | tar xz