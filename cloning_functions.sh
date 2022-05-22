function clo() {
    git clone $1 `echo "$1" | sed "s@://@__@g" | sed "s@/@__@g"`
}

function asc() {
    clo $1 &> output_$(basename $1).txt & 
}

cat ../../codeLinksToDownload.txt | grep cgit.kde.org > ../../codeLinks_kde.txt