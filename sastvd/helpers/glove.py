"""Python wrapper to call StanfordNLP Glove."""

from pathlib import Path

import sastvd as svd


def glove(
    CORPUS,
    VOCAB_FILE="vocab.txt",
    COOCCURRENCE_FILE="cooccurrence.bin",
    COOCCURRENCE_SHUF_FILE="cooccurrence.shuf.bin",
    SAVE_FILE="vectors",
    VERBOSE=2,
    MEMORY=4.0,
    VOCAB_MIN_COUNT=5,
    VECTOR_SIZE=200,
    MAX_ITER=15,
    WINDOW_SIZE=15,
    BINARY=2,
    NUM_THREADS=8,
    X_MAX=10,
):
    """Run StanfordNLP Glove on a corpus. Mainly copied from demo.sh in their repo."""
    savedir = Path(CORPUS).parent
    VOCAB_FILE = savedir / VOCAB_FILE
    COOCCURRENCE_FILE = savedir / COOCCURRENCE_FILE
    COOCCURRENCE_SHUF_FILE = savedir / COOCCURRENCE_SHUF_FILE
    SAVE_FILE = savedir / SAVE_FILE

    cmd1 = f"vocab_count \
        -min-count {VOCAB_MIN_COUNT} \
        -verbose {VERBOSE} \
        < {CORPUS} > {VOCAB_FILE}"

    cmd2 = f"cooccur \
        -memory {MEMORY} \
        -vocab-file {VOCAB_FILE} \
        -verbose {VERBOSE} \
        -window-size {WINDOW_SIZE} \
        < {CORPUS} > {COOCCURRENCE_FILE}"

    cmd3 = f"shuffle \
        -memory {MEMORY} \
        -verbose {VERBOSE} \
        < {COOCCURRENCE_FILE} > {COOCCURRENCE_SHUF_FILE}"

    cmd4 = f"glove \
        -save-file {SAVE_FILE} \
        -threads {NUM_THREADS} \
        -input-file {COOCCURRENCE_SHUF_FILE} \
        -x-max {X_MAX} -iter {MAX_ITER} \
        -vector-size {VECTOR_SIZE} \
        -binary {BINARY} \
        -vocab-file {VOCAB_FILE} \
        -verbose {VERBOSE}"

    svd.watch_subprocess_cmd(cmd1)
    svd.watch_subprocess_cmd(cmd2)
    svd.watch_subprocess_cmd(cmd3)
    svd.watch_subprocess_cmd(cmd4)
