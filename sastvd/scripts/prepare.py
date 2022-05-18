import sastvd.helpers.datasets as svdd
import sastvd.ivdetect.evaluate as ivde
import sys


def bigvul():
    """Run preperation scripts for BigVul dataset."""
    sample = True if "sample" in sys.argv else False
    print("get df")
    svdd.bigvul(sample=sample)
    print("get dep add lines")
    ivde.get_dep_add_lines_bigvul(sample=sample)
    if "glove" in sys.argv:
        svdd.generate_glove("bigvul")
    if "d2v" in sys.argv:
        svdd.generate_d2v("bigvul")
    print("done")


if __name__ == "__main__":
    bigvul()
