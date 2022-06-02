import argparse
import sastvd as svd
import sastvd.helpers.datasets as svdd
import sastvd.ivdetect.evaluate as ivde
import sys


def bigvul():
    """Run preperation scripts for BigVul dataset."""
    svdd.bigvul(sample=args.sample)
    ivde.get_dep_add_lines_bigvul(sample=args.sample)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare master dataframe')
    parser.add_argument('--sample', action='store_true', help='Extract a sample only')
    parser.add_argument('--global_workers', type=int, help='Number of workers to use')
    args = parser.parse_args()

    if args.global_workers is not None:
        svd.DFMP_WORKERS = args.global_workers
    
    bigvul()
