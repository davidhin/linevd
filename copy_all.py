import os
import shutil
import tqdm
import multiprocessing
import traceback

from_d = "storage/processed/bigvul/before/"
to_d = "storage/processed/bigvul/old/before/"

def do(l):
    try:
        shutil.move(from_d + l, to_d + l)
    except Exception:
        traceback.print_exc()

#with multiprocessing.Pool(4) as pool:
lis = os.listdir(from_d)
for _ in tqdm.tqdm(map(do, lis), total=len(lis)):
    pass
