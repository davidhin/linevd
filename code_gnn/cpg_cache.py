import os
import pickle
import shutil

# Utilities to get the CPG from files
class CPGCache:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)

    def get_source_filename(self, index):
        return str(index) + '.c'
    def get_cpg_filename(self, index):
        return self.get_source_filename(index) + '_cpg.pkl'
    def get_cpg_filepath(self, index):
        return self.cache_dir / self.get_cpg_filename(index)
    def write_cpg(self, index, cpg):
        fname = self.get_cpg_filepath(index)
        with open(fname, 'wb') as f:
            pickle.dump(cpg, f)
    def load_cpg(self, index):
        fname = self.get_cpg_filepath(index)
        if os.path.exists(fname):
            with open(fname, 'rb') as f:
                return pickle.load(f)
    def clear(self):
        shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
