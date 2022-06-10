conda create --name linevd python=3.8 tqdm numpy gdown nltk gensim matplotlib pydot seaborn pytest scikit-learn scipy pandas fastparquet networkx graphviz tensorboard unidiff pytorch-lightning  pytorch torchvision torchaudio cudatoolkit=11.3 dgl-cuda11.3 -c conda-forge -c pytorch -c dglteam
pip install transformers ray[tune]
conda install pexpect psutil