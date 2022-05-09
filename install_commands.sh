conda install -c dglteam dgl-cuda10.2
cat requirements.txt | xargs -n 1 pip install
pip install dgl-cu102 -f https://data.dgl.ai/wheels/repo.html
conda install -y pygraphviz
pip install nltk
python -c 'import nltk; nltk.download("punkt")'