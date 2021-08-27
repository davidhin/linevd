from importlib import reload

import sastvd as svd
import sastvd.helpers.joern as svdj

test_func = """\
short add (short b) {
    short a = 32767;
    if (b > 0) {
        a = a + b;
    }
    return a;
}
"""


def test_joern_graph():
    """Test 1."""
    reload(svdj)
    svdj.full_run_joern_from_string(test_func, "test", "test")
    filepath = svd.interim_dir() / "test" / "test.c"
    nodes, edges = svdj.get_node_edges(filepath)
    assert len(nodes) == 53
    assert len(edges) == 116


# Bigvul suspicious Joern IDs
# 178958, 179986, 180111, 180254, 180256
# 183008, 186024, 186854, 185466, 186856
# 179552, 185465, 117854, 182671, 183185
# 179986, 179989, 180109, 180110, 180187
# 180244, 180249, 180249, 180252, 180253

# print(before_func)
# df = svdd.bigvul()
# before_func = df.iloc[1].before
# after_func = df.iloc[3].after
# test_joern_graph()
# path = svd.interim_dir() / f"{items[iid]['dataset']}/{items[iid]['id']}.c"
# svdj.get_node_edges(path)[0]
