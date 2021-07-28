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
    reload(svdj)
    svdj.full_run_joern_from_string(test_func, "test", "test")
    filepath = svd.interim_dir() / "test" / "test.c"
    nodes, edges = svdj.get_node_edges(filepath)
    assert len(nodes) == 74
    assert len(edges) == 116


# print(before_func)
# df = svdd.bigvul()
# before_func = df.iloc[1].before
# after_func = df.iloc[3].after
# test_joern_graph()
# path = svd.interim_dir() / f"{items[iid]['dataset']}/{items[iid]['id']}.c"
# svdj.get_node_edges(path)[0]
