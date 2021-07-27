import sastvd.helpers.datasets as svdd
import sastvd.helpers.joern as svdj


def test_joern_graph():

    df = svdd.bigvul()
    before_func = df.iloc[0].before
    svdj.full_run_joern_from_string(before_func, "bigvul", "test")


test_joern_graph()
# path = svd.interim_dir() / f"{items[iid]['dataset']}/{items[iid]['id']}.c"
# svdj.get_node_edges(path)[0]
