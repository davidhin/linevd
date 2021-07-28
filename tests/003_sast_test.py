import sastvd.helpers.datasets as svdd
import sastvd.helpers.sast as sast


def test_sast_1():
    df = svdd.bigvul()
    sample = df[df.id == 182193]
    ret = sast.run_sast(sample.before.item(), verbose=1)
    assert len(ret) == 7
