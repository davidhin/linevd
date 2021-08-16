import sastvd.helpers.datasets as svdd
import sastvd.ivdetect.evaluate as ivde
import sastvd.ivdetect.helpers as ivdh


def bigvul():
    """Run preperation scripts for BigVul dataset."""
    svdd.bigvul()
    ivde.get_dep_add_lines_bigvul()
    ivdh.BigVulGraphDataset(partition="train").cache_features()
    ivdh.BigVulGraphDataset(partition="val").cache_features()
    ivdh.BigVulGraphDataset(partition="test").cache_features()
    svdd.generate_glove("bigvul")
