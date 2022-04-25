import logging

from dgl.dataloading import GraphDataLoader

logger = logging.getLogger(__name__)


class MyDataLoader(GraphDataLoader):
    def __init__(self, dataset, shuffle, batch_size, *args, **kwargs):
        super().__init__(dataset, shuffle=shuffle, batch_size=batch_size, *args, **kwargs)
        self.has_warned = False

    def set_epoch(self, epoch):
        if self.use_ddp:
            self.dist_sampler.set_epoch(epoch)
        elif not self.has_warned:
            # logger.warning('set_epoch is only available when use_ddp is True.')
            self.has_warned = True
