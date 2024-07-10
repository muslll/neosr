import queue as Queue
from collections.abc import Iterator
from threading import Thread
from typing import Any

import torch
from torch.utils.data import DataLoader


class PrefetchGenerator(Thread):
    """A general prefetch generator.

    Reference: https://stackoverflow.com/questions/7323664/python-generator-pre-fetch

    Args:
    ----
        generator: Python generator.
        num_prefetch_queue (int): Number of prefetch queue.

    """

    def __init__(self, generator, num_prefetch_queue: int) -> None:
        Thread.__init__(self)
        self.queue: Queue.Queue[Any] = Queue.Queue(num_prefetch_queue)
        self.generator = generator
        self.daemon = True
        self.start()

    def run(self) -> None:
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def __next__(self) -> Any:
        next_item = self.queue.get()  # type: Any
        if next_item is None:
            raise StopIteration
        return next_item

    def __iter__(self) -> Iterator:
        return self


class PrefetchDataLoader(DataLoader):
    """Prefetch version of dataloader.

    Reference: https://github.com/IgorSusmelj/pytorch-styleguide/issues/5#

    Todo:
    ----
    Need to test on single gpu and ddp (multi-gpu). There is a known issue in
    ddp.

    Args:
    ----
        num_prefetch_queue (int): Number of prefetch queue.
        kwargs (dict): Other arguments for dataloader.

    """

    def __init__(self, num_prefetch_queue: int, **kwargs) -> None:
        self.num_prefetch_queue = num_prefetch_queue
        super().__init__(**kwargs)

    def __iter__(self):  # type: ignore[reportIncompatibleMethodOverride]
        return PrefetchGenerator(super().__iter__(), self.num_prefetch_queue)


class CUDAPrefetcher:
    """CUDA prefetcher.

    Reference: https://github.com/NVIDIA/apex/issues/304#

    It may consume more GPU memory.

    Args:
    ----
        loader: Dataloader.
        opt (dict): Options.

    """

    def __init__(self, loader: DataLoader, opt: dict[str, Any]) -> None:
        self.ori_loader = loader
        self.loader = iter(loader)
        self.opt = opt
        self.stream = torch.cuda.Stream()
        self.device = torch.device("cuda")
        self.preload()

    def preload(self) -> None:
        try:
            self.batch = next(self.loader)  # self.batch is a dict
        except StopIteration:
            self.batch = None
            return
        # put tensors to gpu
        with torch.cuda.stream(self.stream):  # type: ignore[reportArgumentType]
            for k, v in self.batch.items():
                if torch.is_tensor(v):
                    self.batch[k] = self.batch[k].to(
                        device=self.device, non_blocking=True
                    )

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch

    def reset(self) -> None:
        self.loader = iter(self.ori_loader)
        self.preload()
