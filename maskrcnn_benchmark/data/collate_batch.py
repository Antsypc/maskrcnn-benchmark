# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from maskrcnn_benchmark.structures.image_list import to_image_list


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        images = to_image_list(transposed_batch[0], self.size_divisible)
        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        return images, targets, img_ids
#         return SimpleCustomBatch(batch, self.size_divisible)
#
#
# class SimpleCustomBatch(object):
#
#     def __init__(self, batch, size_divisible):
#         transposed_batch = list(zip(*batch))
#         self.images = to_image_list(transposed_batch[0], size_divisible)
#         self.targets = transposed_batch[1]
#         self.img_ids = transposed_batch[2]
#
#     def pin_memory(self):
#         self.images.tensors = self.images.tensors.pin_memory()
#         for target in self.targets:
#             target.bbox = target.bbox.pin_memory()
#         return self
