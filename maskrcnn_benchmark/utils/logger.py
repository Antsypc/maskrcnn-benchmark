# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import os
import sys


def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

# every logger dose logging and save to log_{}.txt file
# def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
#     logger = logging.getLogger(name)
#     logger.setLevel(logging.DEBUG)
#     formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
#     # don't log results for the non-master process
#     if distributed_rank == 0:
#         ch = logging.StreamHandler(stream=sys.stdout)
#         ch.setLevel(logging.DEBUG)
#         ch.setFormatter(formatter)
#         logger.addHandler(ch)
#
#     if save_dir:
#         filename = "log_{}.txt".format(distributed_rank)
#         fh = logging.FileHandler(os.path.join(save_dir, filename))
#         fh.setLevel(logging.DEBUG)
#         fh.setFormatter(formatter)
#         logger.addHandler(fh)
#     return logger
