import numpy as np
from pycocotools import mask as mask_utils


def encode_binary_mask(binary_mask):
    arr = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_utils.encode(arr)
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("utf-8")
    return {"size": [int(s) for s in rle["size"]], "counts": counts}


def decode_rle(rle_dict):
    counts = rle_dict["counts"]
    if isinstance(counts, str):
        counts = counts.encode("utf-8")
    return mask_utils.decode({"size": rle_dict["size"], "counts": counts})
