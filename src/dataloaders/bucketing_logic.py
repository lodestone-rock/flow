import os
import csv
import sys
import math
import random
import logging
from tqdm import tqdm
import pandas as pd

import multiprocessing as mp
from functools import partial
from itertools import islice
from tqdm import tqdm
import psutil

from .utils import read_jsonl

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2147483647)
log = logging.getLogger(__name__)


def _bucket_generator(base_resolution=256, ratio_cutoff=2, step=64):
    # bucketing follows the logic of 1/x
    # https://www.desmos.com/calculator/byizruhsry
    x = base_resolution
    y = base_resolution
    half_buckets = [(x, y)]
    while y / x <= ratio_cutoff:
        y += step
        x = int((base_resolution**2 / y) // step * step)
        if x != half_buckets[-1][0]:
            half_buckets.append(
                (
                    x,
                    y,
                )
            )

    another_half_buckets = []
    for bucket in reversed(half_buckets[1:]):
        another_half_buckets.append(bucket[::-1])

    return another_half_buckets + half_buckets


def _euclidian_distance_2d(x1, y1, x2, y2):
    result = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return result


def _normalize_width_height(w, h):
    # scale the width and height to a unit value
    t = math.sqrt(1.0 / (w * h))
    w = w * t
    h = h * t
    return w, h


def _closest_bucket(w, h, standarized_bucket_dict):
    w, h = _normalize_width_height(w, h)

    # loop over the bucket and compute the closest 2d euclidian distance to the bucket
    # this ensures the closest bucket and not exceeding the threshold
    ration_err = []  # store the error for all possible bucket
    for b_st in standarized_bucket_dict.keys():
        ration_err.append(
            (_euclidian_distance_2d(*b_st, w, h), standarized_bucket_dict[b_st])
        )
    # loop through and find the shortest distance
    min_err, closest_b = ration_err[0]
    for err, b in ration_err:
        if err < min_err:
            min_err = err
            closest_b = b

    return closest_b


def create_bucket_column(
    in_csv_path,
    out_csv_path,
    base_resolution,  # Need to be an array like this: `[384, 512, 640, 768, 896, 1024]`
    step=8,
    ratio_cutoff=2,
    height_col_name="image_height",
    width_col_name="image_width",
    bucket_col_name="bucket",
    return_bucket=False,
):
    # this may explode if the csv is ridiculously big
    all_rows = []

    with open(in_csv_path, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)
        for row in tqdm(reader, desc="loading csv"):
            all_rows.append(row)

    # image resolution metadata columns
    image_height_index = header.index(height_col_name)
    image_width_index = header.index(width_col_name)

    multires_buckets = []
    for res in base_resolution:
        buckets = _bucket_generator(res, ratio_cutoff, step)
        multires_buckets.append(buckets)

    # bucket scaling factor normalized
    # for example (1, 1) meaning it's square scaled
    # (1.41, 0.70) <- (1448, 720) non square normalization
    standardized_buckets = list()
    for buckets in tqdm(
        multires_buckets, total=len(multires_buckets), desc="creating multi-res buckets"
    ):
        this_standardized_bucket = dict()
        for bucket in buckets:
            b_st = _normalize_width_height(*bucket)  # (w, h)
            this_standardized_bucket[b_st] = bucket
        standardized_buckets.append(this_standardized_bucket)

    # iterate csv rows
    for row in tqdm(all_rows, total=len(all_rows), desc="appending bucket"):
        # grab image width and height
        image_width = int(row[image_width_index])
        image_height = int(row[image_height_index])

        # guard check
        if image_width > 0 and image_height > 0:
            aspect_ratio = image_width / image_height
            if ratio_cutoff > aspect_ratio > 1.0 / ratio_cutoff:
                w, h = _normalize_width_height(image_width, image_height)

                # randomly choose a resolution
                this_standardized_bucket = random.choice(standardized_buckets)

                closest_b = _closest_bucket(w, h, this_standardized_bucket)
                row.append(str(closest_b))
            else:
                pass
        else:
            pass

    # save it
    with open(out_csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header.append(bucket_col_name)
        writer.writerow(header)
        for row in all_rows:
            writer.writerow(row)

    if return_bucket:
        return standardized_buckets


def create_bucket_column_pandas(
    in_csv_path,
    out_path,
    base_resolution,  # Need to be an array like this: `[384, 512, 640, 768, 896, 1024]`
    step=8,
    ratio_cutoff=2,
    height_col_name="image_height",
    width_col_name="image_width",
    chunksize=1000,
):
    # NOTE: BE CAREFUL THIS FUNCTION APPENDS to OUTPUT CSV RATHER THAN OVERWRITE!
    # create a list of resolution bucket given base resolution
    # example
    # 384: [(512, 256), (448, 320), (384, 384), (320, 448), (256, 512)]
    # 512: [(704, 320), (640, 384), (576, 448), (512, 512), (448, 576), (384, 640), (320, 704)]
    multires_buckets = {}
    for res in base_resolution:
        buckets = _bucket_generator(res, ratio_cutoff, step)
        multires_buckets[res] = buckets

    # create scaled version for easy distance calculation
    # {
    #     (1.41, 0.70): (512, 256),
    #     (1.18, 0.84): (448, 320),
    #     (1.0, 1.0): (384, 384),
    #     (0.84, 1.18): (320, 448),
    #     (0.70, 1.41): (256, 512)
    # }
    standardized_buckets = {}
    for res, buckets in tqdm(
        multires_buckets.items(),
        total=len(multires_buckets),
        desc="creating multi-res buckets",
    ):
        this_standardized_bucket = {}
        for bucket in buckets:
            b_st = _normalize_width_height(*bucket)  # (w, h)
            this_standardized_bucket[b_st] = bucket
        standardized_buckets[res] = this_standardized_bucket

    csv_pbar = tqdm(desc="Processing csv chunks", unit_scale=chunksize)

    for df in pd.read_csv(in_csv_path, chunksize=chunksize):
        df["aspect_ratio"] = df[width_col_name] / df[height_col_name]
        # remove ridiculous aspect ratio
        df = df[
            (df.aspect_ratio > 1.0 / ratio_cutoff) & (df.aspect_ratio < ratio_cutoff)
        ]
        # normalize width height into relative width height for computing bucket assignment
        df["norm_width_height"] = df.apply(
            lambda row: _normalize_width_height(
                row[width_col_name], row[height_col_name]
            ),
            axis=1,
        )

        # assign bucket based on euclidean distance
        for res, buckets in standardized_buckets.items():
            df[f"{res}"] = df.apply(
                lambda row: _closest_bucket(*row["norm_width_height"], buckets), axis=1
            )

        if os.path.join(*os.path.split(out_path)[:-1]) != "":
            os.makedirs(os.path.join(*os.path.split(out_path)[:-1]), exist_ok=True)
        # df.to_csv(out_csv_path, mode='a', index=False, header=not pd.io.common.file_exists(out_csv_path))
        df.to_json(out_path, mode="a", orient="records", lines=True)
        csv_pbar.update(1)
        print()


def create_bucket_jsonl(
    jsonl_path,
    base_resolution,  # Need to be an array like this: `[384, 512, 640, 768, 896, 1024]`
    step=8,
    ratio_cutoff=2,
    height_key_name="height",
    width_key_name="width",
):
    jsonl = read_jsonl(jsonl_path)
    # create a list of resolution bucket given base resolution
    # example
    # 384: [(512, 256), (448, 320), (384, 384), (320, 448), (256, 512)]
    # 512: [(704, 320), (640, 384), (576, 448), (512, 512), (448, 576), (384, 640), (320, 704)]
    multires_buckets = {}
    for res in base_resolution:
        buckets = _bucket_generator(res, ratio_cutoff, step)
        multires_buckets[res] = buckets

    # create scaled version for easy distance calculation
    # {
    #     (1.41, 0.70): (512, 256),
    #     (1.18, 0.84): (448, 320),
    #     (1.0, 1.0): (384, 384),
    #     (0.84, 1.18): (320, 448),
    #     (0.70, 1.41): (256, 512)
    # }
    standardized_buckets = {}
    for res, buckets in tqdm(
        multires_buckets.items(),
        total=len(multires_buckets),
        desc="creating multi-res buckets",
    ):
        this_standardized_bucket = {}
        for bucket in buckets:
            b_st = _normalize_width_height(*bucket)  # (w, h)
            this_standardized_bucket[b_st] = bucket
        standardized_buckets[res] = this_standardized_bucket

    processed_jsonl = []
    for i in tqdm(range(len(jsonl)), desc="generating bucket"):
        item = jsonl.pop(0)
        # grab image width and height
        image_width = int(item[width_key_name])
        image_height = int(item[height_key_name])

        # compute the closest bucket while also removing extreme aspect ratio
        if image_width > 0 and image_height > 0:  # guard check
            aspect_ratio = image_width / image_height
            if (
                ratio_cutoff > aspect_ratio > 1.0 / ratio_cutoff
            ):  # remove extreme aspect ratio
                w, h = _normalize_width_height(image_width, image_height)

                # enumerate all possible bucket
                item["buckets"] = []
                for res, buckets in standardized_buckets.items():
                    closest_bucket = _closest_bucket(w, h, buckets)
                    item["buckets"].append(closest_bucket)

                processed_jsonl.append(item)
            else:
                log.info("deleted", item["filename"], f"bad aspect ratio", aspect_ratio)
                pass
        else:
            log.info("deleted", item["filename"], "zero px metadata")
            pass

    return processed_jsonl


def chunk_list(lst, n):
    """Split list into n chunks of approximately equal size"""
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m) : (i + 1) * k + min(i + 1, m)] for i in range(n)]


def process_chunk(
    chunk, standardized_buckets, ratio_cutoff, width_key_name, height_key_name
):
    """Process a chunk of JSONL data"""
    processed_chunk = []

    for item in chunk:
        # grab image width and height
        image_width = int(item[width_key_name])
        image_height = int(item[height_key_name])

        # compute the closest bucket while also removing extreme aspect ratio
        if image_width > 0 and image_height > 0:  # guard check
            aspect_ratio = image_width / image_height
            if (
                ratio_cutoff > aspect_ratio > 1.0 / ratio_cutoff
            ):  # remove extreme aspect ratio
                w, h = _normalize_width_height(image_width, image_height)

                # enumerate all possible bucket
                item["buckets"] = []
                for res, buckets in standardized_buckets.items():
                    closest_bucket = _closest_bucket(w, h, buckets)
                    item["buckets"].append(closest_bucket)

                processed_chunk.append(item)
            else:
                log.info("deleted", item["filename"], f"bad aspect ratio", aspect_ratio)
        else:
            log.info("deleted", item["filename"], "zero px metadata")

    return processed_chunk


def create_bucket_jsonl(
    jsonl_path,
    base_resolution,  # Need to be an array like this: `[384, 512, 640, 768, 896, 1024]`
    step=8,
    ratio_cutoff=2,
    height_key_name="height",
    width_key_name="width",
    num_processes=None,
):
    """Create bucket JSONL with multiprocessing support"""
    if num_processes is None:
        num_processes = psutil.cpu_count(logical=False) - 1

    jsonl = read_jsonl(jsonl_path)

    # create a list of resolution bucket given base resolution
    multires_buckets = {}
    for res in base_resolution:
        buckets = _bucket_generator(res, ratio_cutoff, step)
        multires_buckets[res] = buckets

    # create scaled version for easy distance calculation
    standardized_buckets = {}
    for res, buckets in tqdm(
        multires_buckets.items(),
        total=len(multires_buckets),
        desc="creating multi-res buckets",
    ):
        this_standardized_bucket = {}
        for bucket in buckets:
            b_st = _normalize_width_height(*bucket)  # (w, h)
            this_standardized_bucket[b_st] = bucket
        standardized_buckets[res] = this_standardized_bucket

    # Split the data into chunks
    chunks = chunk_list(jsonl, num_processes)

    # Create a partial function with the fixed arguments
    process_func = partial(
        process_chunk,
        standardized_buckets=standardized_buckets,
        ratio_cutoff=ratio_cutoff,
        width_key_name=width_key_name,
        height_key_name=height_key_name,
    )

    # Process chunks in parallel
    with mp.Pool(processes=num_processes) as pool:
        results = list(
            tqdm(
                pool.imap(process_func, chunks),
                total=len(chunks),
                desc="generating buckets (parallel)",
            )
        )

    # Combine results from all processes
    processed_jsonl = []
    for chunk_result in results:
        processed_jsonl.extend(chunk_result)

    return processed_jsonl
