import os
import math
import json
import random
import hashlib
import logging

from tqdm import tqdm
from PIL import Image
import pillow_avif
import imageio.v3 as iio
import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as v2

from io import BytesIO
import concurrent.futures
from requests.exceptions import RequestException, Timeout

from .utils import read_jsonl
from .tag_preprocess_utils import create_tree, prune
from . import color_profile_handling
from .bucketing_logic import create_bucket_jsonl
import psutil

log = logging.getLogger(__name__)


class TextImageDataset(Dataset):
    def __init__(
        self,
        batch_size,
        jsonl_path,
        image_folder_path,
        tag_implication_path=None,
        ratio_cutoff=2.0,
        resolution_step=8,
        base_res=[
            1024
        ],  # resolution computation should be in here instead of in csv prep!
        shuffle_tags=True,
        tag_drop_percentage=0.8,
        uncond_percentage=0.01,
        seed=0,
        rank=0,
        num_gpus=1,
        timeout=10,
        thread_per_worker=100,
    ):
        # coarsened dataset, the batch is handled by the dataset and not the dataloader,
        # increase dataloader prefetch so this thing  run optimally!

        self.jsonl_path = jsonl_path
        self.tag_implication_path = tag_implication_path
        self.batch_size = batch_size
        self.base_res = base_res
        self.ratio_cutoff = ratio_cutoff
        self.resolution_step = resolution_step
        self.image_folder_path = image_folder_path
        self.tag_drop_percentage = tag_drop_percentage  # only  applicable to tags
        self.shuffle_tags = shuffle_tags  # replace this with "shuffle_tags" because we're going to put the metadata in the jsonl if it's a tag or caption based
        self.uncond_percentage = uncond_percentage
        self.num_gpus = num_gpus
        self.rank = rank
        self.rank_batch_size = batch_size // num_gpus
        self.timeout = timeout
        assert (
            batch_size % num_gpus
        ) == 0, "batch size is not divisible by the number of GPUs!"

        random.seed(seed)
        # just simple pil image to tensor conversion
        self.image_transforms = v2.Compose(
            [v2.ToImage(), v2.ToDtype(torch.float32, scale=True)]
        )

        # TODO: batches has to be preprocessed for batching!!!!
        self.batches = self._load_batches()

        # slice batches using round robbin
        self._round_robin()
        self.thread_per_worker = thread_per_worker
        # self.executor = concurrent.futures.ThreadPoolExecutor(thread_per_worker)

    def _load_batches(self):
        batch_size = self.batch_size

        jsonl = create_bucket_jsonl(
            jsonl_path=self.jsonl_path,
            base_resolution=self.base_res,
            ratio_cutoff=self.ratio_cutoff,
            height_key_name="height",
            width_key_name="width",
            step=self.resolution_step,
            num_processes=psutil.cpu_count(logical=False) // self.num_gpus - 1,
        )

        if self.tag_implication_path is not None:
            implication_tree = create_tree(self.tag_implication_path)

        buckets = {}
        for i in tqdm(range(len(jsonl)), desc="preparing captions and buckets"):
            if self.tag_implication_path is not None and self.tag_based:
                caption_or_tags = prune(
                    jsonl[i]["caption_or_tags"].split(" "), implication_tree
                )
            else:
                caption_or_tags = jsonl[i]["caption_or_tags"]
            res_bucket = tuple(random.choice(jsonl[i]["buckets"]))  # use seed!!

            # this is already in standarized format
            sample = {
                "filename": jsonl[i]["filename"],
                "caption_or_tags": caption_or_tags,
                "bucket": res_bucket,
                "is_tag_based": jsonl[i]["is_tag_based"],
                "is_url_based": jsonl[i]["is_url_based"],
                "loss_weight": jsonl[i]["loss_weight"]
            }

            if res_bucket in buckets:
                buckets[res_bucket].append(sample)
            else:
                buckets[res_bucket] = [sample]

        # pre-shuffle bucket for maximum randomness :P
        for key in tqdm(buckets.keys(), desc="shuffling buckets"):
            random.shuffle(buckets[key])

        # simple post counter
        post_count = 0
        for k in buckets:
            post_count += len(buckets[k])
        print(f"There are {post_count} text image pairs.")

        # arrange into batches
        batches = []
        for b in buckets:
            samples = []
            # if not full batch drop the last batch to prevent bad things
            for s in buckets[b]:
                if len(samples) < batch_size:
                    samples.append(s)
                elif len(samples) == batch_size:
                    batches.append(samples)
                    samples = [s]
        print(f"We got {len(batches)} batches from these pairs.")

        random.shuffle(batches)
        return batches

    # <some utility method here>
    @staticmethod
    def dummy_collate_fn(batch):
        return batch

    @staticmethod
    def scale_and_crop_long_axis(image: Image, target_height, target_width):
        # resize image to the standard size of this bucket
        # surely we could replace the interpolation with pytorch ops right?
        # but optimizing this might have meaningless impact so let it be atm
        if (target_width / target_height) >= (image.width / image.height):
            image = v2.functional.resize(
                image,
                [round(target_width * image.height / image.width), target_width],
                interpolation=v2.InterpolationMode.LANCZOS,
            )
            image = v2.functional.center_crop(
                image,
                [target_height, target_width],
            )
        else:
            image = v2.functional.resize(
                image,
                [target_height, round(image.width * target_height / image.height)],
                interpolation=v2.InterpolationMode.LANCZOS,
            )
            image = v2.functional.center_crop(
                image,
                [target_height, target_width],
            )
        return image

    @staticmethod
    def _sample_elements_by_percentage(my_list, percentage):
        if not 0 <= percentage <= 1:
            raise ValueError("Percentage must be between 0 and 1")

        sample_size = math.ceil(len(my_list) * percentage)
        return random.sample(my_list, sample_size)

    def get_hash(self):
        s = json.dumps(self.batches)
        str_sha256 = hashlib.sha256(s.encode("utf-8")).hexdigest()
        return str(str_sha256)

    def get_batches(self):
        return self.batches

    # def _round_robin(self):
    #     # reason we do round robbin here instead of classic torch distributed batch is because
    #     # we have bucketing and the shape is different for each gpu
    #     # drop last batch
    #     self.batches = self.batches[
    #         : len(self.batches) - len(self.batches) % self.num_gpus
    #     ]
    #     # slice round-robin
    #     subset_for_this_worker = []
    #     for i in range(0, len(self.batches), self.num_gpus):
    #         subset_for_this_worker.append(self.batches[i + self.rank])
    #         # print(i + self.rank)
    #     self.batches = subset_for_this_worker

    def _round_robin(self):
        # slice round-robin
        subset_for_this_worker = []
        for batch in self.batches:
            subset_for_this_worker.append(
                batch[
                    self.rank * self.rank_batch_size : self.rank * self.rank_batch_size
                    + self.rank_batch_size
                ]
            )
        self.batches = subset_for_this_worker

    # </some utility method here>

    def _load_image(self, sample, image_folder_path, timeout):
        try:
            img_array = None
            if sample["is_url_based"]:
                img_array = iio.imread(sample["filename"])
            else:
                image_path = os.path.join(image_folder_path, sample["filename"])

                # Check if a JXL version of the file exists and prioritize it
                jxl_image_path = os.path.splitext(image_path)[0] + ".jxl"
                if os.path.exists(jxl_image_path):
                    # Custom handling for JXL format
                    return color_profile_handling.open_srgb(jxl_image_path).convert(
                        "RGB"
                    )
                elif os.path.exists(image_path):
                    # Standard handling if the specified file exists
                    img_array = iio.imread(image_path)
                else:
                    # Try alternative extensions if the main file doesn't exist
                    filename, _ = os.path.splitext(sample["filename"])
                    extensions = ["png", "jpg", "jpeg", "webp", "bmp", "avif", "tif", "tiff"]
                    for ext in extensions:
                        alt_image_path = os.path.join(
                            image_folder_path, f"{filename}.{ext}"
                        )
                        if os.path.exists(alt_image_path):
                            img_array = iio.imread(alt_image_path)
            if img_array is None:
                return None
            else:
                if img_array.ndim == 2:
                    image = Image.fromarray(img_array, mode='L')
                elif img_array.ndim == 3 or (img_array.ndim == 4 and img_array.shape[0] == 1):
                    if img_array.ndim == 4:
                        # When the image has a frame dimension, only the first frame is taken.
                        img_array = img_array[0]
                    height, width, channels = img_array.shape
                    if channels == 3:  # RGB
                        image = Image.fromarray(img_array, mode='RGB')
                    elif channels == 4:  # RGBA
                        image = Image.fromarray(img_array, mode='RGBA')
                    else:
                        raise ValueError(f"Unsupported number of channels: {channels}")
                else:
                    raise ValueError(f"Unsupported image shape: {img_array.shape}")

                if image.mode != 'RGB':
                    image = image.convert('RGB')
                
                return image
        except Exception as e:
            log.error(
                f"An error occurred: {e} for {sample['filename']} on rank {self.rank}"
            )
            return None

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, index):
        batch = self.batches[index]  # the batches is already batched and it's a list

        # Use threading for concurrent image loading
        raw_images = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._load_image,
                    sample,
                    self.image_folder_path,
                    self.timeout,
                )
                for sample in batch
            ]
            raw_images = [future.result() for future in futures]

        images = []
        training_prompts = []
        loss_weighting = []

        for i, sample in enumerate(batch):
            try:

                standard_width, standard_height = sample["bucket"]

                image = self.scale_and_crop_long_axis(
                    raw_images[i], standard_height, standard_width
                )
                image = self.image_transforms(image)
                images.append(image)

                # unconditional drop out
                tmp = random.random()
                if tmp >= 1 - self.uncond_percentage:
                    sample["caption_or_tags"] = ""

                # tags processing
                if self.shuffle_tags and sample["is_tag_based"]:

                    tags = sample["caption_or_tags"].split(",")
                    random.shuffle(tags)

                    tags = self._sample_elements_by_percentage(
                        tags, random.uniform(1 - self.tag_drop_percentage, 1)
                    )
                    tags = ",".join(tags).lstrip()
                    training_prompts.append(tags)
                else:
                    training_prompts.append(sample["caption_or_tags"])

                if "loss_weight" in sample:
                    loss_weighting.append(sample["loss_weight"])
                else:
                    loss_weighting.append(1)

            except Exception as e:
                log.error(
                    f"An error occurred: {e} for {sample['filename']} on rank {self.rank}"
                )
                continue

        # echo short batch
        if self.rank_batch_size > 1:
            while len(images) < self.rank_batch_size:
                log.info(
                    f"only {len(images)} out of {self.rank_batch_size} exist, echoing"
                )
                echoed_index = random.choice(list(range(len(images))))
                images.append(images[echoed_index])
                training_prompts.append(training_prompts[echoed_index])
                loss_weighting.append(loss_weighting[echoed_index])

        # resample randomly if the entire batch failed
        while len(images) < 1:
            log.info(
                f"An Empty batch is caught! This batch will be discarded and a random batch will be fetch."
            )
            new_batch_index = random.randrange(0, len(self.batches))
            return self.__getitem__(new_batch_index)

        images = torch.stack(images, dim=0)

        return images, training_prompts, index, loss_weighting
