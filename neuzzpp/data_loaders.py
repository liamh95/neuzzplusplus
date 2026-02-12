# Copyright (c) 2023 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Module containing data loaders for model training and evaluation."""
import logging
import os
import pathlib
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
# import tensorflow as tf
import torch
# from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from neuzzpp.preprocess import create_bitmap_from_raw_coverage, create_path_coverage_bitmap
from neuzzpp.utils import get_max_file_size

logger = logging.getLogger(__name__)


class SeedFolderDataset(Dataset):
    """PyTorch Dataset for seeds from a folder with coverage bitmaps."""

    DEFAULT_PERCENTILE = 80

    def __init__(
        self,
        seeds_path: pathlib.Path,
        target: List[str],
        bitmap_func: Callable[
            [List[str], List[pathlib.Path]], Dict[pathlib.Path, Optional[Set[int]]]
        ],
        max_len: Optional[int] = None,
        percentile_len: int = DEFAULT_PERCENTILE,
    ) -> None:
        """
        Initialize a PyTorch Dataset for seeds and their coverage bitmaps.

        Args:
            seeds_path: Path to the seeds folder.
            target: Name of the fuzzing target in a callable format with its arguments.
            bitmap_func: Callable that computes coverage bitmaps for given seeds.
            max_len: Optional limit for the maximum seed length.
            percentile_len: Optional length percentile to keep as maximum seed length (1-100);
                ignored if `max_len` is provided.
        """
        if max_len is not None and max_len <= 0:
            err = f"Maximum seed length must be greater than 0, received: {max_len}"
            logger.exception(err)
            raise ValueError(err)

        self.seeds_path = pathlib.Path(seeds_path)
        self.target = target
        self.bitmap_func = bitmap_func
        self.max_len = max_len
        self.percentile_len = percentile_len
        self.seed_list: List[pathlib.Path] = []
        self.raw_coverage_info: Dict[pathlib.Path, Set[int]] = {}
        self.reduced_bitmap: Optional[np.ndarray] = None
        self.max_file_size: int = 0
        self.max_bitmap_size: int = 0

        # Load seeds on initialization?
        self.load_seeds_from_folder()

    def load_seeds_from_folder(self) -> None:
        """
        Load seeds information from folder and compute coverage bitmaps.
        Only new seeds are considered since last load.
        """
        # Determine new seeds in folder
        new_seed_list: List[pathlib.Path] = [
            seed for seed in self.seeds_path.glob("*") if seed.is_file()
        ]
        if self.seed_list:
            new_seeds = list(set(new_seed_list) - set(self.seed_list))
        else:
            new_seeds = new_seed_list

        # Get coverage for new seeds and recompute coverage bitmap
        if new_seeds:
            new_raw = self.bitmap_func(self.target, new_seeds)
            filtered_raw: Dict[pathlib.Path, Set[int]] = {
                seed: cov for seed, cov in new_raw.items()
                if cov is not None
            }
            self.raw_coverage_info.update(filtered_raw)
            self.seed_list, self.reduced_bitmap = create_bitmap_from_raw_coverage(
                self.raw_coverage_info
            )

        # Compute "optimal" max seed length if not provided
        max_file_size = get_max_file_size(self.seeds_path)
        if self.max_len is None:
            max_len = compute_input_size_from_seeds(self.seeds_path, percentile=self.percentile_len)
            logger.info(
                f"No max length provided. "
                f"Using {max_len} based on existing seeds ({self.percentile_len}%)."
            )
        else:
            max_len = self.max_len
        self.max_file_size = min(max_file_size, max_len)
        if self.reduced_bitmap is not None:
            self.max_bitmap_size = self.reduced_bitmap.shape[1]

    def __len__(self) -> int:
        """Return the number of seeds in the dataset."""
        return len(self.seed_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return normalized seed and its coverage bitmap at index.

        Args:
            idx: Index of the seed to retrieve.

        Returns:
            Tuple of (normalized_seed, coverage_bitmap) as torch tensors.
        """
        seed_path = self.seed_list[idx]
        seed = read_seed(seed_path) # this is an np.ndarray

        seed = seed[-self.max_file_size:]
        pad_length = self.max_file_size - len(seed)
        if pad_length > 0:
            seed = np.pad(seed, (0, pad_length), mode="constant")
        seed_normalized = seed.astype("float32") / 255.0

        # Get bitmap
        assert self.reduced_bitmap is not None
        bitmap = self.reduced_bitmap[idx].astype("int32")

        return torch.from_numpy(seed_normalized), torch.from_numpy(bitmap)

    def get_class_weights(self) -> Tuple[Dict[int, float], float]:
        """
        Compute class weights and initial bias based on coverage distribution.

        Returns:
            Tuple of (class_weights_dict, initial_bias).
        """
        assert self.reduced_bitmap is not None
        bitmap_flat = self.reduced_bitmap.flatten()
        n_neg = np.count_nonzero(bitmap_flat == 0)
        n_total = bitmap_flat.size
        n_pos = n_total - n_neg

        weight_for_uncovered = (1.0 / n_neg) * (n_total / 2.0) if n_neg > 0 else 1.0
        weight_for_covered = (1.0 / n_pos) * (n_total / 2.0) if n_pos > 0 else 1.0
        class_weights = {0: weight_for_uncovered, 1: weight_for_covered}
        initial_bias = float(np.log([n_pos / n_neg])) if n_neg > 0 else 0.0

        logger.info(
            "Dataset:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n".format(
                n_total, n_pos, 100 * n_pos / n_total
            )
        )

        return class_weights, initial_bias


class CoverageSeedDataset(SeedFolderDataset):
    """PyTorch Dataset for coverage-based seed selection."""

    def __init__(
        self,
        seeds_path: pathlib.Path,
        target: List[str],
        max_len: Optional[int] = None,
        percentile_len: int = SeedFolderDataset.DEFAULT_PERCENTILE,
    ) -> None:
        """
        Initialize a PyTorch Dataset for coverage-based seeds.

        Args:
            seeds_path: Path to the seeds folder.
            target: Name of the fuzzing target in a callable format with its arguments.
            max_len: Optional limit for the maximum seed length.
            percentile_len: Optional length percentile to keep as maximum seed length (1-100).
        """
        super().__init__(
            seeds_path,
            target,
            bitmap_func=create_path_coverage_bitmap,
            max_len=max_len,
            percentile_len=percentile_len,
        )


def load_normalized_seeds(seed_list: List[Union[pathlib.Path, str]], max_len: int) -> np.ndarray:
    """
    Read a batch of seeds from files, normalize and convert to Numpy array.

    Args:
        seed_list: List of paths to the seeds to read.
        max_len: Max length of seed. Longer seeds are cut, shorter ones are
            padded with zeros to reach this length.

    Returns:
        Seed content of length `max_len`, normalized between 0 and 1.
    """
    seeds = read_seeds(seed_list)
    seeds_preproc = []
    
    # Truncate, pad, and normalize seeds
    for seed in seeds:
        truncated = seed[-max_len:]
        pad_length = max_len - len(truncated)
        if pad_length > 0:
            truncated = np.pad(truncated, (0, pad_length), mode="constant")
        normalized = truncated.astype("float32") / 255.0
        seeds_preproc.append(normalized)

    return np.array(seeds_preproc)

def read_seed(path: Union[pathlib.Path, str]) -> np.ndarray:
    """
    Read one seed from file name provided as input and return as Numpy arrays.

    Args:
        path: Path to seed to read.

    Returns:
        The content of the seed.
    """
    with open(path, "rb") as seed_file:
        seed = seed_file.read()
    return np.asarray(bytearray(seed), dtype="uint8")


def read_seeds(seed_list: List[Union[pathlib.Path, str]]) -> List[np.ndarray]:
    """
    Read multiple seeds from the list of paths provided as input and return
    them as Numpy arrays.

    Args:
        seed_list: List of paths to seeds.

    Returns:
        List of arrays of seeds.
    """
    return [read_seed(seed_file) for seed_file in seed_list]


def compute_input_size_from_seeds(
    seeds_path: pathlib.Path, percentile: int = 80, margin: int = 5
) -> int:# remember we ignore percentile_len when max_len is provided
    """
    Compute the maximum allowed size for seeds based on a heuristic:

      * Only seeds up to the provided percentile are considered. This is to remove
        the long tail of the seed distribution.
      * An extra `margin` percent is added to the percentile value in order to still allow
        for seed growth based on inserts, splicing, etc.

    Args:
        seeds_path: Path to the seeds folder.
        percentile: Value in the 1-100 range representing the cutting point for seeds length.
        margin: Percentage in 0-100 of the percentile value to add as margin.

    Returns:
        The maximum size of a seed.
    """
    perc = seed_len_percentile(seeds_path, percentile)
    return int((1.0 + 0.01 * margin) * perc)


def seed_len_percentile(seeds_path: pathlib.Path, percentile: int = 90) -> float:
    """
    Compute the `percentile` seed length for a given input folder.

    Args:
        seeds_path: Path to the seeds folder.
        percentile: Value in the 1-100 range representing the cutting point for seeds length.

    Returns:
        The length of the seed corresponding to `percentile`.
    """
    seed_lens = [get_seed_len(seed) for seed in seeds_path.glob("*") if seed.is_file()]
    return np.percentile(seed_lens, percentile)


def get_seed_len(path: Union[pathlib.Path, str]) -> int:
    """
    Return the length of a seed based on its path.

    Args:
        path: The full path of the seed.

    Returns:
        The lengths of the seed in bytes.
    """
    return os.path.getsize(path)
