import os
from typing import Optional, Tuple, List, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from ddpm.utils.utils import normalize_delta


class FieldDataset(Dataset):
    """
    2.5D dataset:
    each sample is a stack of consecutive slabs from the same
    mass / realization / axis.

    Expected directory structure:
        <split>/<mass>/<realization>/fR_p_r000_ax0_slab0000.npy

    Returns
    -------
    x : torch.Tensor
        Shape (2*half + 1, H, W), float32
    y : torch.Tensor
        Class label, dtype long
    names : list[str]
        Filenames used in THIS sample only
    """

    def __init__(
        self,
        directory: str,
        norm_stats_path: str,
        slab_size: int,
        half: int = 2,
        realization_limit: Optional[int] = None,
        use_percentile_clip: bool = False,  # clip z to the 0.1-99.9 percentiles of the training set
    ):
        self.directory = directory
        self.norm_stats_path = norm_stats_path
        self.slab_size = int(slab_size)
        self.half = int(half)
        self.realization_limit = realization_limit
        self.use_percentile_clip = use_percentile_clip

        self.label_map = {
            "fR_p": 0,
            "fR_pp": 1,
            "fR_ppp": 2,
            "fR_pppp": 3,
        }

        # ----------------------------
        # normalization stats
        # ----------------------------
        stats = np.load(norm_stats_path)
        self.mu = float(stats["MU"])
        self.sigma = float(stats["SIGMA"])
        self.method = str(stats["METHOD"])
        self.alpha = float(stats["ALPHA"])
        self.eps = float(stats["EPS"])

        self.clip_min = None
        self.clip_max = None

        if "P001_Z" in stats and "P999_Z" in stats:
            self.clip_min = float(stats["P001_Z"])
            self.clip_max = float(stats["P999_Z"])
        elif "P001" in stats and "P999" in stats:
            self.clip_min = float(stats["P001"])
            self.clip_max = float(stats["P999"])

        if self.sigma <= 0:
            raise ValueError(f"SIGMA must be > 0. Got {self.sigma}")

        if "SLAB_SIZE" in stats:
            stats_slab_size = int(stats["SLAB_SIZE"])

            if self.slab_size > stats_slab_size:
                raise ValueError(
                    f"Requested slab_size={self.slab_size}, but stats file was computed "
                    f"for SLAB_SIZE={stats_slab_size}. Cannot crop larger than native size."
                )

            self.native_slab_size = stats_slab_size
        else:
            self.native_slab_size = None

        # ----------------------------
        # discover files by known structure
        # ----------------------------
        self.groups: Dict[Tuple[str, int, int], List[Tuple[int, str]]] = {}
        self.samples: List[Tuple[str, int, int, int, str]] = []
        self.group_index_map: Dict[Tuple[Tuple[str, int, int], int, str], int] = {}

        for mass in self.label_map:
            mass_dir = os.path.join(directory, mass)
            if not os.path.isdir(mass_dir):
                continue

            # realizations that actually exist in this split
            real_dirs = sorted(
                d for d in os.listdir(mass_dir)
                if os.path.isdir(os.path.join(mass_dir, d)) and d.isdigit()
            )
            real_list = [int(d) for d in real_dirs]

            if realization_limit is not None:
                real_list = real_list[:realization_limit]

            for real_idx in real_list:
                real_dir = os.path.join(mass_dir, str(real_idx))
                if not os.path.isdir(real_dir):
                    continue

                for f in sorted(os.listdir(real_dir)):
                    if not f.endswith(".npy"):
                        continue

                    full_path = os.path.join(real_dir, f)

                    try:
                        mass_f, real_f, axis_idx, slab_idx = self._parse_filename(f)
                    except ValueError:
                        continue

                    if mass_f != mass or real_f != real_idx:
                        continue

                    key = (mass, real_idx, axis_idx)
                    self.groups.setdefault(key, []).append((slab_idx, full_path))
                    self.samples.append((mass, real_idx, axis_idx, slab_idx, full_path))

        if len(self.samples) == 0:
            raise RuntimeError(f"No .npy slabs found under directory: {directory}")

        # sort groups and build lookup
        for key in self.groups:
            self.groups[key].sort(key=lambda x: x[0])
            for i, (slab_idx, fpath) in enumerate(self.groups[key]):
                self.group_index_map[(key, slab_idx, fpath)] = i

        # deterministic order
        self.samples.sort(key=lambda x: (x[0], x[1], x[2], x[3], x[4]))



    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _parse_filename(filename: str) -> Tuple[str, int, int, int]:
        """
        Parse:
            fR_p_r012_ax0_slab0416.npy
        """
        base = filename.replace(".npy", "")
        parts = base.split("_")

        if len(parts) == 5 and parts[0] == "fR":
            mass = parts[0] + "_" + parts[1]
            real_idx = int(parts[2].replace("r", ""))
            axis_idx = int(parts[3].replace("ax", ""))
            slab_idx = int(parts[4].replace("slab", ""))
            return mass, real_idx, axis_idx, slab_idx

        raise ValueError(f"Bad filename format: {filename}")

    def _load_and_normalize(self, fpath: str) -> np.ndarray:
        d = np.load(fpath).astype(np.float32)

        H, W = d.shape

        if H < self.slab_size or W < self.slab_size:
            raise ValueError(
                f"Input slab too small: {d.shape} for requested crop {self.slab_size}"
            )

        # ----------------------------------------
        # random crop
        # ----------------------------------------
        if H != self.slab_size or W != self.slab_size:
            y0 = np.random.randint(0, H - self.slab_size + 1)
            x0 = np.random.randint(0, W - self.slab_size + 1)

            d = d[
                y0:y0 + self.slab_size,
                x0:x0 + self.slab_size,
            ]

        y = normalize_delta(
            d,
            mu=self.mu,
            sigma=self.sigma,
            method=self.method,
            alpha=self.alpha,
            eps=self.eps,
        )

        if self.use_percentile_clip and self.clip_min is not None and self.clip_max is not None:
            y = np.clip(y, self.clip_min, self.clip_max)

        return y

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        mass, real_idx, axis_idx, slab_idx, fpath = self.samples[index]

        # label
        label = self.label_map[mass]
        label_tensor = torch.tensor(label, dtype=torch.long)

        # all slabs from same mass / realization / axis
        key = (mass, real_idx, axis_idx)
        same_files = self.groups[key]
        n = len(same_files)

        central_i = self.group_index_map[(key, slab_idx, fpath)]

        # consecutive neighbors with periodic boundary (fields are periodic by construction)
        neighbors = []
        for j in range(central_i - self.half, central_i + self.half + 1):
            j_wrapped = j % n
            neighbors.append(same_files[j_wrapped])

        arrs = []
        names = []
        for _, p in neighbors:
            arrs.append(self._load_and_normalize(p))
            names.append(os.path.basename(p))

        x = torch.from_numpy(np.stack(arrs, axis=0).astype(np.float32))
        return x, label_tensor, names


def create_data_loaders(
    directory: str,
    norm_stats_path: str,
    slab_size: int,
    batch_size: int,
    shuffle: bool = True,
    dataset_limit: Optional[int] = None,
    half: int = 2,
    realization_limit: Optional[int] = None,
    use_percentile_clip: bool = False,
) -> DataLoader:
    dataset = FieldDataset(
        directory=directory,
        norm_stats_path=norm_stats_path,
        slab_size=slab_size,
        half=half,
        realization_limit=realization_limit,
        use_percentile_clip=use_percentile_clip,
    )

    if dataset_limit is not None:
        indices = list(range(min(dataset_limit, len(dataset))))
        dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
    )
    return loader