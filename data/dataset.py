from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from .g711 import G711Decoder, G711Encoder

torch.set_num_threads(1)


@dataclass
class DataConfig:
    manifests: List[str]
    batch_size: int
    num_workers: int
    sampling_rate: int
    max_samples: int
    
class BWEDataset(Dataset):
    def __init__(self, cfg: DataConfig, train: bool):
        self.filelist = [f for m in cfg.manifests for f in m.files]
        self.sampling_rate = cfg.sampling_rate
        self.max_samples = cfg.max_samples
        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers
        self.train = train
        self.transcoding = [G711Encoder(), G711Decoder()]


    def __len__(self) -> int:
        return len(self.filelist)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        audio_path = self.filelist[index]
        y, sr = torchaudio.load(audio_path)

        if y.size(0) > 1:
            # mix to mono
            y = y.mean(dim=0, keepdim=True)

        gain = np.random.uniform(-1, -6) if self.train else -3
        y = torchaudio.functional.gain(y, gain)

        if sr != self.sampling_rate:
            y = torchaudio.functional.resample(
                y, orig_freq=sr, new_freq=self.sampling_rate
            )

        if y.size(-1) < self.max_samples:
            pad_length = self.max_samples - y.size(-1)
            padding_tensor = y.repeat(1, 1 + pad_length // y.size(-1))
            y = torch.cat((y, padding_tensor[:, :pad_length]), dim=1)
        elif self.train:
            start = np.random.randint(low=0, high=y.size(-1) - self.max_samples + 1)
            y = y[:, start : start + self.max_samples]
        else:
            # During validation, take always the first segment for determinism
            y = y[:, : self.max_samples]

        # Obtain narrow-band audio
        x = y.clone().detach()
        for tnfm in self.transcoding:
            x = tnfm(x)

        return x[0], y[0]

    def to_dataloder(self) -> DataLoader:
        return DataLoader(
            self,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.train,
            pin_memory=True,
        )
