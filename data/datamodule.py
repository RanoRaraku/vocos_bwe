import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from icefall.utils import AttributeDict
from lhotse import CutSet, load_manifest
from lhotse.dataset import BucketingSampler, SimpleCutSampler
from lhotse.utils import fix_random_seed, ifnone
from lhotse.workarounds import Hdf5MemoryIssueFix
from torch.utils.data.dataloader import DataLoader

from vocos.feature_extractors import MelSpectrogramFeatures

class BWEDataset(torch.utils.data.Dataset):
    """
    The PyTorch Dataset for the BandWidth Extension.

    This dataset expects to be queried with lists of cut IDs,
    for which it loads features and automatically collates/batches them.

    To use it with a PyTorch DataLoader, set ``batch_size=None``
    and provide a :class:`SimpleCutSampler` sampler.

    Each batch in this dataset is a dict of:
        {
            'input_feats': float tensor with shape determined by :attr:`targets_strategy`:
                        - features: (B, F, T)
            'target_feats': float tensor with shape determined by :attr:`targets_strategy`:
                        - features: (B, F, T)
        }

    Dimension symbols legend:
    * ``B`` - batch size (number of Cuts)
    * ``T`` - number of frames of the longest Cut
    * ``F`` - number of features
    """

    def __init__(
        self,
        return_cuts: bool = False,
        cut_transforms: List[Callable[[CutSet], CutSet]] = [],
        inputs_transforms: List[Callable[[torch.Tensor], torch.Tensor]] = [],
        inputs_extractor: Callable = MelSpectrogramFeatures,
        targets_extractor: Callable = MelSpectrogramFeatures,
    ):
        """
        :param return_cuts: When ``True``, will additionally return a "cut" field in each batch with the Cut
            objects used to create that batch.
        :param cut_transforms: A list of transforms to be applied on each sampled batch,
            before converting cuts to an input representation (audio/features).
            Examples: cut concatenation, noise cuts mixing, etc.
        :param targets_extractor: Converts cuts into a collated batch of audio/features.
            By default, reads pre-computed features from disk. This is 16khz audio
            we learn to convert to.
        :param inputs_transforms: A list of transforms to be applied on each sampled batch,
            after the cuts are converted to audio. Default is G711_encoder -> G711_decoder.
        :param inputs_extractor: Converts cuts into a collated batch of audio/features.
            By default, compute them on the fly. This is upsampled PCM telephone audio
            we learn to convert from.
        """
        super().__init__()
        # Initialize the fields
        self.return_cuts = return_cuts
        self.targets_extractor = targets_extractor()
        self.inputs_extractor = inputs_extractor()
        self.cut_transforms = cut_transforms if cut_transforms else []
        self.inputs_transforms = inputs_transforms if inputs_transforms else [G711Encoder(), G711Decoder()]


        # This attribute is a workaround to constantly growing HDF5 memory
        # throughout the epoch. It regularly closes open file handles to
        # reset the internal HDF5 caches.
        self.hdf5_fix = Hdf5MemoryIssueFix(reset_interval=100)

    def __getitem__(self, cuts: CutSet) -> Dict[str, Union[torch.Tensor, List[str]]]:
        """
        Return a new batch, with the batch size automatically determined using the
        constraints of max_frames and max_cuts. Tensor with batched feature matrices
        shape (B, F, T)

        :target_feats: features from original audio
        :input_feats: features for simulated telephone speech
        """
        self.hdf5_fix.update()

        # Sort the cuts by duration so that the first one determines the batch time dimensions.
        cuts = cuts.sort_by_duration(ascending=False)

        # Optional CutSet transforms - e.g. padding, or speed perturbation
        for tnfm in self.cut_transforms:
            cuts = tnfm(cuts)

        # Sort the cuts again after transforms
        cuts = cuts.sort_by_duration(ascending=False)
        # padd and extract audios
        audio, audio_lens = cuts.load_audio(collate=True)
        audio_target = torch.from_numpy(audio_target)

        # Generate targets 1st
        #target_feats = self.targets_extractor(audios)

        # Apply codecs and generate inputs
        audio_input = audio_target
        for tnfm in self.inputs_transforms:
            audio_input = tnfm(audio_input)
        #input_feats = self.inputs_extractor(audios)

        # Collate batch
        batch = {
            "audio_input": audio_input,
            "audio_target": audio_target,
        }

        return batch


class _SeedWorkers:
    def __init__(self, seed: int):
        self.seed = seed

    def __call__(self, worker_id: int):
        fix_random_seed(self.seed + worker_id)


class LibriTTSDataModule:
    """
    DataModule for BWE experiments.
    It assumes there is always one train and valid dataloader,
    but there can be multiple test dataloaders (e.g. LibriSpeech test-clean
    and test-other).

    It contains all the common data pipeline modules used in experiments, e.g.:
    - dynamic batch size,
    - bucketing samplers,
    - cut concatenation,
    - augmentation,
    - on-the-fly feature extraction

    This class should be derived for specific corpora used in BWE tasks.
    """

    def __init__(self, **kwargs):
        self.args = self.get_arguments(**kwargs)

        # Note - train_cuts are set to dev subset for fast development
        self.train_cuts = self.dev_clean_cuts()
        self.valid_cuts = self.dev_clean_cuts()
        self.test_cuts = self.test_clean_cuts()

    @classmethod
    def get_arguments(cls, **kwargs) -> AttributeDict:
        args = AttributeDict(
            {
                "on_the_fly_feats": True,
                "manifest_dir": Path("d:/data/libritts/manifests/"),
                "batch_max_duration": 200,
                "bucketing_sampler": True,
                "num_buckets": 16,
                "concatenate_cuts": False,
                "duration_factor": 1.0,
                "gap": 1.0,
                "shuffle": True,
                "return_cuts": False,
                "num_workers": 3
            }
        )
        args.update(kwargs)

        return args        

    def train_dataloader(
        self,
        sampler_state_dict: Optional[Dict[str, Any]] = None,
    ) -> DataLoader:
        """
        Args:
          sampler_state_dict:
            The state dict for the training sampler.
        """
        logging.info("About to create train dataset")
        train = BWEDataset()

        if self.args.bucketing_sampler:
            logging.info("Using BucketingSampler.")
            train_sampler = BucketingSampler(
                self.train_cuts,
                max_duration=self.args.batch_max_duration,
                shuffle=self.args.shuffle,
                num_buckets=self.args.num_buckets,
                # bucket_method="equal_duration",
                drop_last=True,
            )
        else:
            logging.info("Using SimpleCutSampler.")
            train_sampler = SimpleCutSampler(
                self.train_cuts,
                max_duration=self.args.batch_max_duration,
                shuffle=self.args.shuffle,
            )
        logging.info("About to create train dataloader")

        if sampler_state_dict is not None:
            logging.info("Loading sampler state dict")
            train_sampler.load_state_dict(sampler_state_dict)

        # 'seed' is derived from the current random state, which will have
        # previously been set in the main process.
        seed = torch.randint(0, 100000, ()).item()
        worker_init_fn = _SeedWorkers(seed)

        train_dl = DataLoader(
            train,
            sampler=train_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
            worker_init_fn=worker_init_fn,
        )

        return train_dl

    def valid_dataloader(self) -> DataLoader:
        logging.info("About to create dev dataset")

        validate = BWEDataset()
        valid_sampler = BucketingSampler(
            self.valid_cuts,
            max_duration=self.args.batch_max_duration,
            shuffle=False,
        )
        logging.info("About to create dev dataloader")
        valid_dl = DataLoader(
            validate,
            sampler=valid_sampler,
            batch_size=None,
            num_workers=self.args.num_workers,
            persistent_workers=False,
        )

        return valid_dl

    def test_dataloader(self, test_cuts: CutSet) -> DataLoader:
        logging.debug("About to create test dataset")
        test = BWEDataset(return_cuts=self.args.return_cuts)
        sampler = BucketingSampler(
            test_cuts, max_duration=self.args.batch_max_duration, shuffle=False
        )
        logging.debug("About to create test dataloader")
        test_dl = DataLoader(
            test,
            batch_size=None,
            sampler=sampler,
            num_workers=self.args.num_workers,
        )
        return test_dl

    @lru_cache()
    def train_clean_100_cuts(self) -> CutSet:
        logging.info("About to get train-clean-100 cuts")
        return load_manifest(
            Path(self.args.manifest_dir) / "cuts_train-clean-100.jsonl.gz"
        )

    @lru_cache()
    def train_clean_360_cuts(self) -> CutSet:
        logging.info("About to get train-clean-360 cuts")
        return load_manifest(
            Path(self.args.manifest_dir) / "cuts_train-clean-360.jsonl.gz"
        )

    @lru_cache()
    def train_other_500_cuts(self) -> CutSet:
        logging.info("About to get train-other-500 cuts")
        return load_manifest(
            Path(self.args.manifest_dir) / "cuts_train-other-500.jsonl.gz"
        )

    @lru_cache()
    def dev_clean_cuts(self) -> CutSet:
        logging.info("About to get dev-clean cuts")
        return load_manifest(Path(self.args.manifest_dir) / "cuts_dev-clean.jsonl.gz")

    @lru_cache()
    def dev_other_cuts(self) -> CutSet:
        logging.info("About to get dev-other cuts")
        return load_manifest(Path(self.args.manifest_dir) / "cuts_dev-other.jsonl.gz")

    @lru_cache()
    def test_clean_cuts(self) -> CutSet:
        logging.info("About to get test-clean cuts")
        return load_manifest(Path(self.args.manifest_dir) / "cuts_test-clean.jsonl.gz")

    @lru_cache()
    def test_other_cuts(self) -> CutSet:
        logging.info("About to get test-other cuts")
        return load_manifest(Path(self.args.manifest_dir) / "cuts_test-other.jsonl.gz")
