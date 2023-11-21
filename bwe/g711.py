import argparse
from pathlib import Path
from typing import Union

import numpy as np
import torch
from icefall.utils import AttributeDict
from lhotse import CutSet, Fbank, FbankConfig, load_manifest
from lhotse.audio import RecordingSet
from lhotse.utils import EPSILON
from torchaudio.functional import highpass_biquad, lowpass_biquad
from torchaudio.transforms import MuLawDecoding, MuLawEncoding, Resample


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--audio_dir",
        type=str,
        default="D:/data/librispeech",
        help="""LibriSpeech dataset folder""",
    )
    parser.add_argument(
        "--train_subsets",
        type=list,
        default=["train-clean-100"],
        help="""Training subsets to process.
        All subsets will combine to single manifest.""",
    )
    parser.add_argument(
        "--dev_subsets",
        type=list,
        default=["dev-clean"],
        help="""Dev subsets to process.
        All subsets will combine to single manifest.""",
    )
    parser.add_argument(
        "--test_subsets",
        type=list,
        default=["test-clean"],
        help="""Test subsets to process.
        All subsets will combine to single manifest.""",
    )
    parser.add_argument(
        "--manifest_dir",
        type=str,
        default="D:/BandWidth_Ext/data/",
        help="""
        Folder to save manifests.
        """,
    )
    parser.add_argument(
        "--on-the-fly-feats",
        type=bool,
        default=False,
        help="""Compute features during training""",
    )
    parser.add_argument(
        "--feats_dir",
        type=str,
        default="D:/BandWidth_Ext/data",
        help="""
        Folder to save extracted feautures.
        Applied only if `--on-the-fly-feats` == False.
        """,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=1,
        help="""
        Number of workers for DataModule.
        """,
    )

    return parser


class G711Encoder(torch.nn.Module):
    """
    Pipeline to simulate G711 mu-law 8000kHz audio (i.e. Fisher, SWB, Callhome).

    Args:
    :source_freq:
    :target_freq:
    :quantization_steps:
    :lowpass_freq:
    :highpass_freq:
    """

    def __init__(
        self,
        source_freq=16000,
        target_freq=8000,
        quantization_steps=256,
        lowpass_freq=3400,
        highpass_freq=300,
    ):
        super().__init__()
        self.source_freq = source_freq
        self.target_freq = target_freq
        self.quantization_steps = quantization_steps
        self.lowpass_freq = lowpass_freq
        self.highpass_freq = highpass_freq

        self.downsample = Resample(self.source_freq, self.target_freq)
        self.mulaw_enc = MuLawEncoding(self.quantization_steps)


    def g711encoder_defaults(self, **kwargs) -> AttributeDict:
        """Return a dict containing target audio encoding parameters."""
        params = AttributeDict(
            {
                "source_freq": 16000,
                "target_freq": 8000,
                "quantization_steps": 256,
                "lowpass_freq": 3400,
                "highpass_freq": 300,
            }
        )
        params.update(kwargs)
        return params

    def forward(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Input is 16kHz PCM. Output is G711mu in [0, quantization_steps] range.
        Waveform expected values are in [-1,1] range.
        """
        # Maybe convert to torch.Tensor
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)

        # Maybe normalize to [-1,1]
        max_val = torch.max(torch.abs(waveform)).item()
        if max_val > 1:
            waveform /= max_val

        # pipeline
        e = self.downsample(waveform)
        e = lowpass_biquad(
            highpass_biquad(e, self.target_freq, self.highpass_freq),
            self.target_freq,
            self.lowpass_freq,
        )
        e = self.mulaw_enc(e)
        return e


class G711Decoder(torch.nn.Module):
    """
    Pipeline to simulate G711 mu-law 8000kHz audio (i.e. Fisher, SWB, Callhome).

    Args:
    :source_freq:
    :target_freq:
    :quantization_steps:
    :lowpass_freq:
    :highpass_freq:
    """

    def __init__(
        self,
        source_freq=16000,
        target_freq=8000,
        quantization_steps=256,
    ):
        super().__init__()
        self.source_freq = source_freq
        self.target_freq = target_freq
        self.quantization_steps = quantization_steps

        self.mulaw_dec = MuLawDecoding(self.quantization_steps)
        self.upsample = Resample(self.target_freq, self.source_freq)


    def g711decoder_defaults(self, **kwargs) -> AttributeDict:
        """Return a dict containing target audio encoding parameters."""
        params = AttributeDict(
            {
                "source_freq": 16000,
                "target_freq": 8000,
                "quantization_steps": 256,
            }
        )
        params.update(kwargs)
        return params

    def forward(self, waveform: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Input is 16kHz PCM signal and output is 16kHz signal. Waveform expected
        values are in [0, quantization_steps] range.
        """
        # Maybe convert to torch.Tensor
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform)

        # Maybe normalize to [0, quantization_steps]
        max_val = torch.max(torch.abs(waveform)).item()
        if max_val <= 1:
            waveform = (waveform * self.quantization_steps).to(torch.int32)

        # pipeline
        e = self.mulaw_dec(waveform)
        e = self.upsample(e)

        return e


def main():
    parser = get_parser()
    args = parser.parse_args()

    # Assume all subsets are lists
    # for subset in args.train_subsets + args.dev_subsets + args.test_subsets:
    for subset in args.dev_subsets + args.test_subsets:
        recs_path = Path(args.manifest_dir) / f"recs_{subset}.jsonl.gz"
        cuts_path = Path(args.manifest_dir) / f"cuts_{subset}.jsonl.gz"

        # Skip if recordings manifests exist
        if not cuts_path.is_file():
            if not recs_path.is_file():
                audio_dir = Path(args.audio_dir) / subset
                recs = RecordingSet.from_dir(audio_dir, "*.flac")
                recs.to_file(Path(args.manifest_dir) / f"recs_{subset}.jsonl.gz")
            else:
                recs = RecordingSet.from_file(recs_path)
            cuts = CutSet.from_manifests(recs)
        else:
            cuts = RecordingSet.from_file(cuts_path)

        # Extract feats
        if not args.on_the_fly_feats:
            extractor = Fbank(FbankConfig(**fbank_params()))
            cuts = cuts.compute_and_store_features(
                extractor=extractor,
                storage_path=Path(args.feats_dir) / subset,
                num_jobs=args.num_workers,
            )

        # Save cuts
        cuts.to_file(Path(args.manifest_dir) / f"cuts_{subset}.jsonl.gz")


if __name__ == "__main__":
    main()
