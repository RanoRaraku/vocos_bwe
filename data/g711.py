from typing import Union

import numpy as np
import torch
from torchaudio.functional import highpass_biquad, lowpass_biquad
from torchaudio.transforms import MuLawDecoding, MuLawEncoding, Resample


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
        source_freq=24000,
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

    def g711encoder_defaults(self, **kwargs):
        """Return a dict containing target audio encoding parameters."""
        params = dict(
            {
                "source_freq": 24000,
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
        source_freq=24000,
        target_freq=8000,
        quantization_steps=256,
    ):
        super().__init__()
        self.source_freq = source_freq
        self.target_freq = target_freq
        self.quantization_steps = quantization_steps

        self.mulaw_dec = MuLawDecoding(self.quantization_steps)
        self.upsample = Resample(self.target_freq, self.source_freq)

    def g711decoder_defaults(self, **kwargs):
        """Return a dict containing target audio encoding parameters."""
        params = dict(
            {
                "source_freq": 24000,
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
