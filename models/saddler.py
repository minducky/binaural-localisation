"""Saddler (Phaselocknet ``simplified_IHC3000``) sound-localization CNN.

A single, arch-selectable port of the peripheral (`models/saddler_peripheral.py`)
+ perceptual (`models/saddler_perceptual.py`) pipeline from ``Phaselocknet_torch``.
Which of the 10 vendored CNN architectures (`arch01`-`arch10`, see
`models/saddler_configs/simplified_IHC3000/`) is used is a config choice
(`config["SADDLER"]["ARCH_DIR"]`), not a separate class per arch.

Unlike `Ducky`/`Yang`, the peripheral stages here have no learnable
parameters (fixed DSP: gammatone filterbank, IHC transduction/lowpass,
sigmoid auditory-nerve rate-level function, binomial spike sampling) and
include a random crop that stays stochastic even at eval time, matching the
original model rather than being made deterministic. The CNN backbone
(`perceptual_model`) trains from random initialization, like `Ducky`/`Yang`.
"""

import json
import os

import torch
import torch.nn as nn
import torchaudio

from models.saddler_perceptual import PerceptualModel
from models.saddler_peripheral import PeripheralModel

# `input_shape`/`config_random_slice` for the sound_localization/`signal`
# path aren't in config.json/arch.json - they're hardcoded in
# Phaselocknet_torch's phaselocknet_model.get_model() dispatch logic.
# Reproduced here as the default, overridable via config["SADDLER"]["RANDOM_SLICE"].
_DEFAULT_RANDOM_SLICE = {"size": [50, 10000], "buffer": [0, 1000]}


def load_saddler_arch(arch_dir: str) -> tuple[dict, list]:
    """Loads a vendored `(config.json, arch.json)` pair.

    Args:
        arch_dir: Path to a vendored arch directory (relative to the repo
            root or absolute), e.g.
            `'models/saddler_configs/simplified_IHC3000/arch02'`.

    Returns:
        `(config_model, architecture)`, the parsed contents of
        `config.json` and `arch.json` respectively.
    """
    with open(os.path.join(arch_dir, "config.json")) as f:
        config_model = json.load(f)
    with open(os.path.join(arch_dir, "arch.json")) as f:
        architecture = json.load(f)
    return config_model, architecture


def pad_or_trim_to_len(
    x: torch.Tensor, n: int, dim: int = -1, kwargs_pad: dict = None
) -> torch.Tensor:
    """Symmetrically pads or trims `x` to length `n` along `dim`."""
    if kwargs_pad is None:
        kwargs_pad = {}
    n_orig = int(x.shape[dim])
    if n_orig < n:
        n0 = (n - n_orig) // 2
        n1 = (n - n_orig) - n0
        pad = []
        for d in range(x.ndim):
            pad.extend([n0, n1] if d == dim else [0, 0])
        x = nn.functional.pad(x, pad, **kwargs_pad)
    if n_orig > n:
        n0 = (n_orig - n) // 2
        ind = [slice(None)] * x.ndim
        ind[dim] = slice(n0, n0 + n)
        x = x[tuple(ind)]
    return x


class SaddlerModel(nn.Module):
    """Saddler peripheral+perceptual CNN. Input: (B, T, 2) @ 44100 Hz stereo.

    Output: (B, num_classes) logits.
    """

    def __init__(self, config: dict):
        """Builds the peripheral and perceptual submodules from config.

        Args:
            config: Top-level experiment config. Reads `config["NUM_CLASSES"]`
                (auto-set by `experiment_manager.py`) and `config["SADDLER"]`
                (`ARCH_DIR`, optional `INPUT_SR`/`TARGET_INPUT_SAMPLES`/
                `RANDOM_SLICE`).
        """
        super().__init__()
        saddler_cfg = config["SADDLER"]
        config_model, architecture = load_saddler_arch(saddler_cfg["ARCH_DIR"])
        kwargs_cochlea = config_model["kwargs_cochlea"]

        self.input_sr = saddler_cfg.get("INPUT_SR", 44100)
        self.target_input_samples = saddler_cfg.get("TARGET_INPUT_SAMPLES", None)
        self.resampler = torchaudio.transforms.Resample(
            orig_freq=self.input_sr, new_freq=int(kwargs_cochlea["sr_input"])
        )

        random_slice_cfg = saddler_cfg.get("RANDOM_SLICE", _DEFAULT_RANDOM_SLICE)
        config_random_slice = {
            "size": random_slice_cfg["SIZE" if "SIZE" in random_slice_cfg else "size"],
            "buffer": random_slice_cfg[
                "BUFFER" if "BUFFER" in random_slice_cfg else "buffer"
            ],
        }

        kwargs_ihc_lowpass = dict(kwargs_cochlea["kwargs_fir_lowpass_filter_output"])
        assert kwargs_ihc_lowpass.pop("ihc_filter", True)
        self.peripheral_model = PeripheralModel(
            sr_input=kwargs_cochlea["sr_input"],
            sr_output=kwargs_cochlea["sr_output"],
            config_cochlear_filterbank=kwargs_cochlea["config_filterbank"],
            config_ihc_transduction=kwargs_cochlea["config_subband_processing"],
            config_ihc_lowpass_filter=kwargs_ihc_lowpass,
            config_anf_rate_level=kwargs_cochlea["kwargs_sigmoid_rate_level_function"],
            config_anf_spike_generator=kwargs_cochlea[
                "kwargs_spike_generator_binomial"
            ],
            config_random_slice=config_random_slice,
        )

        # Read NUM_CLASSES dynamically (like Ducky/Yang), not the vendored
        # JSON's own n_classes_dict value (always 504 there).
        num_classes = config["NUM_CLASSES"]
        heads = dict.fromkeys(config_model["n_classes_dict"], num_classes)
        assert len(heads) == 1, "SaddlerModel currently supports a single output head"
        self._head_key = next(iter(heads))

        with torch.no_grad():
            probe = torch.zeros(2, int(self.input_sr * 1.3), 2)
            probe_shape = self.peripheral_model(self._resample(probe)).shape
        self.perceptual_model = PerceptualModel(
            architecture=architecture, input_shape=probe_shape, heads=heads
        )

    def _resample(self, x: torch.Tensor) -> torch.Tensor:
        """Resamples binaural audio from `self.input_sr` to the arch's `sr_input`.

        Args:
            x: Audio, shape (batch, time, 2) at `self.input_sr` Hz.

        Returns:
            Resampled audio, shape (batch, time', 2).
        """
        x = torch.stack(
            [self.resampler(x[..., channel]) for channel in range(x.shape[-1])],
            axis=-1,
        )
        if self.target_input_samples is not None:
            x = pad_or_trim_to_len(x, n=self.target_input_samples, dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Runs the full pipeline: resample -> peripheral -> perceptual.

        Args:
            x: Stereo audio, shape (batch, time, 2) at `self.input_sr` Hz.

        Returns:
            Logits, shape (batch, num_classes).
        """
        x = self._resample(x)
        nervegram = self.peripheral_model(x)
        logits = self.perceptual_model(nervegram)
        return logits[self._head_key]
