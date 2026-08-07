"""Saddler/Phaselocknet perceptual model: a CNN built dynamically from JSON.

Faithful port of the ``perceptual_model.py`` module from
``Phaselocknet_torch``. `PerceptualModel` builds a `torch.nn.Sequential`
stack from an ``arch.json`` layer-description list (conv/pool/relu/
batchnorm/dense/dropout, with TF-style "same"/"valid"/"valid_time" padding
semantics reimplemented in plain PyTorch - despite the ``layer_type``
strings looking TF-flavored, e.g. ``"tf.nn.relu"``, no TensorFlow is
involved). The only functional change from the source: `torch.nn.SyncBatchNorm`
is replaced with rank-specific `torch.nn.BatchNorm1d`/`BatchNorm2d`, since
this repo trains single-GPU without `torch.distributed` process groups (and
plain `BatchNormNd`, unlike `SyncBatchNorm`, requires an exact input rank).
"""

import collections

import numpy as np
import scipy.signal
import torch
import torch.nn as nn

from models.saddler_peripheral import FIRFilterbank


class PerceptualModel(nn.Module):
    """CNN dynamically constructed from an `arch.json`-style layer list.

    Each layer description is applied in order to build `self.body`, until
    a layer with no `"args"`/`"name"` (a `"branch"` or `"fc_top"` marker)
    splits the graph into one `nn.Sequential` head per key in `heads`,
    each ending in a `Linear(..., heads[key])` output layer.
    """

    def __init__(
        self,
        architecture: list = None,
        input_shape: list = None,
        heads: dict = None,
    ):
        """Constructs the model graph from an architecture description.

        Args:
            architecture: Layer descriptions loaded from `arch.json`.
            input_shape: Shape probe used to infer per-layer channel counts
                while building the graph (a dummy all-zeros tensor of this
                shape is run through the body during construction).
            heads: Maps output head name to number of output classes.
        """
        super().__init__()
        if architecture is None:
            architecture = []
        if input_shape is None:
            input_shape = [2, 6, 50, 20000]
        if heads is None:
            heads = {"label": 1}
        self.input_shape = input_shape
        self.body = collections.OrderedDict()
        self.heads = heads
        self.head = {k: collections.OrderedDict() for k in self.heads}
        self.construct_model(architecture)

    def get_layer_from_description(self, d: dict, x: torch.Tensor) -> nn.Module:
        """Builds one `nn.Module` layer from a single `arch.json` entry.

        Args:
            d: One layer description, with `"layer_type"` and `"args"` keys.
            x: Current shape-probe tensor (used to infer `in_channels`/
                `in_features` for conv/dense layers).

        Returns:
            The constructed layer, or `None` for a body/head split marker
            (`"branch"`/`"fc_top"` layer types).
        """
        layer_type = d["layer_type"].lower()
        if "conv" in layer_type:
            layer = CustomPaddedConv2d(
                in_channels=x.shape[1],
                out_channels=d["args"]["filters"],
                kernel_size=d["args"]["kernel_size"],
                stride=d["args"]["strides"],
                padding=d["args"]["padding"],
                dilation=d["args"].get("dilation", 1),
                groups=d["args"].get("groups", 1),
                bias=d["args"].get("bias", True),
                padding_mode="zeros",
            )
        elif "dense" in layer_type:
            layer = nn.Linear(
                in_features=x.shape[1],
                out_features=d["args"]["units"],
                bias=d["args"].get("use_bias", True),
            )
        elif "dropout" in layer_type:
            layer = nn.Dropout(p=d["args"]["rate"], inplace=False)
        elif "flatten" in layer_type:
            layer = CustomFlatten(start_dim=1, end_dim=-1, permute_dims=(0, 2, 3, 1))
        elif "maxpool" in layer_type:
            layer = nn.MaxPool2d(
                kernel_size=d["args"]["pool_size"],
                stride=d["args"]["strides"],
                padding=0,
            )
        elif "hpool" in layer_type:
            layer = HanningPooling(
                stride=d["args"]["strides"],
                kernel_size=d["args"]["pool_size"],
                padding=d["args"]["padding"],
                sqrt_window=d["args"].get("sqrt_window", False),
                normalize=d["args"].get("normalize", False),
            )
        elif "batchnorm" in layer_type.replace("_", ""):
            # SyncBatchNorm (source) -> BatchNorm1d/2d: this repo trains
            # single-GPU without torch.distributed process groups. Unlike
            # SyncBatchNorm (which accepts any input rank >= 2), plain
            # BatchNormNd is rank-specific, so pick 1d/2d by the current
            # tensor's rank (4D mid-conv-stack, 2D post-flatten).
            batchnorm_cls = {2: nn.BatchNorm1d, 4: nn.BatchNorm2d}[x.ndim]
            layer = batchnorm_cls(
                num_features=x.shape[1] if d["args"].get("axis", -1) == -1 else None,
                eps=d["args"].get("eps", 1e-05),
                momentum=d["args"].get("momentum", 0.1),
                affine=d["args"].get("scale", True),
            )
        elif "layernorm" in layer_type.replace("_", ""):
            layer = CustomNorm(
                input_shape=x.shape,
                dim_affine=1 if d["args"].get("scale", True) else None,
                dim_norm=1 if d["args"]["axis"] == -1 else d["args"]["axis"],
                correction=1,
                eps=d["args"].get("eps", 1e-05),
            )
        elif "permute" in layer_type:
            layer = Permute(dims=d["args"]["dims"])
        elif "unsqueeze" in layer_type:
            layer = Unsqueeze(dim=d["args"]["dim"])
        elif "expandlastdimension" in layer_type:
            layer = ExpandLastDimension(num_dims=d["args"]["num_dims"])
        elif "resample" in layer_type:
            layer = FIRResample(
                sr_input=d["args"]["sr_input"],
                sr_output=d["args"]["sr_output"],
                **d["args"]["kwargs_fir_lowpass_filter"],
            )
        elif "relu" in layer_type:
            layer = nn.ReLU(inplace=False)
        elif ("branch" in layer_type) or ("fc_top" in layer_type):
            layer = None
        else:
            print(f"[WARNING] {layer_type=} --> torch.nn.Identity")
            layer = nn.Identity()
        return layer

    def construct_model(self, architecture: list) -> None:
        """Builds `self.body` and `self.head` from an architecture description."""
        x = torch.zeros(self.input_shape)
        is_body_layer = True
        for d in architecture:
            if is_body_layer:
                layer = self.get_layer_from_description(d, x)
            else:
                layer = {
                    k: self.get_layer_from_description(d, x[k]) for k in self.heads
                }
            if (layer is None) or (
                isinstance(layer, dict) and list(layer.values())[0] is None
            ):
                is_body_layer = False
                if not isinstance(x, dict):
                    x = {k: torch.clone(x) for k in self.heads}
            else:
                if is_body_layer:
                    self.body[d["args"]["name"]] = layer
                    x = layer(x)
                else:
                    for k in self.heads:
                        self.head[k][d["args"]["name"]] = layer[k]
                        x[k] = layer[k](x[k])
        self.body = nn.Sequential(self.body)
        if not isinstance(x, dict):
            x = {k: torch.clone(x) for k in self.heads}
        for k in self.heads:
            self.head[k]["fc_output"] = nn.Linear(
                in_features=x[k].shape[1],
                out_features=self.heads[k],
                bias=True,
            )
            self.head[k] = nn.Sequential(self.head[k])
        self.head = nn.ModuleDict(self.head)

    def forward(self, x: torch.Tensor) -> dict:
        """Runs the CNN body then each output head.

        Returns:
            Dict mapping head name to logits, shape (batch, heads[name]).
        """
        x = self.body(x)
        logits = {k: self.head[k](x) for k in self.heads}
        return logits


# %% Padding helpers (TF-style "same"/"valid" padding math, plain PyTorch)


def calculate_same_pad(input_dim: int, kernel_dim: int, stride: int) -> int:
    """Computes TF "SAME"-style total padding for one spatial dimension."""
    pad = (np.ceil(input_dim / stride) - 1) * stride + (kernel_dim - 1) + 1 - input_dim
    return int(max(pad, 0))


def custom_conv_pad(
    x: torch.Tensor, pad, weight: torch.Tensor = None, stride: list = None, **kwargs
) -> torch.Tensor:
    """Pads a (batch, channel, freq, time) tensor per a TF-style padding mode.

    Args:
        x: Input, shape (batch, channel, freq, time).
        pad: Either an explicit `(pad_t_left, pad_t_right, pad_f_left,
            pad_f_right)` tuple, or one of the strings `"same"`,
            `"same_freq"`/`"valid_time"`, `"same_time"`/`"valid_freq"`,
            `"valid"`.
        weight: Conv weight (used to read the kernel size for `"same"`-style
            modes).
        stride: `(stride_freq, stride_time)`.
        kwargs: Extra kwargs forwarded to `torch.nn.functional.pad`.

    Returns:
        Padded tensor.
    """
    msg = f"Expected input shape [batch, channel, freq, time]: received {x.shape=}"
    assert x.ndim == 4, msg
    msg = f"Expected tuple or integers or a string: received {pad=}"
    assert isinstance(pad, (tuple, str)), msg
    if isinstance(pad, str):
        if pad.lower() == "same":
            pad_f = calculate_same_pad(x.shape[-2], weight.shape[-2], stride[-2])
            pad_t = calculate_same_pad(x.shape[-1], weight.shape[-1], stride[-1])
        elif pad.lower() in ["same_freq", "valid_time"]:
            pad_f = calculate_same_pad(x.shape[-2], weight.shape[-2], stride[-2])
            pad_t = 0
        elif pad.lower() in ["same_time", "valid_freq"]:
            pad_f = 0
            pad_t = calculate_same_pad(x.shape[-1], weight.shape[-1], stride[-1])
        elif pad.lower() == "valid":
            pad_f = 0
            pad_t = 0
        else:
            raise ValueError(f"Mode `{pad=}` is not recognized")
        pad = (pad_t // 2, pad_t - pad_t // 2, pad_f // 2, pad_f - pad_f // 2)
    return nn.functional.pad(x, pad, **kwargs)


class ChannelwiseConv2d(nn.Module):
    """Depthwise 2D convolution with a single fixed (non-learnable) kernel."""

    def __init__(
        self,
        kernel: np.ndarray,
        pad: tuple = (0, 0),
        stride: tuple = (1, 1),
        dtype: torch.dtype = torch.float32,
    ):
        """Initializes the fixed depthwise conv.

        Args:
            kernel: Convolution kernel, shape (freq, time).
            pad: Passed through to `custom_conv_pad` as the `pad` argument.
            stride: `(stride_freq, stride_time)`.
            dtype: Dtype for the kernel buffer.
        """
        super().__init__()
        assert kernel.ndim == 2, "Expected kernel with shape [freq, time]"
        self.register_buffer(
            "weight", torch.tensor(kernel[None, None, :, :], dtype=dtype)
        )
        self.pad = pad
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the fixed kernel to every channel of `x` independently."""
        y = custom_conv_pad(
            x,
            pad=self.pad,
            weight=self.weight,
            stride=self.stride,
            mode="constant",
            value=0,
        )
        y = y.view(-1, 1, *y.shape[-2:])
        y = nn.functional.conv2d(
            input=y,
            weight=self.weight,
            bias=None,
            stride=self.stride,
            padding="valid",
            dilation=1,
            groups=1,
        )
        y = y.view(*x.shape[:-2], *y.shape[-2:])
        return y


class HanningPooling(ChannelwiseConv2d):
    """Weighted-average pooling using a (separable) Hanning-window kernel."""

    def __init__(
        self,
        stride: list = None,
        kernel_size: list = None,
        padding="same",
        sqrt_window: bool = False,
        normalize: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        """Initializes the Hanning-window pooling kernel.

        Args:
            stride: `(stride_freq, stride_time)`.
            kernel_size: `(kernel_freq, kernel_time)`.
            padding: Passed through to `custom_conv_pad`.
            sqrt_window: If True, uses the square root of the Hanning window.
            normalize: If True, normalizes the kernel to sum to 1.
            dtype: Dtype for the kernel.
        """
        if stride is None:
            stride = [1, 1]
        if kernel_size is None:
            kernel_size = [1, 1]
        kernel = torch.ones(kernel_size, dtype=dtype)
        for dim, m in enumerate(kernel_size):
            shape = [-1 if _ == dim else 1 for _ in range(len(kernel_size))]
            kernel = kernel * torch.signal.windows.hann(
                m, sym=True, dtype=dtype
            ).reshape(shape)
        if sqrt_window:
            kernel = torch.sqrt(kernel)
        if normalize:
            kernel = kernel / torch.sum(kernel)
        super().__init__(kernel.numpy(), pad=padding, stride=stride, dtype=dtype)


class CustomPaddedConv2d(nn.Conv2d):
    """`nn.Conv2d` with TF-style string padding modes (`"same"`/`"valid_time"`/...)."""

    def __init__(self, *args, **kwargs):
        """See `torch.nn.Conv2d`; `padding` may additionally be a TF-style string."""
        self.pad = kwargs.get("padding", 0)
        if isinstance(self.pad, int):
            self.pad = (self.pad, self.pad)
        if isinstance(self.pad, str):
            kwargs["padding"] = 0
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pads per `self.pad` (TF-style) then convolves."""
        y = custom_conv_pad(
            x,
            pad=self.pad,
            weight=self.weight,
            stride=self.stride,
            mode="constant" if self.padding_mode == "zeros" else self.padding_mode,
            value=0,
        )
        y = nn.functional.conv2d(
            input=y,
            weight=self.weight,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        return y


class CustomFlatten(nn.Module):
    """Optional permute followed by `torch.flatten`."""

    def __init__(
        self, start_dim: int = 0, end_dim: int = -1, permute_dims: tuple = None
    ):
        """Initializes the flatten layer.

        Args:
            start_dim: First dim to flatten (see `torch.flatten`).
            end_dim: Last dim to flatten (see `torch.flatten`).
            permute_dims: If given, `torch.permute(x, dims=permute_dims)` is
                applied before flattening (e.g. channels-first ->
                channels-last, to match the TF reference's flatten order).
        """
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim
        self.permute_dims = permute_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Permutes (if configured) then flattens `x`."""
        if self.permute_dims is not None:
            x = torch.permute(x, dims=self.permute_dims)
        return torch.flatten(x, start_dim=self.start_dim, end_dim=self.end_dim)


class CustomNorm(nn.Module):
    """Generic normalization layer (e.g. layer norm) over an arbitrary dim."""

    def __init__(
        self,
        input_shape: list = None,
        dim_affine: int = None,
        dim_norm: int = None,
        correction: int = 1,
        eps: float = 1e-05,
        dtype: torch.dtype = torch.float32,
    ):
        """Initializes the normalization layer.

        Args:
            input_shape: Shape of the input, required if `dim_affine` is set.
            dim_affine: Dim to apply a learnable per-channel scale/bias over,
                or None to skip the affine transform.
            dim_norm: Dim(s) to normalize (mean/variance) over.
            correction: Bessel's correction for `torch.var_mean`.
            eps: Numerical-stability epsilon added to the variance.
            dtype: Dtype for the affine parameters.
        """
        super().__init__()
        if input_shape is None:
            input_shape = [None, None, None, None]
        self.input_shape = input_shape
        self.dim_affine = dim_affine
        self.dim_norm = dim_norm
        self.correction = correction
        self.eps = eps
        self.dtype = dtype
        if self.dim_affine is not None:
            msg = "`input_shape` is required when `dim_affine` is not None"
            assert self.input_shape is not None, msg
            size = input_shape[self.dim_affine]
            self.shape = [1 for _ in self.input_shape]
            self.shape[self.dim_affine] = input_shape[self.dim_affine]
            self.weight = nn.parameter.Parameter(
                data=torch.squeeze(torch.ones(size, dtype=self.dtype)),
                requires_grad=True,
            )
            self.bias = nn.parameter.Parameter(
                data=torch.squeeze(torch.zeros(size, dtype=self.dtype)),
                requires_grad=True,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes `x` over `self.dim_norm`, then applies the affine transform."""
        x_var, x_mean = torch.var_mean(
            x, dim=self.dim_norm, correction=self.correction, keepdim=True
        )
        y = (x - x_mean) / torch.sqrt(x_var + self.eps)
        if self.dim_affine is not None:
            w = self.weight.view(self.shape)
            b = self.bias.view(self.shape)
            y = (y * w) + b
        return y


class Permute(nn.Module):
    """Permutes all dims after the batch dim."""

    def __init__(self, dims: list = None):
        """Initializes the permutation (batch dim is always kept first)."""
        super().__init__()
        self.dims = [0] + list(dims)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies `torch.permute(x, dims=self.dims)`."""
        return torch.permute(x, dims=self.dims)


class Unsqueeze(nn.Module):
    """Inserts a size-1 dim at `self.dim`."""

    def __init__(self, dim: int = None):
        """Initializes the unsqueeze layer."""
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies `torch.unsqueeze(x, dim=self.dim)`."""
        return torch.unsqueeze(x, dim=self.dim)


class ExpandLastDimension(nn.Module):
    """Pads `x`'s shape with trailing size-1 dims up to `self.num_dims` dims."""

    def __init__(self, num_dims: int = None):
        """Initializes the layer."""
        super().__init__()
        self.num_dims = num_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reshapes `x` to have `self.num_dims` dims, padding with size-1 dims."""
        shape = list(x.shape)
        shape = shape + [1] * (self.num_dims - len(shape))
        return x.view(shape)


class FIRResample(FIRFilterbank):
    """Fixed FIR-filter resampling layer (unused by the currently vendored archs)."""

    def __init__(
        self,
        sr_input: float = 20e3,
        sr_output: float = 10e3,
        fir_dur: float = 0.05,
        cutoff: float = None,
        window: tuple = ("kaiser", 5.0),
        dtype: torch.dtype = torch.float32,
    ):
        """Initializes the resampling filter.

        Args:
            sr_input: Input sample rate in Hz.
            sr_output: Output sample rate in Hz. Must be `<= sr_input` and
                evenly divide it.
            fir_dur: FIR filter duration in seconds.
            cutoff: Lowpass cutoff in Hz. Defaults to `sr_output / 2`.
            window: `scipy.signal.firwin` window spec.
            dtype: Dtype for the filter weights.
        """
        assert sr_output <= sr_input, f"{sr_output=} > {sr_input=}"
        numtaps = int(sr_input * fir_dur)
        if numtaps % 2 == 0:
            numtaps = numtaps + 1
        if cutoff is None:
            cutoff = sr_output / 2
        fir = scipy.signal.firwin(
            numtaps=numtaps,
            cutoff=cutoff,
            width=None,
            window=tuple(window),
            pass_zero=True,
            scale=True,
            fs=sr_input,
        )
        stride = int(sr_input / sr_output)
        msg = f"{sr_input=} and {sr_output=} require non-integer stride"
        assert np.isclose(stride, sr_input / sr_output), msg
        super().__init__(fir, dtype=dtype, stride=stride)
