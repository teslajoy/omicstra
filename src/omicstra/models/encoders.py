"""the encoder registry and the two wrappers the seed cohort declares.

an Encoder is deliberately tiny:

    .spec              what it is, and what it may be used on
    .embed(batch)      inputs -> (n, dim) float32

everything else - tiling, pooling, neighbour graphs - belongs to measures/,
because it is the same maths whichever encoder produced the vectors.

capability metadata is not decoration. `min_units` and `trained_on` are read by
the compute contract's `platform_floor` and `encoder_compatibility` gates, which
is the difference between "Novae needs 512 spots" being a fact the system acts on
and a sentence in a README nobody reads.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


class EncoderUnavailable(RuntimeError):
    """the encoder cannot run here, and the message says which of the reasons.

    three distinct causes, deliberately one exception: the name is not declared,
    the tensor library is absent, or the weights are gated and no token was
    supplied. a caller that only wants to know "can I proceed" should not have to
    catch three types, and the gate that asks a person reads the message.
    """


@dataclass(frozen=True)
class EncoderSpec:
    """what a cohort is choosing when it names an encoder.

    `min_units` is a property of the ENCODER, not of the cohort - Novae's
    prototype count sets a floor below which its output is not meaningful, and
    that floor is the same on every dataset. the cohort decides what to do about
    units below it; the encoder only declares where the line is.
    """
    name: str
    dim: int
    role: str                       # primary | st | pathway
    modality: str                   # he | molecular
    unit: str                       # tile | spot | niche
    trained_on: str                 # provenance, read by the compatibility gate
    gated: bool = False             # weights require accepting terms + a token
    min_units: int | None = None    # below this the output is not meaningful
    notes: str = ""
    extras: tuple[str, ...] = field(default=("encode",))


_REGISTRY: dict[str, tuple[EncoderSpec, Callable[[], Any]]] = {}


def register(spec_: EncoderSpec, loader: Callable[[], Any]) -> None:
    """add an encoder. a third party can call this without editing this file."""
    _REGISTRY[spec_.name] = (spec_, loader)


def available() -> list[str]:
    return sorted(_REGISTRY)


def spec(name: str) -> EncoderSpec:
    """metadata only - importable and callable with no tensor library present.

    this is what lets `describe` and the preflight gates reason about an encoder
    on a machine that could not possibly run it.
    """
    if name not in _REGISTRY:
        raise EncoderUnavailable(
            f"no encoder named {name!r}. declared encoders: {', '.join(available()) or 'none'}. "
            "register one with omicstra.models.register() rather than editing the package.")
    return _REGISTRY[name][0]


def load(name: str) -> Any:
    """materialise the model. THIS is the call that needs torch."""
    spec(name)          # refuse an undeclared name BEFORE importing a tensor library
    try:
        return _REGISTRY[name][1]()
    except ImportError as e:
        raise EncoderUnavailable(
            f"{name} needs the encode extra: pip install 'omicstra[encode]' ({e})") from e


# --- Virchow2 ---------------------------------------------------------------
def _load_virchow2():
    import timm
    import torch
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from timm.layers import SwiGLUPacked

    m = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True,
                          mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)
    m.eval()
    # the transform travels WITH the encoder. it is resolved from the model's own
    # pretrained_cfg - mean, std and interpolation are properties of the weights,
    # not of the cohort - so a caller that supplies its own normalisation is
    # silently encoding different images than the published grid did.
    return _Virchow2(m, create_transform(**resolve_data_config(m.pretrained_cfg, model=m)))


class _Virchow2:
    """CLS token only, which is what the published grid used.

    Virchow2 returns CLS + 4 register tokens + patch tokens. the registers are
    an artefact of the architecture and are NOT image content; concatenating
    mean-pooled patch tokens is the other common recipe and gives 2560-d. the
    grid used 1280-d, so CLS alone is what reproduces it - and that choice is
    recorded here rather than left to whoever reads the tensor next.
    """

    def __init__(self, model: Any, transform: Any = None) -> None:
        self.model = model
        self.transform = transform

    def to(self, device: Any) -> _Virchow2:
        self.model = self.model.to(device)
        self.device = device
        return self

    def embed(self, batch: Any) -> Any:
        import torch
        with torch.inference_mode():
            out = self.model(batch)          # (n, 1 + 4 + patches, 1280)
        return out[:, 0].float().cpu().numpy()

    def embed_images(self, images: list, batch_size: int = 32) -> Any:
        """PIL images -> (n, 1280) float32, batched.

        batch_size is the published grid's 32. it is not a free parameter for
        reproduction: batching changes nothing mathematically, but it is recorded
        so a run that differs has one fewer unexplained difference.
        """
        import numpy as np
        import torch

        dev = getattr(self, "device", None)
        out = []
        for j in range(0, len(images), batch_size):
            b = torch.stack([self.transform(p) for p in images[j:j + batch_size]])
            out.append(self.embed(b.to(dev) if dev is not None else b))
        return np.vstack(out).astype(np.float32) if out else np.zeros((0, 1280), np.float32)


register(
    EncoderSpec(
        name="virchow2", dim=1280, role="primary", modality="he", unit="tile",
        trained_on="3.1M whole-slide images, Memorial Sloan Kettering (Vorontsov et al. 2024, Paige)",
        gated=True,
        notes="CLS token. gated on Hugging Face - accepting terms and a token is the "
              "gated_weights preflight gate, not a runtime surprise.",
    ),
    _load_virchow2,
)


# --- Novae ------------------------------------------------------------------
def _load_novae():
    import novae

    return _Novae(novae.Novae.from_pretrained("MICS-Lab/novae-human-0"))


class _Novae:
    """zero-shot representations over a spatial neighbour graph.

    the graph is built from the cohort's declared `st_graph`, not chosen here -
    bare Delaunay reproduces the published grid, and a radius cap is a different
    arm rather than a tweak. see platform.json#platforms.*.st_graph.
    """

    def __init__(self, model: Any) -> None:
        self.model = model

    def embed(self, adata: Any) -> Any:
        self.model.compute_representations(adata, zero_shot=True)
        return adata.obsm["novae_latent"]


register(
    EncoderSpec(
        name="novae", dim=64, role="st", modality="molecular", unit="spot",
        trained_on="image-based ST only - MERSCOPE, Xenium, CosMx, ~30M cells at "
                   "subcellular resolution (Blampey et al. 2025, Nat. Methods, MICS-Lab)",
        min_units=512,
        notes="min_units is the prototype count: below 512 spots the zero-shot output is "
              "not meaningful, which is the platform_floor gate. the training distribution "
              "does NOT overlap 100um spot platforms, so compatibility on such a cohort is "
              "measured by the encoder_compatibility probe rather than assumed.",
    ),
    _load_novae,
)