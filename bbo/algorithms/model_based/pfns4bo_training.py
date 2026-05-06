"""Utilities for training custom PFNs4BO-compatible surrogate checkpoints."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .pfns4bo_utils import require_pfns4bo, require_torch, select_pfns_device

DEFAULT_CUSTOM_PFN_MODEL_NAME = "custom_pfn_model.pt"
SUPPORTED_CUSTOM_PFN_PRIORS = ("gp", "hebo")


@dataclass(frozen=True)
class CustomPfnTrainingArtifact:
    """Files emitted by a custom PFN training run."""

    model_path: Path
    metadata_path: Path
    prior: str
    max_features: int
    epochs: int
    steps_per_epoch: int
    seq_len: int


def _load_pfns4bo_training_modules() -> dict[str, Any]:
    require_pfns4bo()
    from pfns4bo import bar_distribution, encoders, priors, train, utils

    return {
        "bar_distribution": bar_distribution,
        "encoders": encoders,
        "priors": priors,
        "train": train,
        "utils": utils,
    }


def _build_encoder_generator(modules: dict[str, Any]) -> Any:
    encoders = modules["encoders"]
    return encoders.get_normalized_uniform_encoder(encoders.get_variable_num_features_encoder(encoders.Linear))


def _build_hebo_recipe(
    modules: dict[str, Any],
    *,
    max_features: int,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    seq_len: int,
    emsize: int,
    nhid: int,
    nlayers: int,
    nhead: int,
    lr: float,
    train_mixed_precision: bool,
) -> dict[str, Any]:
    priors = modules["priors"]
    utils = modules["utils"]
    return {
        "priordataloader_class": priors.get_batch_to_dataloader(
            priors.get_batch_sequence(
                priors.hebo_prior.get_batch,
                priors.utils.sample_num_feaetures_get_batch,
            )
        ),
        "encoder_generator": _build_encoder_generator(modules),
        "emsize": emsize,
        "nhead": nhead,
        "warmup_epochs": 1,
        "y_encoder_generator": modules["encoders"].Linear,
        "batch_size": batch_size,
        "scheduler": utils.get_cosine_schedule_with_warmup,
        "extra_prior_kwargs_dict": {
            "num_features": max_features,
            "hyperparameters": {
                "lengthscale_concentration": 1.2106559584074301,
                "lengthscale_rate": 1.5212245992840594,
                "outputscale_concentration": 0.8452312502679863,
                "outputscale_rate": 0.3993553245745406,
                "add_linear_kernel": False,
                "power_normalization": False,
                "hebo_warping": False,
                "unused_feature_likelihood": 0.3,
                "observation_noise": True,
                "sample_num_features": True,
            },
        },
        "epochs": epochs,
        "lr": lr,
        "bptt": seq_len,
        "single_eval_pos_gen": utils.get_uniform_single_eval_pos_sampler(max(2, seq_len - 10), min_len=1),
        "aggregate_k_gradients": 1,
        "nhid": nhid,
        "steps_per_epoch": steps_per_epoch,
        "weight_decay": 0.0,
        "train_mixed_precision": train_mixed_precision,
        "efficient_eval_masking": True,
        "nlayers": nlayers,
    }


def _build_gp_recipe(
    modules: dict[str, Any],
    *,
    max_features: int,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    seq_len: int,
    emsize: int,
    nhid: int,
    nlayers: int,
    nhead: int,
    lr: float,
    train_mixed_precision: bool,
) -> dict[str, Any]:
    priors = modules["priors"]
    utils = modules["utils"]
    return {
        "priordataloader_class": priors.get_batch_to_dataloader(
            priors.get_batch_sequence(
                priors.fast_gp.get_batch,
                priors.utils.sample_num_feaetures_get_batch,
            )
        ),
        "encoder_generator": _build_encoder_generator(modules),
        "emsize": emsize,
        "nhead": nhead,
        "warmup_epochs": 1,
        "y_encoder_generator": modules["encoders"].Linear,
        "batch_size": batch_size,
        "scheduler": utils.get_cosine_schedule_with_warmup,
        "extra_prior_kwargs_dict": {
            "num_features": max_features,
            "hyperparameters": {
                "noise": 0.1,
                "outputscale": 0.1,
                "lengthscale": 0.1,
                "sampling": "uniform",
                "observation_noise": True,
                "sample_num_features": True,
            },
        },
        "epochs": epochs,
        "lr": lr,
        "bptt": seq_len,
        "single_eval_pos_gen": utils.get_uniform_single_eval_pos_sampler(max(2, seq_len - 10), min_len=1),
        "aggregate_k_gradients": 1,
        "nhid": nhid,
        "steps_per_epoch": steps_per_epoch,
        "weight_decay": 0.0,
        "train_mixed_precision": train_mixed_precision,
        "efficient_eval_masking": True,
        "nlayers": nlayers,
    }


def build_custom_pfn_training_recipe(
    *,
    prior: str,
    max_features: int,
    epochs: int,
    steps_per_epoch: int,
    batch_size: int,
    seq_len: int,
    emsize: int,
    nhid: int,
    nlayers: int,
    nhead: int,
    lr: float,
    train_mixed_precision: bool,
) -> dict[str, Any]:
    """Build a PFNs4BO training recipe aligned with the official prior families."""

    if prior not in SUPPORTED_CUSTOM_PFN_PRIORS:
        available = ", ".join(SUPPORTED_CUSTOM_PFN_PRIORS)
        raise ValueError(f"Unknown custom PFN prior `{prior}`. Available: {available}")

    modules = _load_pfns4bo_training_modules()
    if max_features <= 0:
        raise ValueError("max_features must be positive.")
    if epochs <= 0 or steps_per_epoch <= 0 or batch_size <= 0 or seq_len <= 1:
        raise ValueError("Training hyperparameters must be positive and seq_len must be > 1.")

    recipe_kwargs = {
        "modules": modules,
        "max_features": int(max_features),
        "epochs": int(epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "batch_size": int(batch_size),
        "seq_len": int(seq_len),
        "emsize": int(emsize),
        "nhid": int(nhid),
        "nlayers": int(nlayers),
        "nhead": int(nhead),
        "lr": float(lr),
        "train_mixed_precision": bool(train_mixed_precision),
    }
    if prior == "hebo":
        return _build_hebo_recipe(**recipe_kwargs)
    return _build_gp_recipe(**recipe_kwargs)


def _feature_probe_counts(max_features: int) -> tuple[int, ...]:
    probes = {1, min(2, max_features), min(8, max_features), max_features}
    return tuple(sorted(probe for probe in probes if probe > 0))


def build_custom_pfn_criterion(recipe: dict[str, Any], *, device: str, num_buckets: int = 256) -> Any:
    """Match the official PFNs4BO training recipe for target-space bucketization."""

    pfns4bo_module = require_pfns4bo()
    torch = require_torch()
    modules = _load_pfns4bo_training_modules()
    bar_distribution = modules["bar_distribution"]

    batch_size = int(recipe["batch_size"])
    seq_len = int(recipe.get("seq_len", recipe.get("bptt", 10)))
    extra_prior_kwargs = recipe["extra_prior_kwargs_dict"]
    prior_kwargs = dict(extra_prior_kwargs.get("hyperparameters", {}))
    prior_kwargs["num_hyperparameter_samples_per_batch"] = -1
    max_features = int(extra_prior_kwargs["num_features"])

    all_targets = []
    for feature_count in _feature_probe_counts(max_features):
        batch = recipe["priordataloader_class"].get_batch_method(
            batch_size,
            seq_len,
            feature_count,
            epoch=0,
            device=device,
            hyperparameters=prior_kwargs,
        )
        all_targets.append(batch.target_y.flatten())
    ys = torch.cat(all_targets, 0).cpu()
    return bar_distribution.FullSupportBarDistribution(
        bar_distribution.get_bucket_limits(num_buckets, ys=ys)
    )


def train_custom_pfn_model(
    *,
    output_dir: str | Path,
    prior: str,
    max_features: int,
    model_name: str = DEFAULT_CUSTOM_PFN_MODEL_NAME,
    device: str | None = None,
    epochs: int = 1,
    steps_per_epoch: int = 4,
    batch_size: int = 8,
    seq_len: int = 32,
    emsize: int = 64,
    nhid: int = 128,
    nlayers: int = 4,
    nhead: int = 4,
    lr: float = 1e-4,
    train_mixed_precision: bool = False,
    num_buckets: int = 256,
) -> CustomPfnTrainingArtifact:
    """Train one small custom PFN checkpoint and persist both weights and metadata."""

    modules = _load_pfns4bo_training_modules()
    torch = require_torch()
    resolved_device = select_pfns_device(device)
    recipe = build_custom_pfn_training_recipe(
        prior=prior,
        max_features=max_features,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        seq_len=seq_len,
        emsize=emsize,
        nhid=nhid,
        nlayers=nlayers,
        nhead=nhead,
        lr=lr,
        train_mixed_precision=train_mixed_precision,
    )
    criterion = build_custom_pfn_criterion(recipe, device=resolved_device, num_buckets=num_buckets)
    recipe_with_criterion = {
        **recipe,
        "criterion": criterion,
        "gpu_device": resolved_device,
        "progress_bar": False,
        "verbose": True,
    }
    _, _, model, _ = modules["train"].train(**recipe_with_criterion)

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / model_name
    metadata_path = model_path.with_suffix(".metadata.json")
    torch.save(model, model_path)

    metadata = {
        "prior": prior,
        "max_features": int(max_features),
        "epochs": int(epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "batch_size": int(batch_size),
        "seq_len": int(seq_len),
        "emsize": int(emsize),
        "nhid": int(nhid),
        "nlayers": int(nlayers),
        "nhead": int(nhead),
        "lr": float(lr),
        "train_mixed_precision": bool(train_mixed_precision),
        "num_buckets": int(num_buckets),
        "device": resolved_device,
        "model_path": str(model_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return CustomPfnTrainingArtifact(
        model_path=model_path,
        metadata_path=metadata_path,
        prior=prior,
        max_features=int(max_features),
        epochs=int(epochs),
        steps_per_epoch=int(steps_per_epoch),
        seq_len=int(seq_len),
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train one custom PFNs4BO-compatible checkpoint.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prior", choices=SUPPORTED_CUSTOM_PFN_PRIORS, default="hebo")
    parser.add_argument("--max-features", type=int, required=True)
    parser.add_argument("--model-name", default=DEFAULT_CUSTOM_PFN_MODEL_NAME)
    parser.add_argument("--device", default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--emsize", type=int, default=64)
    parser.add_argument("--nhid", type=int, default=128)
    parser.add_argument("--nlayers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-mixed-precision", action="store_true")
    parser.add_argument("--num-buckets", type=int, default=256)
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    artifact = train_custom_pfn_model(
        output_dir=args.output_dir,
        prior=args.prior,
        max_features=args.max_features,
        model_name=args.model_name,
        device=args.device,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        emsize=args.emsize,
        nhid=args.nhid,
        nlayers=args.nlayers,
        nhead=args.nhead,
        lr=args.lr,
        train_mixed_precision=args.train_mixed_precision,
        num_buckets=args.num_buckets,
    )
    print(json.dumps(asdict(artifact), default=str, indent=2, sort_keys=True))


__all__ = [
    "CustomPfnTrainingArtifact",
    "DEFAULT_CUSTOM_PFN_MODEL_NAME",
    "SUPPORTED_CUSTOM_PFN_PRIORS",
    "build_custom_pfn_criterion",
    "build_custom_pfn_training_recipe",
    "main",
    "train_custom_pfn_model",
]


if __name__ == "__main__":
    main()
