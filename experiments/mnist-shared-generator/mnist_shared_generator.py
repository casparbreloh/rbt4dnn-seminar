from __future__ import annotations

import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from statistics import mean, stdev

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from mnist import PAPER, evaluate_mnist_folders
from shared import CsvRow, find_repo_root, validate_image_corpus, write_csv, write_text

REQUIREMENTS = ["M1", "M2", "M3", "M4", "M5", "M6"]
RESULT_FIELDS = [
    "requirement",
    "n_seeds",
    "n_train_images",
    "n_generated_images_per_seed",
    "pass_rate_mean",
    "pass_rate_std",
    "pass_rate_min",
    "pass_rate_max",
    "paper_lr_pass_rate",
    "delta_mean_vs_paper_lr",
    "failure_count_mean",
    "nearest_train_mse_mean",
    "nearest_train_mse_min",
    "exact_train_matches_total",
    "train_seconds_total",
]
SEED_RESULT_FIELDS = [
    "seed",
    "requirement",
    "n_train_images",
    "n_generated_images",
    "pass_rate",
    "paper_lr_pass_rate",
    "delta_vs_paper_lr",
    "failure_count",
    "nearest_train_mse_mean",
    "nearest_train_mse_min",
    "exact_train_matches",
    "last_loss",
    "train_seconds",
]
LOG_FIELDS = ["seed", "epoch", "loss", "reconstruction_loss", "kl_loss"]
DEFAULT_SEEDS = [7, 13, 29]


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 100
    batch_size: int = 64
    latent_dim: int = 48
    learning_rate: float = 1e-3
    kl_weight: float = 0.01
    latent_noise: float = 0.15
    samples_per_requirement: int = 100
    seed: int = 7


class RequirementImages(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, root: Path) -> None:
        self.items: list[tuple[Path, int]] = []
        for label, requirement in enumerate(REQUIREMENTS):
            folder = root / "data" / "images" / "mnist" / requirement
            self.items.extend((path, label) for path in sorted(folder.glob("*.png")))
        self.transform = transforms.Compose(
            [transforms.Grayscale(), transforms.Resize((64, 64)), transforms.ToTensor()]
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.items[index]
        with Image.open(path) as image:
            tensor = self.transform(image)
        return tensor, torch.tensor(label, dtype=torch.long)


class ConditionalVAE(nn.Module):
    def __init__(self, n_labels: int, latent_dim: int) -> None:
        super().__init__()
        self.n_labels = n_labels
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1 + n_labels, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_decode = nn.Linear(latent_dim + n_labels, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
            nn.Sigmoid(),
        )

    def one_hot(self, labels: torch.Tensor) -> torch.Tensor:
        return F.one_hot(labels, self.n_labels).float()

    def condition_image(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cond = self.one_hot(labels).view(labels.shape[0], self.n_labels, 1, 1)
        cond = cond.expand(-1, -1, images.shape[2], images.shape[3])
        return torch.cat([images, cond], dim=1)

    def encode(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(self.condition_image(images, labels))
        return self.fc_mu(features), self.fc_logvar(features)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = torch.cat([z, self.one_hot(labels)], dim=1)
        features = self.fc_decode(features).view(-1, 256, 4, 4)
        return self.decoder(features)

    def forward(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(images, labels)
        return self.decode(self.reparameterize(mu, logvar), labels), mu, logvar


def device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loss_parts(
    generated: torch.Tensor,
    images: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    reconstruction = F.binary_cross_entropy(generated, images, reduction="sum") / images.shape[0]
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / images.shape[0]
    return reconstruction + kl_weight * kl, reconstruction, kl


def output_dir(root: Path) -> Path:
    return root / "outputs" / "mnist-shared-generator"


def generated_folders(root: Path) -> dict[str, Path]:
    base = output_dir(root) / "generated"
    return {requirement: base / requirement for requirement in REQUIREMENTS}


def requirement_folder(root: Path, requirement: str) -> Path:
    return root / "data" / "images" / "mnist" / requirement


def train_model(root: Path, config: TrainConfig) -> tuple[ConditionalVAE, list[CsvRow], float]:
    seed_everything(config.seed)
    device = torch.device(device_name())
    dataset = RequirementImages(root)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    model = ConditionalVAE(len(REQUIREMENTS), config.latent_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    log_rows: list[CsvRow] = []
    start = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        total_loss = total_reconstruction = total_kl = 0.0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            generated, mu, logvar = model(images, labels)
            loss, reconstruction, kl = loss_parts(generated, images, mu, logvar, config.kl_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_reconstruction += reconstruction.item()
            total_kl += kl.item()

        n_batches = len(loader)
        log_rows.append(
            {
                "epoch": str(epoch),
                "seed": str(config.seed),
                "loss": f"{total_loss / n_batches:.6f}",
                "reconstruction_loss": f"{total_reconstruction / n_batches:.6f}",
                "kl_loss": f"{total_kl / n_batches:.6f}",
            }
        )
    return model, log_rows, time.perf_counter() - start


@torch.no_grad()
def training_latents(
    root: Path,
    model: ConditionalVAE,
    config: TrainConfig,
) -> dict[int, torch.Tensor]:
    device = next(model.parameters()).device
    loader = DataLoader(RequirementImages(root), batch_size=config.batch_size)
    latents: dict[int, list[torch.Tensor]] = {label: [] for label in range(len(REQUIREMENTS))}

    model.eval()
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        mu, _ = model.encode(images, labels)
        for label in range(len(REQUIREMENTS)):
            selected = mu[labels == label]
            if len(selected) > 0:
                latents[label].append(selected)
    return {label: torch.cat(parts) for label, parts in latents.items()}


@torch.no_grad()
def generate_images(root: Path, model: ConditionalVAE, config: TrainConfig) -> None:
    seed_everything(config.seed + 1)
    device = next(model.parameters()).device
    latents = training_latents(root, model, config)
    folders = generated_folders(root)
    for folder in folders.values():
        if folder.exists():
            for path in folder.glob("*.png"):
                path.unlink()
        folder.mkdir(parents=True, exist_ok=True)

    model.eval()
    for label, requirement in enumerate(REQUIREMENTS):
        labels = torch.full(
            (config.samples_per_requirement,), label, dtype=torch.long, device=device
        )
        source = latents[label]
        indices = torch.randint(len(source), (config.samples_per_requirement,), device=device)
        z = source[indices] + config.latent_noise * torch.randn(
            config.samples_per_requirement,
            config.latent_dim,
            device=device,
        )
        images = model.decode(z, labels).cpu()
        for index, image in enumerate(images):
            save_image(image, folders[requirement] / f"{index}.png")


def save_sample_grid(root: Path) -> Path:
    images = []
    for requirement, folder in generated_folders(root).items():
        for path in sorted(folder.glob("*.png"))[:8]:
            image = transforms.ToTensor()(Image.open(path).convert("L"))
            images.append(image)
        if len(images) % 8 != 0:
            raise RuntimeError(f"Missing generated samples for {requirement}")
    grid = make_grid(images, nrow=8, padding=2)
    path = root / "experiments" / "mnist-shared-generator" / "samples.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, path)
    return path


def image_matrix(folder: Path) -> torch.Tensor:
    transform = transforms.Compose(
        [transforms.Grayscale(), transforms.Resize((64, 64)), transforms.ToTensor()]
    )
    images = []
    for path in sorted(folder.glob("*.png")):
        with Image.open(path) as image:
            images.append(transform(image).flatten())
    return torch.stack(images)


def nearest_training_metrics(root: Path, requirement: str, generated: Path) -> CsvRow:
    generated_images = image_matrix(generated)
    training_images = image_matrix(requirement_folder(root, requirement))
    nearest = torch.cdist(generated_images, training_images).min(dim=1).values.pow(2)
    nearest_mse = nearest / generated_images.shape[1]
    return {
        "nearest_train_mse_mean": f"{nearest_mse.mean().item():.6f}",
        "nearest_train_mse_min": f"{nearest_mse.min().item():.6f}",
        "exact_train_matches": str(int((nearest_mse <= 1e-8).sum().item())),
    }


def evaluate_generated(
    root: Path,
    log_rows: list[CsvRow],
    train_seconds: float,
    seed: int,
    batch_size: int = 100,
) -> list[CsvRow]:
    last_loss = log_rows[-1]["loss"] if log_rows else ""
    counts = {
        requirement: len(list(folder.glob("*.png")))
        for requirement, folder in generated_folders(root).items()
    }
    rows: list[CsvRow] = []
    for row in evaluate_mnist_folders(generated_folders(root), batch_size=batch_size):
        requirement = row["requirement"]
        paper = PAPER[requirement]
        novelty = nearest_training_metrics(root, requirement, generated_folders(root)[requirement])
        rows.append(
            {
                "seed": str(seed),
                "requirement": requirement,
                "n_train_images": str(
                    len(list(requirement_folder(root, requirement).glob("*.png")))
                ),
                "n_generated_images": str(counts[requirement]),
                "pass_rate": f"{row['pass_rate']:.6f}",
                "paper_lr_pass_rate": f"{paper:.6f}",
                "delta_vs_paper_lr": f"{row['pass_rate'] - paper:+.6f}",
                "failure_count": str(len(row["failures"])),
                "nearest_train_mse_mean": novelty["nearest_train_mse_mean"],
                "nearest_train_mse_min": novelty["nearest_train_mse_min"],
                "exact_train_matches": novelty["exact_train_matches"],
                "last_loss": last_loss,
                "train_seconds": f"{train_seconds:.2f}",
            }
        )
    return rows


def aggregate_seed_rows(rows: list[CsvRow]) -> list[CsvRow]:
    by_requirement: dict[str, list[CsvRow]] = {}
    for row in rows:
        by_requirement.setdefault(row["requirement"], []).append(row)

    out: list[CsvRow] = []
    for requirement, req_rows in sorted(by_requirement.items()):
        pass_rates = [float(row["pass_rate"]) for row in req_rows]
        nearest_means = [float(row["nearest_train_mse_mean"]) for row in req_rows]
        nearest_mins = [float(row["nearest_train_mse_min"]) for row in req_rows]
        paper = float(req_rows[0]["paper_lr_pass_rate"])
        failure_counts = [float(row["failure_count"]) for row in req_rows]
        train_seconds = [float(row["train_seconds"]) for row in req_rows]
        pass_mean = mean(pass_rates)

        out.append(
            {
                "requirement": requirement,
                "n_seeds": str(len(req_rows)),
                "n_train_images": req_rows[0]["n_train_images"],
                "n_generated_images_per_seed": req_rows[0]["n_generated_images"],
                "pass_rate_mean": f"{pass_mean:.6f}",
                "pass_rate_std": f"{stdev(pass_rates):.6f}" if len(pass_rates) > 1 else "0.000000",
                "pass_rate_min": f"{min(pass_rates):.6f}",
                "pass_rate_max": f"{max(pass_rates):.6f}",
                "paper_lr_pass_rate": f"{paper:.6f}",
                "delta_mean_vs_paper_lr": f"{pass_mean - paper:+.6f}",
                "failure_count_mean": f"{mean(failure_counts):.3f}",
                "nearest_train_mse_mean": f"{mean(nearest_means):.6f}",
                "nearest_train_mse_min": f"{min(nearest_mins):.6f}",
                "exact_train_matches_total": str(
                    sum(int(row["exact_train_matches"]) for row in req_rows)
                ),
                "train_seconds_total": f"{sum(train_seconds):.2f}",
            }
        )
    return out


def write_outputs(
    root: Path,
    result_rows: list[CsvRow],
    seed_rows: list[CsvRow],
    log_rows: list[CsvRow],
    sample_grid: Path,
) -> list[Path]:
    out_dir = root / "experiments" / "mnist-shared-generator"
    results = out_dir / "results.csv"
    seed_results = out_dir / "seed-results.csv"
    log = out_dir / "training-log.csv"
    summary_path = out_dir / "summary.md"
    write_csv(results, RESULT_FIELDS, result_rows)
    write_csv(seed_results, SEED_RESULT_FIELDS, seed_rows)
    write_csv(log, LOG_FIELDS, log_rows)
    write_text(summary_path, summary(result_rows, sample_grid))
    return [results, seed_results, log, summary_path, sample_grid]


def summary(rows: list[CsvRow], sample_grid: Path) -> str:
    mean_pass = sum(float(row["pass_rate_mean"]) for row in rows) / len(rows)
    mean_paper = sum(float(row["paper_lr_pass_rate"]) for row in rows) / len(rows)
    exact_matches = sum(int(row["exact_train_matches_total"]) for row in rows)
    mean_nearest_mse = sum(float(row["nearest_train_mse_mean"]) for row in rows) / len(rows)
    worst = min(rows, key=lambda row: float(row["pass_rate_mean"]))
    n_generated = rows[0]["n_generated_images_per_seed"]
    n_seeds = rows[0]["n_seeds"]
    lines = [
        "# MNIST Shared Generator Summary",
        "",
        "**Question:** Can a small shared generator approximate the paper's per-requirement "
        "LoRA behavior on MNIST?",
        "",
        "**Method:** Train one conditional VAE on copied RBT4DNN MNIST LoRA images for M1-M6, "
        f"using {n_seeds} seeds and {n_generated} generated images per requirement per seed.",
        "",
        f"**Result:** Mean pass rate is {mean_pass:.3f} versus {mean_paper:.3f} for the "
        "paper's per-requirement LoRA reference, with no exact training-image matches.",
        "",
        "**Limitation:** This is a cheap shared-generator baseline, not a FLUX LoRA "
        "reproduction, and MNIST is much easier than natural-image datasets.",
        "",
        f"Exact generated/training image matches: {exact_matches}. Mean nearest-train MSE: "
        f"{mean_nearest_mse:.4f}.",
        f"Worst requirement: {worst['requirement']} at mean pass {worst['pass_rate_mean']} "
        f"({worst['failure_count_mean']} mean failures).",
        f"Sample grid: `{sample_grid.relative_to(sample_grid.parents[2])}`.",
        "",
    ]
    lines += [
        f"- {row['requirement']}: mean pass {row['pass_rate_mean']} "
        f"(std {row['pass_rate_std']}, delta {row['delta_mean_vs_paper_lr']})"
        for row in rows
    ]
    return "\n".join(lines) + "\n"


def train_and_evaluate(
    root: Path | None = None,
    config: TrainConfig | None = None,
    seeds: list[int] | None = None,
) -> list[Path]:
    root = find_repo_root(root)
    validate_image_corpus(root, ["mnist"])
    config = config or TrainConfig()
    seeds = seeds or DEFAULT_SEEDS
    seed_rows: list[CsvRow] = []
    all_log_rows: list[CsvRow] = []
    sample_grid: Path | None = None

    for seed in seeds:
        seed_config = replace(config, seed=seed)
        model, log_rows, train_seconds = train_model(root, seed_config)
        generate_images(root, model, seed_config)
        seed_rows.extend(evaluate_generated(root, log_rows, train_seconds, seed))
        all_log_rows.extend(log_rows)
        if sample_grid is None:
            sample_grid = save_sample_grid(root)

    if sample_grid is None:
        raise RuntimeError("No shared-generator runs were executed.")
    return write_outputs(root, aggregate_seed_rows(seed_rows), seed_rows, all_log_rows, sample_grid)
