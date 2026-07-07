from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from mnist import PAPER, evaluate_mnist_folders
from shared import CsvRow, find_repo_root, write_csv, write_text

REQUIREMENTS = ["M1", "M2", "M3", "M4", "M5", "M6"]
RESULT_FIELDS = [
    "requirement",
    "n_train_images",
    "n_generated_images",
    "pass_rate",
    "paper_lr_pass_rate",
    "delta_vs_paper_lr",
    "failure_count",
    "last_loss",
    "train_seconds",
]
LOG_FIELDS = ["epoch", "loss", "reconstruction_loss", "kl_loss"]


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 40
    batch_size: int = 64
    latent_dim: int = 48
    learning_rate: float = 1e-3
    kl_weight: float = 0.01
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
                "loss": f"{total_loss / n_batches:.6f}",
                "reconstruction_loss": f"{total_reconstruction / n_batches:.6f}",
                "kl_loss": f"{total_kl / n_batches:.6f}",
            }
        )
    return model, log_rows, time.perf_counter() - start


@torch.no_grad()
def generate_images(root: Path, model: ConditionalVAE, config: TrainConfig) -> None:
    seed_everything(config.seed + 1)
    device = next(model.parameters()).device
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
        z = torch.randn(config.samples_per_requirement, config.latent_dim, device=device)
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


def evaluate_generated(
    root: Path,
    log_rows: list[CsvRow],
    train_seconds: float,
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
        rows.append(
            {
                "requirement": requirement,
                "n_train_images": str(
                    len(list((root / "data" / "images" / "mnist" / requirement).glob("*.png")))
                ),
                "n_generated_images": str(counts[requirement]),
                "pass_rate": f"{row['pass_rate']:.6f}",
                "paper_lr_pass_rate": f"{paper:.6f}",
                "delta_vs_paper_lr": f"{row['pass_rate'] - paper:+.6f}",
                "failure_count": str(len(row["failures"])),
                "last_loss": last_loss,
                "train_seconds": f"{train_seconds:.2f}",
            }
        )
    return rows


def write_outputs(
    root: Path,
    result_rows: list[CsvRow],
    log_rows: list[CsvRow],
    sample_grid: Path,
) -> list[Path]:
    out_dir = root / "experiments" / "mnist-shared-generator"
    results = out_dir / "results.csv"
    log = out_dir / "training-log.csv"
    summary_path = out_dir / "summary.md"
    write_csv(results, RESULT_FIELDS, result_rows)
    write_csv(log, LOG_FIELDS, log_rows)
    write_text(summary_path, summary(result_rows, sample_grid))
    return [results, log, summary_path, sample_grid]


def summary(rows: list[CsvRow], sample_grid: Path) -> str:
    mean_pass = sum(float(row["pass_rate"]) for row in rows) / len(rows)
    mean_paper = sum(float(row["paper_lr_pass_rate"]) for row in rows) / len(rows)
    worst = min(rows, key=lambda row: float(row["pass_rate"]))
    n_generated = rows[0]["n_generated_images"]
    lines = [
        "# MNIST Shared Generator Summary",
        "",
        "A single conditional VAE was trained on the copied RBT4DNN MNIST LoRA images "
        f"for M1-M6, then asked to generate {n_generated} images per requirement.",
        "",
        "This is a cheap shared-generator baseline, not a FLUX LoRA reproduction.",
        "",
        f"Mean pass rate: {mean_pass:.3f} versus {mean_paper:.3f} for the paper's "
        "per-requirement LoRA reference.",
        f"Worst requirement: {worst['requirement']} at pass {worst['pass_rate']} "
        f"({worst['failure_count']} failures).",
        f"Sample grid: `{sample_grid.relative_to(sample_grid.parents[2])}`.",
        "",
    ]
    lines += [
        f"- {row['requirement']}: pass {row['pass_rate']} (delta {row['delta_vs_paper_lr']})"
        for row in rows
    ]
    return "\n".join(lines) + "\n"


def train_and_evaluate(root: Path | None = None, config: TrainConfig | None = None) -> list[Path]:
    root = find_repo_root(root)
    config = config or TrainConfig()
    model, log_rows, train_seconds = train_model(root, config)
    generate_images(root, model, config)
    rows = evaluate_generated(root, log_rows, train_seconds)
    sample_grid = save_sample_grid(root)
    return write_outputs(root, rows, log_rows, sample_grid)
