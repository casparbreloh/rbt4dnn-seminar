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
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from shared import CsvRow, find_repo_root, validate_image_corpus, write_csv, write_text

REQUIREMENTS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
DEFAULT_SEEDS = [7]
RESULT_FIELDS = [
    "requirement",
    "n_seeds",
    "n_train_images",
    "n_generated_images_per_seed",
    "classifier_top1_mean",
    "classifier_top1_std",
    "classifier_confidence_mean",
    "classifier_margin_mean",
    "nearest_train_mse_mean",
    "nearest_train_mse_min",
    "exact_train_matches_total",
    "classifier_val_accuracy",
    "train_seconds_total",
]
SEED_RESULT_FIELDS = [
    "seed",
    "requirement",
    "n_train_images",
    "n_generated_images",
    "classifier_top1",
    "classifier_confidence",
    "classifier_margin",
    "nearest_train_mse_mean",
    "nearest_train_mse_min",
    "exact_train_matches",
    "classifier_val_accuracy",
    "last_loss",
    "train_seconds",
]
LOG_FIELDS = ["seed", "epoch", "loss", "reconstruction_loss", "kl_loss"]


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 25
    batch_size: int = 32
    image_size: int = 64
    latent_dim: int = 64
    learning_rate: float = 8e-4
    kl_weight: float = 0.005
    latent_noise: float = 0.12
    samples_per_requirement: int = 24
    seed: int = 7
    classifier_epochs: int = 10


@dataclass(frozen=True)
class ClassifierEvaluator:
    model: nn.Module
    val_accuracy: float
    device: torch.device


class RequirementImages(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, root: Path, image_size: int) -> None:
        self.items: list[tuple[Path, int]] = []
        for label, requirement in enumerate(REQUIREMENTS):
            folder = requirement_folder(root, requirement)
            self.items.extend((path, label) for path in sorted(folder.glob("*.png")))
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, label = self.items[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, torch.tensor(label, dtype=torch.long)


class ConditionalVAE(nn.Module):
    def __init__(self, n_labels: int, latent_dim: int, image_size: int) -> None:
        super().__init__()
        self.n_labels = n_labels
        self.latent_dim = latent_dim
        self.encoded_size = image_size // 32
        hidden_features = 384 * self.encoded_size * self.encoded_size
        self.encoder = nn.Sequential(
            nn.Conv2d(3 + n_labels, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(256, 384, 4, 2, 1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(hidden_features, latent_dim)
        self.fc_logvar = nn.Linear(hidden_features, latent_dim)
        self.fc_decode = nn.Linear(latent_dim + n_labels, hidden_features)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(384, 256, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
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

    def decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = torch.cat([z, self.one_hot(labels)], dim=1)
        features = self.fc_decode(features).view(-1, 384, self.encoded_size, self.encoded_size)
        return self.decoder(features)

    def forward(
        self, images: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(images, labels)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std
        return self.decode(z, labels), mu, logvar


class RequirementClassifier(nn.Module):
    def __init__(self, n_labels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 192, 4, 2, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(192, n_labels),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


def requirement_folder(root: Path, requirement: str) -> Path:
    return root / "data" / "images" / "celeba-hq" / requirement


def output_dir(root: Path) -> Path:
    return root / "outputs" / "celeba-shared-generator"


def generated_folders(root: Path) -> dict[str, Path]:
    base = output_dir(root) / "generated"
    return {requirement: base / requirement for requirement in REQUIREMENTS}


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
    reconstruction = F.mse_loss(generated, images, reduction="sum") / images.shape[0]
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / images.shape[0]
    return reconstruction + kl_weight * kl, reconstruction, kl


def train_model(root: Path, config: TrainConfig) -> tuple[ConditionalVAE, list[CsvRow], float]:
    seed_everything(config.seed)
    device = torch.device(device_name())
    loader = DataLoader(
        RequirementImages(root, config.image_size),
        batch_size=config.batch_size,
        shuffle=True,
    )
    model = ConditionalVAE(len(REQUIREMENTS), config.latent_dim, config.image_size).to(device)
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
                "seed": str(config.seed),
                "epoch": str(epoch),
                "loss": f"{total_loss / n_batches:.6f}",
                "reconstruction_loss": f"{total_reconstruction / n_batches:.6f}",
                "kl_loss": f"{total_kl / n_batches:.6f}",
            }
        )
        if epoch == 1 or epoch == config.epochs or epoch % 10 == 0:
            print(
                "generator",
                f"seed={config.seed}",
                f"epoch={epoch}/{config.epochs}",
                f"loss={total_loss / n_batches:.3f}",
                flush=True,
            )
    return model, log_rows, time.perf_counter() - start


def split_indices(dataset: RequirementImages) -> tuple[list[int], list[int]]:
    by_label: dict[int, list[int]] = {label: [] for label in range(len(REQUIREMENTS))}
    for index, (_, label) in enumerate(dataset.items):
        by_label[label].append(index)

    train: list[int] = []
    val: list[int] = []
    for indices in by_label.values():
        cutoff = max(1, int(len(indices) * 0.8))
        train.extend(indices[:cutoff])
        val.extend(indices[cutoff:])
    return train, val


def train_classifier(root: Path, config: TrainConfig) -> ClassifierEvaluator:
    seed_everything(config.seed)
    device = torch.device(device_name())
    dataset = RequirementImages(root, config.image_size)
    train_indices, val_indices = split_indices(dataset)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=config.batch_size)
    model = RequirementClassifier(len(REQUIREMENTS)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    for epoch in range(1, config.classifier_epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            loss = F.cross_entropy(model(images), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch == 1 or epoch == config.classifier_epochs:
            print(
                "classifier",
                f"epoch={epoch}/{config.classifier_epochs}",
                f"loss={total_loss / len(train_loader):.3f}",
                flush=True,
            )

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images).argmax(dim=1)
            correct += int((predictions == labels).sum().item())
            total += int(labels.numel())
    accuracy = correct / total if total else 0.0
    print(f"classifier val accuracy={accuracy:.3f}", flush=True)
    return ClassifierEvaluator(model, accuracy, device)


@torch.no_grad()
def training_latents(
    root: Path,
    model: ConditionalVAE,
    config: TrainConfig,
) -> dict[int, torch.Tensor]:
    device = next(model.parameters()).device
    loader = DataLoader(RequirementImages(root, config.image_size), batch_size=config.batch_size)
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


def image_matrix(folder: Path, image_size: int) -> torch.Tensor:
    transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )
    images = []
    for path in sorted(folder.glob("*.png")):
        with Image.open(path) as image:
            images.append(transform(image.convert("RGB")).flatten())
    return torch.stack(images)


def nearest_training_metrics(
    root: Path,
    requirement: str,
    generated: Path,
    image_size: int,
) -> CsvRow:
    generated_images = image_matrix(generated, image_size)
    training_images = image_matrix(requirement_folder(root, requirement), image_size)
    nearest = torch.cdist(generated_images, training_images).min(dim=1).values.pow(2)
    nearest_mse = nearest / generated_images.shape[1]
    return {
        "nearest_train_mse_mean": f"{nearest_mse.mean().item():.6f}",
        "nearest_train_mse_min": f"{nearest_mse.min().item():.6f}",
        "exact_train_matches": str(int((nearest_mse <= 1e-8).sum().item())),
    }


@torch.no_grad()
def classifier_metrics(
    evaluator: ClassifierEvaluator,
    generated: Path,
    requirement: str,
    image_size: int,
) -> CsvRow:
    files = sorted(generated.glob("*.png"))
    top1 = 0
    confidences: list[float] = []
    margins: list[float] = []
    own_index = REQUIREMENTS.index(requirement)
    transform = transforms.Compose(
        [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
    )
    for start in range(0, len(files), 32):
        tensors = []
        for path in files[start : start + 32]:
            with Image.open(path) as image:
                tensors.append(transform(image.convert("RGB")))
        batch = torch.stack(tensors).to(evaluator.device)
        scores = evaluator.model(batch).softmax(dim=1)
        top1 += int((scores.argmax(dim=1) == own_index).sum().item())
        own = scores[:, own_index]
        masked = scores.clone()
        masked[:, own_index] = -999
        margin = own - masked.max(dim=1).values
        confidences.extend(own.tolist())
        margins.extend(margin.tolist())

    return {
        "classifier_top1": f"{top1 / len(files):.6f}",
        "classifier_confidence": f"{mean(confidences):.6f}",
        "classifier_margin": f"{mean(margins):.6f}",
    }


def save_sample_grid(root: Path) -> Path:
    images = []
    for folder in generated_folders(root).values():
        for path in sorted(folder.glob("*.png"))[:6]:
            with Image.open(path) as image:
                images.append(transforms.ToTensor()(image.convert("RGB")))
    grid = make_grid(images, nrow=6, padding=2)
    path = root / "experiments" / "celeba-shared-generator" / "samples.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(grid, path)
    return path


def evaluate_generated(
    root: Path,
    log_rows: list[CsvRow],
    train_seconds: float,
    seed: int,
    config: TrainConfig,
    evaluator: ClassifierEvaluator,
) -> list[CsvRow]:
    last_loss = log_rows[-1]["loss"] if log_rows else ""
    rows = []
    for requirement, folder in generated_folders(root).items():
        novelty = nearest_training_metrics(root, requirement, folder, config.image_size)
        classifier = classifier_metrics(evaluator, folder, requirement, config.image_size)
        rows.append(
            {
                "seed": str(seed),
                "requirement": requirement,
                "n_train_images": str(
                    len(list(requirement_folder(root, requirement).glob("*.png")))
                ),
                "n_generated_images": str(len(list(folder.glob("*.png")))),
                "classifier_top1": classifier["classifier_top1"],
                "classifier_confidence": classifier["classifier_confidence"],
                "classifier_margin": classifier["classifier_margin"],
                "nearest_train_mse_mean": novelty["nearest_train_mse_mean"],
                "nearest_train_mse_min": novelty["nearest_train_mse_min"],
                "exact_train_matches": novelty["exact_train_matches"],
                "classifier_val_accuracy": f"{evaluator.val_accuracy:.6f}",
                "last_loss": last_loss,
                "train_seconds": f"{train_seconds:.2f}",
            }
        )
    return rows


def aggregate_seed_rows(rows: list[CsvRow]) -> list[CsvRow]:
    by_requirement: dict[str, list[CsvRow]] = {}
    for row in rows:
        by_requirement.setdefault(row["requirement"], []).append(row)

    out = []
    for requirement, req_rows in sorted(by_requirement.items()):
        top1 = [float(row["classifier_top1"]) for row in req_rows]
        confidence = [float(row["classifier_confidence"]) for row in req_rows]
        margins = [float(row["classifier_margin"]) for row in req_rows]
        nearest_means = [float(row["nearest_train_mse_mean"]) for row in req_rows]
        nearest_mins = [float(row["nearest_train_mse_min"]) for row in req_rows]
        val_accuracy = [float(row["classifier_val_accuracy"]) for row in req_rows]
        train_seconds = [float(row["train_seconds"]) for row in req_rows]
        out.append(
            {
                "requirement": requirement,
                "n_seeds": str(len(req_rows)),
                "n_train_images": req_rows[0]["n_train_images"],
                "n_generated_images_per_seed": req_rows[0]["n_generated_images"],
                "classifier_top1_mean": f"{mean(top1):.6f}",
                "classifier_top1_std": f"{stdev(top1):.6f}" if len(top1) > 1 else "0.000000",
                "classifier_confidence_mean": f"{mean(confidence):.6f}",
                "classifier_margin_mean": f"{mean(margins):.6f}",
                "nearest_train_mse_mean": f"{mean(nearest_means):.6f}",
                "nearest_train_mse_min": f"{min(nearest_mins):.6f}",
                "exact_train_matches_total": str(
                    sum(int(row["exact_train_matches"]) for row in req_rows)
                ),
                "classifier_val_accuracy": f"{mean(val_accuracy):.6f}",
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
    out_dir = root / "experiments" / "celeba-shared-generator"
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
    mean_top1 = mean(float(row["classifier_top1_mean"]) for row in rows)
    mean_val_accuracy = mean(float(row["classifier_val_accuracy"]) for row in rows)
    exact_matches = sum(int(row["exact_train_matches_total"]) for row in rows)
    hardest = min(rows, key=lambda row: float(row["classifier_top1_mean"]))
    lines = [
        "# CelebA-HQ Shared Generator Summary",
        "",
        "**Question:** Does the cheap shared-generator idea still look plausible on a harder "
        "natural-image dataset?",
        "",
        "**Method:** Train one conditional VAE on copied CelebA-HQ RBT4DNN LoRA images and "
        "evaluate generated images with a small requirement classifier plus nearest-train "
        "image checks.",
        "",
        f"**Result:** Classifier validation accuracy is {mean_val_accuracy:.3f}; generated-image "
        f"classifier top-1 alignment is {mean_top1:.3f}.",
        "",
        "**Limitation:** This is exploratory and not a replacement for the paper's "
        "attribute-classifier pass rate. The classifier itself is weak, so the result is a "
        "caution signal rather than a final benchmark.",
        "",
        f"Exact generated/training image matches: {exact_matches}.",
        f"Hardest requirement by classifier top-1: {hardest['requirement']} "
        f"({hardest['classifier_top1_mean']}).",
        f"Sample grid: `{sample_grid.relative_to(sample_grid.parents[2])}`.",
        "",
    ]
    lines += [
        f"- {row['requirement']}: classifier top-1 {row['classifier_top1_mean']} "
        f"(std {row['classifier_top1_std']}, margin {row['classifier_margin_mean']})"
        for row in rows
    ]
    return "\n".join(lines) + "\n"


def train_and_evaluate(
    root: Path | None = None,
    config: TrainConfig | None = None,
    seeds: list[int] | None = None,
) -> list[Path]:
    root = find_repo_root(root)
    validate_image_corpus(root, ["celeba-hq"])
    config = config or TrainConfig()
    seeds = seeds or DEFAULT_SEEDS
    seed_rows: list[CsvRow] = []
    all_log_rows: list[CsvRow] = []
    sample_grid: Path | None = None
    print(f"device={device_name()}", flush=True)
    evaluator = train_classifier(root, config)
    for seed in seeds:
        print(f"training generator seed={seed}", flush=True)
        seed_config = replace(config, seed=seed)
        model, log_rows, train_seconds = train_model(root, seed_config)
        print(f"generating images seed={seed}", flush=True)
        generate_images(root, model, seed_config)
        print(f"evaluating images seed={seed}", flush=True)
        seed_rows.extend(
            evaluate_generated(root, log_rows, train_seconds, seed, seed_config, evaluator)
        )
        all_log_rows.extend(log_rows)
        if sample_grid is None:
            sample_grid = save_sample_grid(root)

    if sample_grid is None:
        raise RuntimeError("No CelebA shared-generator runs were executed.")
    return write_outputs(root, aggregate_seed_rows(seed_rows), seed_rows, all_log_rows, sample_grid)
