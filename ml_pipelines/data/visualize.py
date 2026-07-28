import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from datasets import Dataset
from config.train_data import TrainDataConfig


data_cfg = TrainDataConfig()

def correlation_heatmap(
    df: pl.DataFrame,
    exclude_cols: list[str] | None = None,
    title: str = "Feature Correlation Heatmap",
) -> None:
    if exclude_cols is None:
        exclude_cols = []

    numeric_df = df.select(
        [
            col for col, dtype in df.schema.items()
            if col not in exclude_cols and dtype.is_numeric()
        ]
    )

    if numeric_df.width == 0:
        raise ValueError("No numeric columns found after exclusions.")

    numeric_df = numeric_df.drop_nulls()

    cols = numeric_df.columns
    arr = numeric_df.to_numpy()

    corr_matrix = np.corrcoef(arr, rowvar=False)

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(corr_matrix, aspect="auto", vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, label="Correlation")

    ax.set_title(title)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_yticks(np.arange(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right")
    ax.set_yticklabels(cols)

    plt.tight_layout()
    plt.show()


def graph_normal_distro(ds: dict[str, Dataset], label: str = data_cfg.target_column, title: str = "Normal Distribution of Labels") -> None:
    fig, ax = plt.subplots()

    # Get all values so every curve uses the same x-axis
    all_values = np.concatenate([
        np.asarray(split[label], dtype=float)
        for split in ds.values()
    ])

    x = np.linspace(all_values.min(), all_values.max(), 500)

    for name, split in ds.items():
        values = np.asarray(split[label], dtype=float)

        mean = values.mean()
        std = values.std(ddof=1)

        y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - mean) / std) ** 2
        )

        ax.plot(x, y, label=f"{name} μ={mean:.3f}, σ={std:.3f}")

    ax.set_title(title or f"Normal distribution of {label}")
    ax.set_xlabel(label)
    ax.set_ylabel("Density")
    ax.legend()
    plt.tight_layout()
    plt.show()
