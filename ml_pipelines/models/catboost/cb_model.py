
from pathlib import Path

import polars as pl
from catboost import CatBoostClassifier, Pool

from ml_pipelines.models.config import CatBoostConfig
from config.settings import get_settings


cb_config = CatBoostConfig()


CAT_DTYPES = (pl.Utf8, pl.Categorical, pl.Enum, pl.Boolean)


def make_pool(
    X: pl.DataFrame,
    y: pl.Series | None = None,
) -> Pool:
    cat_cols = [
        name
        for name, dtype in zip(X.columns, X.dtypes)
        if dtype in CAT_DTYPES
    ]
    return Pool(X, y, cat_features=cat_cols)


def fit_cb_model(
    tr_X: pl.DataFrame,
    tr_y: pl.Series,
    val_X: pl.DataFrame,
    val_y: pl.Series,
    mod_depth: int = cb_config.depth,
    lr: float = cb_config.learning_rate,
    iters: int = cb_config.iterations,
) -> tuple[CatBoostClassifier, Pool]:
    if tr_X.columns != val_X.columns:
        raise ValueError("Train and adv_val columns differ.")

    model = cb_config.make_model(depth=mod_depth, lr=lr, iter=iters)

    tr_pool = make_pool(tr_X, tr_y)
    val_pool = make_pool(val_X, val_y)

    model.fit(
        tr_pool,
        eval_set=val_pool,
        use_best_model=True,
    )
    return model, val_pool


def parameter_sweep(
    tr_X: pl.DataFrame,
    tr_y: pl.Series,
    val_X: pl.DataFrame,
    val_y: pl.Series,
    parameter: str,
    values: list[int | float],
) -> pl.DataFrame:
    results = []

    for value in values:
        kwargs = {
            "mod_depth": value
        } if parameter == "depth" else {
            "lr": value
        }

        model, _ = fit_cb_model(
            tr_X,
            tr_y,
            val_X,
            val_y,
            **kwargs,
        )

        results.append({
            parameter: value,
            "best_iteration": model.get_best_iteration(),
            "best_score": get_validation_auc(model),
        })

    return pl.DataFrame(results).sort(
        "best_score",
        descending=True,
    )


def depth_sweep(
    tr_X: pl.DataFrame,
    tr_y: pl.Series,
    val_X: pl.DataFrame,
    val_y: pl.Series,
) -> pl.DataFrame:
    return parameter_sweep(
        tr_X,
        tr_y,
        val_X,
        val_y,
        parameter="depth",
        values=cb_config.sweep_depths,
    )


def lr_sweep(
    tr_X: pl.DataFrame,
    tr_y: pl.Series,
    val_X: pl.DataFrame,
    val_y: pl.Series,
) -> pl.DataFrame:
    return parameter_sweep(
        tr_X,
        tr_y,
        val_X,
        val_y,
        parameter="lr",
        values=cb_config.sweep_learning_rates,
    )


def save_model(
    model: CatBoostClassifier,
    file_name: str,
    path: str | Path | None = None,
) -> None:
    path = path or get_settings().io.model_dir
    model_path = Path(path) / file_name
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))


def get_validation_auc(model: CatBoostClassifier) -> float:
    best_score = getattr(model, "best_score_", {}) or {}
    validation_scores = (
        best_score.get("validation", {})
        or best_score.get("validation_0", {})
        or {}
    )
    return float(validation_scores.get("AUC", 0.0))
