"""
Step 9: LSTM in-play training pipeline (PyTorch)

This script trains an LSTM classifier for in-play final result prediction.
Instead of treating each checkpoint as an independent row, it builds
ordered sequences for each fixture:

- 10'  -> sequence length 1
- 20'  -> sequence length 2
- ...
- 90'  -> sequence length N

Main design choices:
- Temporal holdout by fixture (last 20% by date as test)
- Validation split on the training fixtures only
- LSTM over checkpoint sequences
- Standardization fitted on training sequences only
- Class-weighted CrossEntropyLoss
- Early stopping on validation macro-F1

Outputs:
- models/inplay_model_lstm.pt
- reports/inplay_report_lstm.txt
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset

DATA_DIR = Path("data/processed")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
LABEL_ORDER = ["A", "D", "H"]
DEFAULT_CHECKPOINT_MINUTES = [10, 20, 30, 40, 45, 50, 60, 70, 80, 90]
PREMATCH_SEQ_LEN = 32
PREMATCH_MAX_H2H = 16
PREMATCH_MAX_TEAM_FORM = 10
PREMATCH_SEQUENCE_FEATURES = [
    "obs_flag",
    "source_h2h",
    "source_home_form",
    "source_away_form",
    "result_code",
    "goal_diff",
    "goals_for",
    "goals_against",
    "was_home",
    "days_ago_scaled",
]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    try:
        major, minor = torch.cuda.get_device_capability(0)
        device_tag = f"sm_{major}{minor}"
        supported_arches = set(torch.cuda.get_arch_list())
        if supported_arches and device_tag not in supported_arches:
            print(
                f"CUDA device architecture {device_tag} is not supported by the installed PyTorch build; "
                "falling back to CPU."
            )
            return torch.device("cpu")
    except Exception as exc:
        print(f"CUDA capability check failed ({exc}); falling back to CPU.")
        return torch.device("cpu")

    return torch.device("cuda")


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def _select_numeric_features(df: pd.DataFrame, exclude_cols: set[str]) -> List[str]:
    candidate_cols = [c for c in df.columns if c not in exclude_cols]
    return [c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])]


def _get_inplay_checkpoint_minutes(df: pd.DataFrame) -> List[int]:
    if "minute" not in df.columns:
        return DEFAULT_CHECKPOINT_MINUTES
    minute_values = sorted(df["minute"].dropna().astype(int).unique().tolist())
    return minute_values or DEFAULT_CHECKPOINT_MINUTES


def _temporal_holdout_by_fixture(
    df: pd.DataFrame,
    date_col: str,
    group_col: str,
    test_frac: float = 0.2,
) -> Tuple[pd.Series, pd.Series]:
    temp = df.copy()
    temp[date_col] = _to_datetime(temp[date_col])

    group_df = (
        temp[[group_col, date_col]]
        .dropna(subset=[date_col])
        .sort_values(date_col)
        .drop_duplicates(subset=[group_col], keep="first")
    )
    n_test_groups = max(1, int(len(group_df) * test_frac))
    test_groups = set(group_df.tail(n_test_groups)[group_col].tolist())
    test_mask = df[group_col].isin(test_groups)
    train_mask = ~test_mask
    return train_mask, test_mask


def _train_val_split_fixture_ids(df_train: pd.DataFrame, val_frac: float = 0.15) -> Tuple[set[int], set[int]]:
    fixture_df = (
        df_train[["fixture_id", "date"]]
        .assign(date=_to_datetime(df_train["date"]))
        .dropna(subset=["date"])
        .sort_values("date")
        .drop_duplicates(subset=["fixture_id"], keep="first")
    )

    split = int(len(fixture_df) * (1.0 - val_frac))
    train_ids = set(fixture_df.iloc[:split]["fixture_id"].tolist())
    val_ids = set(fixture_df.iloc[split:]["fixture_id"].tolist())
    return train_ids, val_ids


def _standardize_sequence_arrays(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_flat = X_train.reshape(-1, X_train.shape[-1])
    mean = train_flat.mean(axis=0)
    std = train_flat.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    X_train_s = (X_train - mean) / std
    X_val_s = (X_val - mean) / std
    X_test_s = (X_test - mean) / std
    return X_train_s, X_val_s, X_test_s, mean, std


def _standardize_tabular_arrays(
    X_train: Optional[np.ndarray],
    X_val: Optional[np.ndarray],
    X_test: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if X_train is None or X_val is None or X_test is None:
        return None, None, None, None, None

    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std, mean, std


def _prematch_obs_vector(
    result_code: float,
    goal_diff: float,
    goals_for: float,
    goals_against: float,
    was_home: float,
    days_ago_scaled: float,
    source_h2h: float,
    source_home_form: float,
    source_away_form: float,
) -> np.ndarray:
    return np.array(
        [
            1.0,
            source_h2h,
            source_home_form,
            source_away_form,
            result_code,
            goal_diff,
            goals_for,
            goals_against,
            was_home,
            days_ago_scaled,
        ],
        dtype=np.float32,
    )

class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, lengths: np.ndarray, y: np.ndarray, X_aux: Optional[np.ndarray] = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
        self.X_aux = torch.tensor(X_aux, dtype=torch.float32) if X_aux is not None else None

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        aux = self.X_aux[idx] if self.X_aux is not None else torch.empty(0, dtype=torch.float32)
        return self.X[idx], self.lengths[idx], aux, self.y[idx]


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        static_input_dim: int = 0,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 3,
        bidirectional: bool = True,
        fuse_last_hidden: bool = True,
        lstm_dropout: float = 0.20,
        head_dropout: float = 0.25,
        attention_dropout: float = 0.10,
        static_hidden_dim: int = 64,
        attention_use_static: bool = False,
        use_dual_head_fusion: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional
        self.fuse_last_hidden = fuse_last_hidden
        self.static_input_dim = static_input_dim
        self.attention_use_static = attention_use_static and static_input_dim > 0
        self.use_dual_head_fusion = use_dual_head_fusion and static_input_dim > 0
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=lstm_dropout if num_layers > 1 else 0.0,
        )
        head_input_dim = hidden_dim * (2 if bidirectional else 1)
        attention_input_dim = head_input_dim + (static_hidden_dim if self.attention_use_static else 0)
        self.attention = nn.Sequential(
            nn.Linear(attention_input_dim, head_input_dim),
            nn.Tanh(),
            nn.Dropout(attention_dropout),
            nn.Linear(head_input_dim, 1),
        )
        if static_input_dim > 0:
            self.static_encoder = nn.Sequential(
                nn.Linear(static_input_dim, static_hidden_dim),
                nn.LayerNorm(static_hidden_dim),
                nn.ReLU(),
                nn.Dropout(head_dropout),
            )
        else:
            self.static_encoder = None
        fused_head_input_dim = head_input_dim * 2 if fuse_last_hidden else head_input_dim
        if self.static_encoder is not None:
            fused_head_input_dim += static_hidden_dim
        seq_feature_dim = head_input_dim * 2 if fuse_last_hidden else head_input_dim
        if self.use_dual_head_fusion:
            self.sequence_head = nn.Sequential(
                nn.Linear(seq_feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(head_dropout),
                nn.Linear(hidden_dim, num_classes),
            )
            self.static_head = nn.Sequential(
                nn.Linear(static_hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(head_dropout),
                nn.Linear(hidden_dim // 2, num_classes),
            )
            self.fusion_gate = nn.Sequential(
                nn.Linear(fused_head_input_dim, num_classes),
                nn.Sigmoid(),
            )
            self.head = None
        else:
            self.sequence_head = None
            self.static_head = None
            self.fusion_gate = None
            self.head = nn.Sequential(
                nn.Linear(fused_head_input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(head_dropout),
                nn.Linear(hidden_dim, num_classes),
            )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, x_aux: Optional[torch.Tensor] = None) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, (hidden, _) = self.lstm(packed)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True, total_length=x.size(1))

        static_features = None
        if self.static_encoder is not None and x_aux is not None and x_aux.numel() > 0:
            static_features = self.static_encoder(x_aux)

        if self.attention_use_static and static_features is not None:
            attn_input = torch.cat(
                (output, static_features.unsqueeze(1).expand(-1, output.size(1), -1)),
                dim=2,
            )
        else:
            attn_input = output

        attn_scores = self.attention(attn_input).squeeze(-1)
        mask = torch.arange(output.size(1), device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)
        attn_scores = attn_scores.masked_fill(mask, float("-inf"))
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(output * attn_weights.unsqueeze(-1), dim=1)
        if self.bidirectional:
            last_hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            last_hidden = hidden[-1]
        if self.fuse_last_hidden:
            seq_features = torch.cat((context, last_hidden), dim=1)
        else:
            seq_features = context
        features = seq_features
        if static_features is not None:
            features = torch.cat((features, static_features), dim=1)
        if self.use_dual_head_fusion and static_features is not None:
            seq_logits = self.sequence_head(seq_features)
            static_logits = self.static_head(static_features)
            gate = self.fusion_gate(features)
            return gate * seq_logits + (1.0 - gate) * static_logits
        return self.head(features)


@dataclass
class TrainConfig:
    epochs: int = 120
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 16
    hidden_dim: int = 128
    num_layers: int = 2
    bidirectional: bool = True
    fuse_last_hidden: bool = True
    grad_clip: float = 1.0
    warmup_epochs: int = 0
    lr_scheduler_patience: int = 4
    lr_scheduler_factor: float = 0.5
    label_smoothing: float = 0.0
    class_weight_power: float = 1.0
    draw_boost: float = 1.0
    lstm_dropout: float = 0.20
    head_dropout: float = 0.25
    attention_dropout: float = 0.10
    static_hidden_dim: int = 0
    attention_use_static: bool = False
    enable_val_bias_search: bool = False
    use_dual_head_fusion: bool = False


def _build_task_config(task_name: str) -> TrainConfig:
    if task_name == "pre-match":
        return TrainConfig(
            epochs=200,
            batch_size=96,
            lr=1.7e-4,
            weight_decay=3e-4,
            patience=26,
            hidden_dim=160,
            num_layers=1,
            bidirectional=True,
            fuse_last_hidden=False,
            grad_clip=0.6,
            warmup_epochs=6,
            lr_scheduler_patience=6,
            lr_scheduler_factor=0.5,
            label_smoothing=0.02,
            class_weight_power=1.0,
            draw_boost=1.04,
            lstm_dropout=0.0,
            head_dropout=0.15,
            attention_dropout=0.08,
            static_hidden_dim=64,
            attention_use_static=True,
            enable_val_bias_search=True,
            use_dual_head_fusion=True,
        )

    return TrainConfig(
        epochs=180,
        batch_size=128,
        lr=3.0e-4,
        weight_decay=2e-4,
        patience=18,
        hidden_dim=144,
        num_layers=1,
        bidirectional=True,
        fuse_last_hidden=False,
        grad_clip=0.6,
        warmup_epochs=5,
        lr_scheduler_patience=5,
        lr_scheduler_factor=0.5,
        label_smoothing=0.01,
        class_weight_power=1.0,
        draw_boost=1.0,
        lstm_dropout=0.0,
        head_dropout=0.15,
        attention_dropout=0.02,
        static_hidden_dim=0,
        attention_use_static=False,
        enable_val_bias_search=False,
        use_dual_head_fusion=False,
    )


def build_sequence_samples(
    df: pd.DataFrame,
    feature_cols: List[str],
    fixture_ids: set[int],
    checkpoint_minutes: List[int],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not fixture_ids:
        return (
            np.zeros((0, len(checkpoint_minutes), len(feature_cols)), dtype=np.float32),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )

    samples_X = []
    samples_len = []
    samples_y = []
    samples_minute = []
    label_to_idx = {label: idx for idx, label in enumerate(LABEL_ORDER)}

    for fixture_id in sorted(fixture_ids):
        fixture_df = df[df["fixture_id"] == fixture_id].sort_values("minute")
        if fixture_df.empty:
            continue

        feature_matrix = fixture_df[feature_cols].fillna(0).values.astype(np.float32)
        minute_values = fixture_df["minute"].astype(int).tolist()
        y_value = label_to_idx[str(fixture_df.iloc[0]["result"])]

        for idx, minute in enumerate(minute_values):
            seq = np.zeros((len(checkpoint_minutes), len(feature_cols)), dtype=np.float32)
            seq[: idx + 1] = feature_matrix[: idx + 1]
            samples_X.append(seq)
            samples_len.append(idx + 1)
            samples_y.append(y_value)
            samples_minute.append(minute)

    return (
        np.asarray(samples_X, dtype=np.float32),
        np.asarray(samples_len, dtype=np.int64),
        np.asarray(samples_y, dtype=np.int64),
        np.asarray(samples_minute, dtype=np.int64),
    )


def build_prematch_h2h_sequence_samples(
    df: pd.DataFrame,
    fixture_ids: set[int],
    static_feature_cols: Optional[List[str]] = None,
    max_seq_len: int = PREMATCH_SEQ_LEN,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    label_to_idx = {label: idx for idx, label in enumerate(LABEL_ORDER)}
    df_sorted = df.copy()
    df_sorted["date"] = _to_datetime(df_sorted["date"])
    df_sorted = df_sorted.sort_values("date")

    pair_history: defaultdict[tuple[str, str], list[dict]] = defaultdict(list)
    team_history: defaultdict[str, list[dict]] = defaultdict(list)
    samples_X = []
    samples_len = []
    samples_y = []
    samples_marker = []
    samples_static = [] if static_feature_cols is not None else None

    for row in df_sorted.itertuples(index=False):
        home_team = str(row.home_team)
        away_team = str(row.away_team)
        pair_key = tuple(sorted((home_team, away_team)))

        if int(row.fixture_id) in fixture_ids:
            seq = np.zeros((max_seq_len, len(PREMATCH_SEQUENCE_FEATURES)), dtype=np.float32)
            current_date = row.date
            candidates: list[tuple[pd.Timestamp, np.ndarray]] = []

            for hist in pair_history[pair_key][-PREMATCH_MAX_H2H:]:
                if hist["home_team"] == home_team and hist["away_team"] == away_team:
                    goals_for = hist["goals_home"]
                    goals_against = hist["goals_away"]
                    was_home = 1.0
                else:
                    goals_for = hist["goals_away"]
                    goals_against = hist["goals_home"]
                    was_home = 0.0

                goal_diff = goals_for - goals_against
                result_code = 1.0 if goal_diff > 0 else (-1.0 if goal_diff < 0 else 0.0)
                days_ago = (current_date - hist["date"]).total_seconds() / 86400.0
                candidates.append(
                    (
                        hist["date"],
                        _prematch_obs_vector(
                            result_code=result_code,
                            goal_diff=float(goal_diff),
                            goals_for=float(goals_for),
                            goals_against=float(goals_against),
                            was_home=was_home,
                            days_ago_scaled=float(days_ago / 365.0),
                            source_h2h=1.0,
                            source_home_form=0.0,
                            source_away_form=0.0,
                        ),
                    )
                )

            for hist in team_history[home_team][-PREMATCH_MAX_TEAM_FORM:]:
                days_ago = (current_date - hist["date"]).total_seconds() / 86400.0
                candidates.append(
                    (
                        hist["date"],
                        _prematch_obs_vector(
                            result_code=float(hist["result_code"]),
                            goal_diff=float(hist["goal_diff"]),
                            goals_for=float(hist["goals_for"]),
                            goals_against=float(hist["goals_against"]),
                            was_home=float(hist["was_home"]),
                            days_ago_scaled=float(days_ago / 365.0),
                            source_h2h=0.0,
                            source_home_form=1.0,
                            source_away_form=0.0,
                        ),
                    )
                )

            for hist in team_history[away_team][-PREMATCH_MAX_TEAM_FORM:]:
                days_ago = (current_date - hist["date"]).total_seconds() / 86400.0
                candidates.append(
                    (
                        hist["date"],
                        _prematch_obs_vector(
                            result_code=float(hist["result_code"]),
                            goal_diff=float(hist["goal_diff"]),
                            goals_for=float(hist["goals_for"]),
                            goals_against=float(hist["goals_against"]),
                            was_home=float(hist["was_home"]),
                            days_ago_scaled=float(days_ago / 365.0),
                            source_h2h=0.0,
                            source_home_form=0.0,
                            source_away_form=1.0,
                        ),
                    )
                )

            recent = sorted(candidates, key=lambda item: item[0])[-max_seq_len:]
            if recent:
                for idx, (_, obs) in enumerate(recent):
                    seq[idx] = obs
                seq_len = len(recent)
            else:
                seq[0] = np.array([0.0] * len(PREMATCH_SEQUENCE_FEATURES), dtype=np.float32)
                seq_len = 1

            samples_X.append(seq)
            samples_len.append(seq_len)
            samples_y.append(label_to_idx[str(row.result)])
            samples_marker.append(0)
            if samples_static is not None:
                static_values = np.array([float(getattr(row, col)) for col in static_feature_cols], dtype=np.float32)
                samples_static.append(static_values)

        pair_history[pair_key].append(
            {
                "date": row.date,
                "home_team": home_team,
                "away_team": away_team,
                "goals_home": int(row.goals_home),
                "goals_away": int(row.goals_away),
            }
        )
        team_history[home_team].append(
            {
                "date": row.date,
                "goals_for": int(row.goals_home),
                "goals_against": int(row.goals_away),
                "goal_diff": int(row.goals_home) - int(row.goals_away),
                "result_code": 1.0 if row.result == "H" else (0.0 if row.result == "D" else -1.0),
                "was_home": 1.0,
            }
        )
        team_history[away_team].append(
            {
                "date": row.date,
                "goals_for": int(row.goals_away),
                "goals_against": int(row.goals_home),
                "goal_diff": int(row.goals_away) - int(row.goals_home),
                "result_code": 1.0 if row.result == "A" else (0.0 if row.result == "D" else -1.0),
                "was_home": 0.0,
            }
        )

    static_array = np.asarray(samples_static, dtype=np.float32) if samples_static is not None else None

    return (
        np.asarray(samples_X, dtype=np.float32),
        np.asarray(samples_len, dtype=np.int64),
        np.asarray(samples_y, dtype=np.int64),
        np.asarray(samples_marker, dtype=np.int64),
        static_array,
    )


def _macro_f1_from_logits(logits: torch.Tensor, y_true: np.ndarray) -> float:
    pred = logits.argmax(dim=1).cpu().numpy()
    return float(f1_score(y_true, pred, average="macro"))


def _predict_probs(
    model: nn.Module,
    X: np.ndarray,
    lengths: np.ndarray,
    device: torch.device,
    batch_size: int,
    X_aux: Optional[np.ndarray] = None,
) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = start + batch_size
            xb = torch.tensor(X[start:end], dtype=torch.float32, device=device)
            lb = torch.tensor(lengths[start:end], dtype=torch.long, device=device)
            auxb = None
            if X_aux is not None:
                auxb = torch.tensor(X_aux[start:end], dtype=torch.float32, device=device)
            logits = model(xb, lb, auxb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            outputs.append(probs)
    return np.vstack(outputs) if outputs else np.zeros((0, len(LABEL_ORDER)), dtype=np.float32)


def _predict_with_class_bias(probs: np.ndarray, class_bias: Optional[np.ndarray]) -> np.ndarray:
    if class_bias is None:
        return probs.argmax(axis=1)

    adjusted_scores = np.log(np.clip(probs, 1e-8, 1.0)) + class_bias.reshape(1, -1)
    return adjusted_scores.argmax(axis=1)


def _draw_recall(y_true: np.ndarray, pred_idx: np.ndarray) -> float:
    draw_idx = LABEL_ORDER.index("D")
    mask = y_true == draw_idx
    if not np.any(mask):
        return 0.0
    return float(np.mean(pred_idx[mask] == draw_idx))


def _draw_f1(y_true: np.ndarray, pred_idx: np.ndarray) -> float:
    draw_idx = LABEL_ORDER.index("D")
    true_draw = y_true == draw_idx
    pred_draw = pred_idx == draw_idx
    tp = float(np.sum(true_draw & pred_draw))
    fp = float(np.sum(~true_draw & pred_draw))
    fn = float(np.sum(true_draw & ~pred_draw))
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    if precision + recall <= 1e-12:
        return 0.0
    return float(2.0 * precision * recall / (precision + recall))


def _score_bias_candidate(
    probs: np.ndarray,
    y_true: np.ndarray,
    candidate_bias: np.ndarray,
    min_allowed_acc: float,
) -> tuple[float, float, float, float, np.ndarray] | None:
    pred_idx = _predict_with_class_bias(probs, candidate_bias)
    cand_acc = float(accuracy_score(y_true, pred_idx))
    if cand_acc < min_allowed_acc:
        return None
    cand_f1 = float(f1_score(y_true, pred_idx, average="macro"))
    cand_draw_recall = _draw_recall(y_true, pred_idx)
    cand_draw_f1 = _draw_f1(y_true, pred_idx)
    cand_score = cand_f1 + 0.28 * cand_draw_f1 + 0.08 * cand_draw_recall + 0.03 * cand_acc
    return cand_score, cand_f1, cand_acc, cand_draw_recall, pred_idx


def _search_class_bias(probs: np.ndarray, y_true: np.ndarray) -> tuple[np.ndarray, float, float, float]:
    if len(probs) < 80:
        baseline_pred = probs.argmax(axis=1)
        return (
            np.zeros(len(LABEL_ORDER), dtype=np.float32),
            float(f1_score(y_true, baseline_pred, average="macro")),
            float(accuracy_score(y_true, baseline_pred)),
            _draw_recall(y_true, baseline_pred),
        )

    rng = np.random.default_rng(SEED)
    shuffled_idx = rng.permutation(len(probs))
    split_at = max(int(len(probs) * 0.55), 1)
    search_idx = shuffled_idx[:split_at]
    eval_idx = shuffled_idx[split_at:]
    if len(eval_idx) == 0:
        eval_idx = search_idx

    probs_search = probs[search_idx]
    y_search = y_true[search_idx]
    probs_eval = probs[eval_idx]
    y_eval = y_true[eval_idx]

    baseline_search_pred = probs_search.argmax(axis=1)
    baseline_eval_pred = probs_eval.argmax(axis=1)
    search_min_allowed_acc = float(accuracy_score(y_search, baseline_search_pred)) - 0.012
    eval_min_allowed_acc = float(accuracy_score(y_eval, baseline_eval_pred)) - 0.008

    candidate_records: list[tuple[float, np.ndarray]] = []

    def maybe_record(candidate_bias: np.ndarray) -> None:
        scored = _score_bias_candidate(probs_search, y_search, candidate_bias, search_min_allowed_acc)
        if scored is None:
            return
        cand_score, cand_f1, cand_acc, cand_draw_recall, pred_idx = scored
        cand_draw_f1 = _draw_f1(y_search, pred_idx)
        tie_break = cand_score + 1e-4 * cand_f1 + 1e-4 * cand_draw_f1 + 1e-5 * cand_acc + 1e-5 * cand_draw_recall
        candidate_records.append((tie_break, candidate_bias.copy()))

    coarse_away = np.linspace(-0.12, 0.12, 5)
    coarse_draw = np.linspace(0.0, 0.75, 11)
    coarse_home = np.linspace(-0.35, 0.05, 9)
    for away_bias in coarse_away:
        for draw_bias in coarse_draw:
            for home_bias in coarse_home:
                maybe_record(np.array([away_bias, draw_bias, home_bias], dtype=np.float32))

    if not candidate_records:
        baseline_pred = probs.argmax(axis=1)
        return (
            np.zeros(len(LABEL_ORDER), dtype=np.float32),
            float(f1_score(y_true, baseline_pred, average="macro")),
            float(accuracy_score(y_true, baseline_pred)),
            _draw_recall(y_true, baseline_pred),
        )

    candidate_records.sort(key=lambda item: item[0], reverse=True)
    top_candidates = [bias for _, bias in candidate_records[:12]]
    refined_candidates: list[np.ndarray] = []
    for base_bias in top_candidates:
        refined_candidates.append(base_bias)
        for step in [0.06, 0.03, 0.015]:
            for delta_away in [-step, 0.0, step]:
                for delta_draw in [-step, 0.0, step]:
                    for delta_home in [-step, 0.0, step]:
                        if delta_away == 0.0 and delta_draw == 0.0 and delta_home == 0.0:
                            continue
                        candidate_bias = base_bias + np.array([delta_away, delta_draw, delta_home], dtype=np.float32)
                        candidate_bias[0] = np.clip(candidate_bias[0], -0.20, 0.20)
                        candidate_bias[1] = np.clip(candidate_bias[1], 0.0, 0.90)
                        candidate_bias[2] = np.clip(candidate_bias[2], -0.45, 0.10)
                        refined_candidates.append(candidate_bias)

    best_bias = np.zeros(len(LABEL_ORDER), dtype=np.float32)
    best_eval_score = float("-inf")
    best_eval_f1 = float("-inf")
    best_eval_acc = float("-inf")
    best_eval_draw_recall = float("-inf")
    best_eval_draw_f1 = float("-inf")

    for candidate_bias in refined_candidates:
        scored = _score_bias_candidate(probs_eval, y_eval, candidate_bias, eval_min_allowed_acc)
        if scored is None:
            continue
        cand_score, cand_f1, cand_acc, cand_draw_recall, pred_idx = scored
        cand_draw_f1 = _draw_f1(y_eval, pred_idx)
        if cand_score > best_eval_score + 1e-9 or (
            abs(cand_score - best_eval_score) <= 1e-9
            and (
                cand_draw_f1 > best_eval_draw_f1 + 1e-9
                or cand_draw_recall > best_eval_draw_recall + 1e-9
                or cand_f1 > best_eval_f1 + 1e-9
                or cand_acc > best_eval_acc + 1e-9
            )
        ):
            best_eval_score = cand_score
            best_eval_f1 = cand_f1
            best_eval_acc = cand_acc
            best_eval_draw_recall = cand_draw_recall
            best_eval_draw_f1 = cand_draw_f1
            best_bias = candidate_bias.copy()

    full_pred = _predict_with_class_bias(probs, best_bias)
    return (
        best_bias,
        float(f1_score(y_true, full_pred, average="macro")),
        float(accuracy_score(y_true, full_pred)),
        _draw_recall(y_true, full_pred),
    )


def _format_classification_report_without_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Away Win", "Draw", "Home Win"],
        output_dict=True,
        zero_division=0,
    )

    row_names = ["Away Win", "Draw", "Home Win", "macro avg", "weighted avg"]
    lines = [f"{'':14s}{'precision':>10s}{'recall':>10s}{'f1-score':>10s}{'support':>10s}", ""]
    for name in row_names:
        row = report[name]
        lines.append(
            f"{name:14s}{row['precision']:10.2f}{row['recall']:10.2f}{row['f1-score']:10.2f}{int(row['support']):10d}"
        )
    return "\n".join(lines)


def train_lstm_task(
    task_name: str,
    data_file: Path,
    feature_names: List[str],
    X_tr: np.ndarray,
    len_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    len_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    len_test: np.ndarray,
    y_test: np.ndarray,
    marker_test: np.ndarray,
    marker_name: str,
    marker_order: List[int],
    model_path: Path,
    report_path: Path,
    cfg: TrainConfig,
    report_lines: List[str],
    static_feature_names: Optional[List[str]] = None,
    X_tr_static: Optional[np.ndarray] = None,
    X_val_static: Optional[np.ndarray] = None,
    X_test_static: Optional[np.ndarray] = None,
) -> None:
    X_tr, X_val, X_test, mean, std = _standardize_sequence_arrays(X_tr, X_val, X_test)
    X_tr_static, X_val_static, X_test_static, static_mean, static_std = _standardize_tabular_arrays(
        X_tr_static,
        X_val_static,
        X_test_static,
    )

    device = _select_device()
    print(f"Using device: {device}")
    print(f"Samples - train: {len(X_tr)}, val: {len(X_val)}, test: {len(X_test)}")
    print(f"Feature count: {len(feature_names)} | Max sequence length: {X_tr.shape[1]}")
    if static_feature_names:
        print(f"Static feature count: {len(static_feature_names)}")

    train_loader = DataLoader(
        SequenceDataset(X_tr, len_tr, y_tr, X_tr_static),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        SequenceDataset(X_val, len_val, y_val, X_val_static),
        batch_size=cfg.batch_size,
        shuffle=False,
    )

    model = LSTMClassifier(
        input_dim=len(feature_names),
        static_input_dim=0 if not static_feature_names else len(static_feature_names),
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        num_classes=len(LABEL_ORDER),
        bidirectional=cfg.bidirectional,
        fuse_last_hidden=cfg.fuse_last_hidden,
        lstm_dropout=cfg.lstm_dropout,
        head_dropout=cfg.head_dropout,
        attention_dropout=cfg.attention_dropout,
        static_hidden_dim=cfg.static_hidden_dim if static_feature_names else 0,
        attention_use_static=cfg.attention_use_static,
        use_dual_head_fusion=cfg.use_dual_head_fusion,
    ).to(device)

    class_counts = np.bincount(y_tr, minlength=len(LABEL_ORDER)).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = np.power(class_weights, cfg.class_weight_power)
    if len(class_weights) >= 2:
        class_weights[1] *= cfg.draw_boost
    class_weights = class_weights / class_weights.mean()
    weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=cfg.lr_scheduler_factor,
        patience=cfg.lr_scheduler_patience,
    )

    best_state = None
    best_val_f1 = -1.0
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(cfg.epochs):
        if cfg.warmup_epochs > 0 and epoch < cfg.warmup_epochs:
            warmup_lr = cfg.lr * float(epoch + 1) / float(cfg.warmup_epochs)
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr

        model.train()
        train_loss = 0.0
        n_train = 0

        for xb, lb, xaux, yb in train_loader:
            xb = xb.to(device)
            lb = lb.to(device)
            xaux = xaux.to(device) if xaux.numel() > 0 else None
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb, lb, xaux)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            batch_size = xb.size(0)
            train_loss += float(loss.item()) * batch_size
            n_train += batch_size

        train_loss = train_loss / max(n_train, 1)

        model.eval()
        val_logits_list = []
        y_val_list = []
        with torch.no_grad():
            for xb, lb, xaux, yb in val_loader:
                xb = xb.to(device)
                lb = lb.to(device)
                xaux = xaux.to(device) if xaux.numel() > 0 else None
                logits = model(xb, lb, xaux)
                val_logits_list.append(logits.cpu())
                y_val_list.append(yb)

        val_logits = torch.cat(val_logits_list, dim=0)
        y_val_np = torch.cat(y_val_list, dim=0).numpy()
        val_f1 = _macro_f1_from_logits(val_logits, y_val_np)
        scheduler.step(val_f1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1:3d}/{cfg.epochs} - train_loss={train_loss:.4f}, val_f1_macro={val_f1:.4f}")

        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.patience:
                print(f"Early stopping triggered at epoch {epoch + 1}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_probs = _predict_probs(model, X_val, len_val, device, cfg.batch_size, X_val_static)
    class_bias = None
    bias_val_f1 = float("nan")
    bias_val_acc = float("nan")
    bias_val_draw_recall = float("nan")
    bias_val_draw_f1 = float("nan")
    if cfg.enable_val_bias_search and len(val_probs):
        class_bias, bias_val_f1, bias_val_acc, bias_val_draw_recall = _search_class_bias(val_probs, y_val)
        bias_val_draw_f1 = _draw_f1(y_val, _predict_with_class_bias(val_probs, class_bias))

    probs = _predict_probs(model, X_test, len_test, device, cfg.batch_size, X_test_static)
    pred_idx = _predict_with_class_bias(probs, class_bias)

    idx_to_label = {idx: label for idx, label in enumerate(LABEL_ORDER)}
    y_test_labels = np.array([idx_to_label[i] for i in y_test])
    pred_labels = np.array([idx_to_label[i] for i in pred_idx])

    acc = accuracy_score(y_test_labels, pred_labels)
    f1m = f1_score(y_test_labels, pred_labels, average="macro")

    try:
        from sklearn.preprocessing import label_binarize

        y_test_bin = label_binarize(y_test_labels, classes=LABEL_ORDER)
        auc = roc_auc_score(y_test_bin, probs, multi_class="ovr", average="weighted")
    except Exception:
        auc = float("nan")

    per_marker_accuracy: dict[int, float] = {}
    marker_text = f"Accuracy by {marker_name} (test holdout):\n"
    for marker in marker_order:
        mask = marker_test == marker
        if mask.any():
            marker_acc = accuracy_score(y_test_labels[mask], pred_labels[mask])
            per_marker_accuracy[int(marker)] = float(marker_acc)
            marker_text += f"  {marker_name}={marker}: {marker_acc:.4f}\n"

    mean_marker_accuracy = float(np.mean(list(per_marker_accuracy.values()))) if per_marker_accuracy else float("nan")

    if task_name == "in-play":
        cls_report = _format_classification_report_without_accuracy(y_test_labels, pred_labels)
    else:
        cls_report = classification_report(y_test_labels, pred_labels, target_names=["Away Win", "Draw", "Home Win"])
    cm = confusion_matrix(y_test_labels, pred_labels, labels=LABEL_ORDER)

    checkpoint = {
        "state_dict": model.state_dict(),
        "input_dim": len(feature_names),
        "hidden_dim": cfg.hidden_dim,
        "num_layers": cfg.num_layers,
        "bidirectional": cfg.bidirectional,
        "num_classes": len(LABEL_ORDER),
        "features": feature_names,
        "static_features": static_feature_names or [],
        "label_order": LABEL_ORDER,
        "norm_mean": mean,
        "norm_std": std,
        "static_norm_mean": static_mean,
        "static_norm_std": static_std,
        "max_seq_len": int(X_tr.shape[1]),
        "marker_name": marker_name,
        "marker_order": marker_order,
        "metrics": {
            "accuracy": float(acc),
            "f1_macro": float(f1m),
            "roc_auc_weighted_ovr": float(auc),
            "mean_marker_accuracy": float(mean_marker_accuracy),
            "marker_accuracy": per_marker_accuracy,
            "best_val_f1": float(best_val_f1),
            "best_epoch": int(best_epoch + 1),
            "bias_search_val_f1": float(bias_val_f1),
            "bias_search_val_accuracy": float(bias_val_acc),
            "bias_search_val_draw_recall": float(bias_val_draw_recall),
            "bias_search_val_draw_f1": float(bias_val_draw_f1),
        },
        "task": task_name,
        "config": cfg.__dict__,
        "class_bias": None if class_bias is None else class_bias.tolist(),
    }
    torch.save(checkpoint, model_path)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write(f"LSTM REPORT - {task_name.upper()}\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Data file: {data_file}\n")
        for line in report_lines:
            f.write(line + "\n")
        f.write(f"Device: {device}\n\n")
        f.write("Model config:\n")
        f.write(str(cfg.__dict__) + "\n\n")
        f.write(f"Best epoch: {best_epoch + 1}\n")
        f.write(f"Best val macro-F1: {best_val_f1:.4f}\n")
        if class_bias is not None:
            f.write(f"Validation-tuned class bias [A, D, H]: {np.round(class_bias, 4).tolist()}\n")
            f.write(f"Bias-tuned val macro-F1: {bias_val_f1:.4f}\n")
            f.write(f"Bias-tuned val accuracy: {bias_val_acc:.4f}\n")
            f.write(f"Bias-tuned val Draw recall: {bias_val_draw_recall:.4f}\n")
            f.write(f"Bias-tuned val Draw F1: {bias_val_draw_f1:.4f}\n")
        if task_name == "in-play":
            f.write(f"Mean {marker_name} accuracy: {mean_marker_accuracy:.4f}\n")
        else:
            f.write(f"Test Accuracy: {acc:.4f}\n")
        f.write(f"Test Macro-F1: {f1m:.4f}\n")
        f.write(f"Test ROC-AUC (weighted ovr): {auc:.4f}\n\n")
        f.write("Classification report:\n")
        f.write(cls_report + "\n")
        f.write("Confusion matrix [A, D, H]:\n")
        f.write(str(cm) + "\n\n")
        f.write(marker_text + "\n")

    print("\n" + "-" * 72)
    print(f"Best epoch: {best_epoch + 1} | Best val F1-macro: {best_val_f1:.4f}")
    if class_bias is not None:
        print(
            "Validation-tuned class bias "
            f"[A, D, H]={np.round(class_bias, 4).tolist()} | "
            f"val F1-macro={bias_val_f1:.4f}, val acc={bias_val_acc:.4f}, "
            f"val Draw recall={bias_val_draw_recall:.4f}, val Draw F1={bias_val_draw_f1:.4f}"
        )
    if task_name == "in-play":
        print(f"Test metrics -> Mean {marker_name} accuracy: {mean_marker_accuracy:.4f}, F1-macro: {f1m:.4f}, ROC-AUC(w-ovr): {auc:.4f}")
    else:
        print(f"Test metrics -> Accuracy: {acc:.4f}, F1-macro: {f1m:.4f}, ROC-AUC(w-ovr): {auc:.4f}")
    print(f"Model saved:  {model_path}")
    print(f"Report saved: {report_path}")


def main() -> None:
    set_seed(SEED)

    print("\n" + "=" * 72)
    print("LSTM training pipeline")
    print("=" * 72)

    print("\n" + "=" * 72)
    print("LSTM training for: Pre-match (0-minute)")
    print("=" * 72)
    pre_cfg = _build_task_config("pre-match")

    pretrain_file = DATA_DIR / "pretrain_dataset.csv"
    pre_df = pd.read_csv(pretrain_file)
    pre_exclude_cols = {
        "fixture_id",
        "date",
        "home_team",
        "away_team",
        "home_formation",
        "away_formation",
        "result",
        "goals_home",
        "goals_away",
    }
    pre_static_feature_cols = _select_numeric_features(pre_df, pre_exclude_cols)
    pre_train_mask, pre_test_mask = _temporal_holdout_by_fixture(pre_df, date_col="date", group_col="fixture_id", test_frac=0.2)
    pre_train_df = pre_df[pre_train_mask].copy()
    pre_test_df = pre_df[pre_test_mask].copy()

    pre_train_fixture_ids, pre_val_fixture_ids = _train_val_split_fixture_ids(pre_train_df, val_frac=0.15)
    pre_all_train_ids = set(pre_train_df["fixture_id"].unique().tolist())
    if not pre_val_fixture_ids:
        sorted_ids = sorted(pre_all_train_ids)
        pre_val_fixture_ids = set(sorted_ids[-1:])
        pre_train_fixture_ids = set(sorted_ids[:-1])

    X_tr_pre, len_tr_pre, y_tr_pre, marker_tr_pre, X_tr_pre_static = build_prematch_h2h_sequence_samples(
        pre_df,
        set(pre_train_fixture_ids),
        static_feature_cols=pre_static_feature_cols,
    )
    X_val_pre, len_val_pre, y_val_pre, marker_val_pre, X_val_pre_static = build_prematch_h2h_sequence_samples(
        pre_df,
        set(pre_val_fixture_ids),
        static_feature_cols=pre_static_feature_cols,
    )
    X_test_pre, len_test_pre, y_test_pre, marker_test_pre, X_test_pre_static = build_prematch_h2h_sequence_samples(
        pre_df,
        set(pre_test_df["fixture_id"].unique().tolist()),
        static_feature_cols=pre_static_feature_cols,
    )

    train_lstm_task(
        task_name="pre-match",
        data_file=pretrain_file,
        feature_names=PREMATCH_SEQUENCE_FEATURES,
        X_tr=X_tr_pre,
        len_tr=len_tr_pre,
        y_tr=y_tr_pre,
        X_val=X_val_pre,
        len_val=len_val_pre,
        y_val=y_val_pre,
        X_test=X_test_pre,
        len_test=len_test_pre,
        y_test=y_test_pre,
        marker_test=marker_test_pre,
        marker_name="minute",
        marker_order=[0],
        model_path=MODEL_DIR / "pretrain_model_lstm.pt",
        report_path=REPORT_DIR / "pretrain_report_lstm.txt",
        cfg=pre_cfg,
        report_lines=[
            f"Sequence feature count: {len(PREMATCH_SEQUENCE_FEATURES)}",
            f"Static feature count: {len(pre_static_feature_cols)}",
            f"Max H2H sequence length: {PREMATCH_SEQ_LEN}",
            f"H2H observations kept: up to {PREMATCH_MAX_H2H}",
            f"Team-form observations per side: up to {PREMATCH_MAX_TEAM_FORM}",
            "Tuning change: larger hybrid history, mild Draw reweighting, static-conditioned attention, gated dual-head fusion, and split-validation bias calibration",
            "Tuning refinement: lighter pre-match model, LR warmup, separate sequence/static logits with learned fusion, and more robust class-bias selection",
            "Sequence meaning: ordered mix of head-to-head history and each side's recent matches before the current fixture",
            f"Train/Val/Test samples: {len(X_tr_pre)}/{len(X_val_pre)}/{len(X_test_pre)}",
            f"Train/Val/Test fixtures: {len(pre_train_fixture_ids)}/{len(pre_val_fixture_ids)}/{pre_test_df['fixture_id'].nunique()}",
        ],
        static_feature_names=pre_static_feature_cols,
        X_tr_static=X_tr_pre_static,
        X_val_static=X_val_pre_static,
        X_test_static=X_test_pre_static,
    )

    print("\n" + "=" * 72)
    print("LSTM training for: In-play")
    print("=" * 72)
    inplay_cfg = _build_task_config("in-play")

    inplay_file = DATA_DIR / "inplay_dataset.csv"
    df = pd.read_csv(inplay_file)
    checkpoint_minutes = _get_inplay_checkpoint_minutes(df)
    exclude_cols = {
        "fixture_id",
        "date",
        "home_team",
        "away_team",
        "result",
        "ft_home",
        "ft_away",
    }
    feature_cols = _select_numeric_features(df, exclude_cols)

    train_mask, test_mask = _temporal_holdout_by_fixture(df, date_col="date", group_col="fixture_id", test_frac=0.2)
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    train_fixture_ids, val_fixture_ids = _train_val_split_fixture_ids(train_df, val_frac=0.15)
    all_train_fixture_ids = set(train_df["fixture_id"].unique().tolist())
    if not val_fixture_ids:
        sorted_ids = sorted(all_train_fixture_ids)
        val_fixture_ids = set(sorted_ids[-1:])
        train_fixture_ids = set(sorted_ids[:-1])

    X_tr, len_tr, y_tr, minute_tr = build_sequence_samples(df, feature_cols, set(train_fixture_ids), checkpoint_minutes)
    X_val, len_val, y_val, minute_val = build_sequence_samples(df, feature_cols, set(val_fixture_ids), checkpoint_minutes)
    X_test, len_test, y_test, minute_test = build_sequence_samples(df, feature_cols, set(test_df["fixture_id"].unique().tolist()), checkpoint_minutes)

    train_lstm_task(
        task_name="in-play",
        data_file=inplay_file,
        feature_names=feature_cols,
        X_tr=X_tr,
        len_tr=len_tr,
        y_tr=y_tr,
        X_val=X_val,
        len_val=len_val,
        y_val=y_val,
        X_test=X_test,
        len_test=len_test,
        y_test=y_test,
        marker_test=minute_test,
        marker_name="minute",
        marker_order=checkpoint_minutes,
        model_path=MODEL_DIR / "inplay_model_lstm.pt",
        report_path=REPORT_DIR / "inplay_report_lstm.txt",
        cfg=inplay_cfg,
        report_lines=[
            f"Feature count: {len(feature_cols)}",
            f"Checkpoint minutes: {checkpoint_minutes}",
            "Tuning change: restored lighter in-play setup, no extra Draw reweighting, and attention pooling without forced final-state fusion",
            "Tuning refinement: slightly lower LR plus longer warmup to recover early-minute stability",
            f"Train/Val/Test samples: {len(X_tr)}/{len(X_val)}/{len(X_test)}",
            f"Train/Val/Test fixtures: {len(train_fixture_ids)}/{len(val_fixture_ids)}/{test_df['fixture_id'].nunique()}",
        ],
    )


if __name__ == "__main__":
    main()