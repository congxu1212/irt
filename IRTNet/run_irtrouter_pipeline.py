#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

SRC_DIR = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC_DIR))
from modules import MoEClassifier  # noqa: E402


class ClassificationDataset(Dataset):
    def __init__(self, df, model_map, prompt_map):
        model_ids = df["model_id"].map(model_map)
        prompt_ids = df["prompt_id"].map(prompt_map)
        if model_ids.isna().any() or prompt_ids.isna().any():
            raise ValueError("Found model_id or prompt_id values missing from the maps.")
        self.model_ids = torch.tensor(model_ids.to_numpy(), dtype=torch.long)
        self.prompt_ids = torch.tensor(prompt_ids.to_numpy(), dtype=torch.long)
        self.labels = torch.tensor(df["label"].astype(float).to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.model_ids[idx], self.prompt_ids[idx], self.labels[idx]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare IRT-Router data, embed questions, train IRTNet, and evaluate routing."
    )
    parser.add_argument("--router-root", type=Path, default=Path("../IRT-Router"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/irtrouter_pipeline"))
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--embedding-device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--embedding-source",
        choices=["sentence-transformer", "irtrouter-bert"],
        default="sentence-transformer",
        help="Generate all-mpnet embeddings or convert IRT-Router's existing BERT query embeddings.",
    )
    parser.add_argument("--embedding-model", default="all-mpnet-base-v2")
    parser.add_argument("--embedding-batch-size", type=int, default=128)
    parser.add_argument("--force-embeddings", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--model-embed-dim", type=int, default=128)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k-experts", type=int, default=2)
    parser.add_argument("--expert-hidden-dim", type=int, default=256)
    parser.add_argument("--shared-expert-hidden-dim", type=int, default=256)
    parser.add_argument("--expert-output-dim", type=int, default=128)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--embedding-noise", type=float, default=0.02)
    parser.add_argument("--bias-update-speed", type=float, default=0.01)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--label-threshold", type=float, default=0.5)
    return parser.parse_args()


def select_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_router_split(router_root, split_name):
    path = router_root / "data" / f"{split_name}.csv"
    return pd.read_csv(
        path,
        usecols=["id", "question", "performance", "task", "llm"],
    )


def convert_split(df, split_name, question_to_id):
    missing = ~df["question"].isin(question_to_id)
    if missing.any():
        examples = df.loc[missing, "question"].head(3).tolist()
        raise ValueError(f"{split_name} has {missing.sum()} questions missing from prompt order: {examples}")
    return pd.DataFrame(
        {
            "model_id": df["llm"].astype(str),
            "prompt_id": df["question"].map(question_to_id).astype(int),
            "prompt": df["question"],
            "label": df["performance"].astype(float).clip(0.0, 1.0),
            "category": df["task"].astype(str),
            "source_split": split_name,
            "source_id": df["id"],
        }
    )


def split_train_by_prompt(train_df, val_fraction, seed):
    rng = np.random.default_rng(seed)
    prompt_info = train_df[["prompt_id", "category"]].drop_duplicates("prompt_id")
    val_prompt_ids = []
    for _, group in prompt_info.groupby("category", sort=True):
        prompt_ids = group["prompt_id"].to_numpy()
        if len(prompt_ids) <= 1:
            continue
        n_val = max(1, int(round(len(prompt_ids) * val_fraction)))
        n_val = min(n_val, len(prompt_ids) - 1)
        val_prompt_ids.extend(rng.choice(prompt_ids, size=n_val, replace=False).tolist())

    val_prompt_ids = set(val_prompt_ids)
    val_df = train_df[train_df["prompt_id"].isin(val_prompt_ids)].reset_index(drop=True)
    inner_train_df = train_df[~train_df["prompt_id"].isin(val_prompt_ids)].reset_index(drop=True)
    if val_df.empty or inner_train_df.empty:
        raise ValueError("Validation split failed; adjust --val-fraction.")
    return inner_train_df, val_df


def prepare_data(args):
    router_root = args.router_root.resolve()
    if not router_root.exists():
        raise FileNotFoundError(f"Router root not found: {router_root}")

    prepared_dir = args.output_dir.resolve() / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)

    raw_splits = {name: read_router_split(router_root, name) for name in ["train", "test1", "test2"]}
    question_order = (
        pd.concat([df[["question"]] for df in raw_splits.values()], ignore_index=True)
        .drop_duplicates("question")
        .reset_index(drop=True)
    )
    question_order.insert(0, "prompt_id", np.arange(len(question_order), dtype=np.int64))
    question_order.rename(columns={"question": "prompt"}, inplace=True)
    question_to_id = question_order.set_index("prompt")["prompt_id"].to_dict()

    converted = {
        split_name: convert_split(df, split_name, question_to_id)
        for split_name, df in raw_splits.items()
    }
    train_df, val_df = split_train_by_prompt(converted["train"], args.val_fraction, args.seed)
    test_dfs = {"test1": converted["test1"], "test2": converted["test2"]}

    train_df.to_csv(prepared_dir / "train.csv", index=False)
    val_df.to_csv(prepared_dir / "val.csv", index=False)
    test_dfs["test1"].to_csv(prepared_dir / "test1.csv", index=False)
    test_dfs["test2"].to_csv(prepared_dir / "test2.csv", index=False)
    question_order.to_csv(prepared_dir / "question_order.csv", index=False)

    all_dfs = [train_df, val_df, *test_dfs.values()]
    summary = {
        "router_root": str(router_root),
        "prepared_dir": str(prepared_dir),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "val_fraction": args.val_fraction,
        "unique_questions": int(len(question_order)),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test1_rows": int(len(test_dfs["test1"])),
        "test2_rows": int(len(test_dfs["test2"])),
        "train_prompts": int(train_df["prompt_id"].nunique()),
        "val_prompts": int(val_df["prompt_id"].nunique()),
        "test1_prompts": int(test_dfs["test1"]["prompt_id"].nunique()),
        "test2_prompts": int(test_dfs["test2"]["prompt_id"].nunique()),
        "all_models": sorted(pd.concat([df["model_id"] for df in all_dfs]).unique().tolist()),
        "tasks": sorted(pd.concat([df["category"] for df in all_dfs]).unique().tolist()),
    }
    (prepared_dir / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return train_df, val_df, test_dfs, question_order, summary


def load_irtrouter_bert_embeddings(args, question_order):
    import pickle

    query_map_path = args.router_root.resolve() / "utils/map/query.csv"
    embedding_pickle_path = args.router_root.resolve() / "utils/bert_embeddings/query_embeddings.pkl"
    query_map = pd.read_csv(query_map_path, usecols=["index", "question"])
    question_to_router_idx = {}
    for row in query_map.itertuples(index=False):
        question_to_router_idx.setdefault(row.question, int(row.index))

    with open(embedding_pickle_path, "rb") as f:
        embedding_records = pickle.load(f)
    embedding_lookup = {
        int(record["index"]): np.asarray(record["embedding"], dtype=np.float32)
        for record in embedding_records
    }

    missing_questions = [
        prompt
        for prompt in question_order["prompt"].astype(str).tolist()
        if prompt not in question_to_router_idx
    ]
    if missing_questions:
        raise ValueError(f"Missing {len(missing_questions)} questions from {query_map_path}")

    missing_embeddings = [
        question_to_router_idx[prompt]
        for prompt in question_order["prompt"].astype(str).tolist()
        if question_to_router_idx[prompt] not in embedding_lookup
    ]
    if missing_embeddings:
        raise ValueError(f"Missing {len(missing_embeddings)} embeddings from {embedding_pickle_path}")

    vectors = [
        embedding_lookup[question_to_router_idx[prompt]]
        for prompt in question_order["prompt"].astype(str).tolist()
    ]
    return torch.tensor(np.stack(vectors), dtype=torch.float32), embedding_pickle_path


def generate_or_load_embeddings(args, question_order, device):
    prepared_dir = args.output_dir.resolve() / "prepared"
    embedding_path = prepared_dir / "question_embeddings.pth"
    metadata_path = prepared_dir / "question_embeddings_metadata.json"
    expected_rows = len(question_order)

    if embedding_path.exists() and not args.force_embeddings:
        embeddings = torch.load(embedding_path, map_location="cpu")
        metadata = {}
        if metadata_path.exists():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        source_matches = metadata.get("embedding_source") in {None, args.embedding_source}
        model_matches = (
            args.embedding_source == "irtrouter-bert"
            or metadata.get("embedding_model") in {None, args.embedding_model}
        )
        if embeddings.shape[0] == expected_rows and source_matches and model_matches:
            print(f"Using existing question embeddings: {embedding_path} shape={tuple(embeddings.shape)}")
            return embeddings, embedding_path
        print(f"Regenerating embeddings because cached metadata/shape does not match this run.")

    if args.embedding_source == "irtrouter-bert":
        embeddings, source_path = load_irtrouter_bert_embeddings(args, question_order)
        torch.save(embeddings, embedding_path)
        metadata = {
            "embedding_source": args.embedding_source,
            "source_path": str(source_path),
            "shape": list(embeddings.shape),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Saved IRT-Router BERT embeddings tensor: {embedding_path} shape={tuple(embeddings.shape)}")
        return embeddings, embedding_path

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "sentence_transformers is required to generate question embeddings. "
            "Install it with: python -m pip install sentence-transformers==3.4.1"
        ) from exc

    print(f"Loading embedding model: {args.embedding_model} on {device}")
    model = SentenceTransformer(args.embedding_model, device=str(device))
    questions = question_order["prompt"].astype(str).tolist()
    embeddings_np = model.encode(
        questions,
        batch_size=args.embedding_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    embeddings = torch.tensor(embeddings_np, dtype=torch.float32)
    torch.save(embeddings, embedding_path)
    metadata = {
        "embedding_source": args.embedding_source,
        "embedding_model": args.embedding_model,
        "embedding_device": str(device),
        "embedding_batch_size": args.embedding_batch_size,
        "shape": list(embeddings.shape),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved question embeddings: {embedding_path} shape={tuple(embeddings.shape)}")
    return embeddings, embedding_path


def make_maps(train_df, val_df, test_dfs, question_order):
    model_series = pd.concat(
        [train_df["model_id"], val_df["model_id"]]
        + [df["model_id"] for df in test_dfs.values()],
        ignore_index=True,
    )
    model_map = {model_id: i for i, model_id in enumerate(model_series.drop_duplicates().tolist())}
    prompt_map = {int(prompt_id): int(prompt_id) for prompt_id in question_order["prompt_id"].tolist()}
    return model_map, prompt_map


def sigmoid_np(logits):
    logits = np.clip(np.asarray(logits, dtype=np.float64), -50, 50)
    return 1.0 / (1.0 + np.exp(-logits))


def compute_binary_metrics(labels, logits, label_threshold):
    labels = np.asarray(labels, dtype=np.float64)
    probs = sigmoid_np(logits)
    binary_labels = labels >= label_threshold
    binary_preds = probs >= 0.5
    metrics = {
        "rows": int(len(labels)),
        "positive_rate_at_threshold": float(binary_labels.mean()) if len(labels) else 0.0,
        "threshold_accuracy": float(accuracy_score(binary_labels, binary_preds)) if len(labels) else 0.0,
        "mae": float(np.mean(np.abs(probs - labels))) if len(labels) else 0.0,
        "rmse": float(np.sqrt(np.mean((probs - labels) ** 2))) if len(labels) else 0.0,
    }
    if len(np.unique(binary_labels)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(binary_labels, probs))
    else:
        metrics["roc_auc"] = None
    return metrics


def evaluate_rows(model, df, model_map, prompt_map, args, device):
    dataset = ClassificationDataset(df, model_map, prompt_map)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    logits_out, labels_out, model_ids_out, prompt_ids_out = [], [], [], []
    loss_sum, count = 0.0, 0

    model.eval()
    with torch.no_grad():
        for model_ids, prompt_ids, labels in tqdm(loader, desc="predict", leave=False):
            model_ids = model_ids.to(device)
            prompt_ids = prompt_ids.to(device)
            labels = labels.to(device)
            logits = model(model_ids, prompt_ids)
            loss_sum += loss_fn(logits, labels).item()
            count += labels.numel()
            logits_out.append(logits.detach().cpu())
            labels_out.append(labels.detach().cpu())
            model_ids_out.append(model_ids.detach().cpu())
            prompt_ids_out.append(prompt_ids.detach().cpu())

    logits = torch.cat(logits_out).numpy()
    labels = torch.cat(labels_out).numpy()
    metrics = compute_binary_metrics(labels, logits, args.label_threshold)
    metrics["bce"] = float(loss_sum / max(count, 1))

    predictions = df.copy()
    predictions["mapped_model_idx"] = torch.cat(model_ids_out).numpy()
    predictions["mapped_prompt_idx"] = torch.cat(prompt_ids_out).numpy()
    predictions["logit"] = logits
    predictions["probability"] = sigmoid_np(logits)
    predictions["binary_label"] = predictions["label"] >= args.label_threshold
    predictions["binary_prediction"] = predictions["probability"] >= 0.5
    return metrics, predictions


def evaluate_by_task(predictions, args):
    return {
        str(task): compute_binary_metrics(
            group["label"].to_numpy(), group["logit"].to_numpy(), args.label_threshold
        )
        for task, group in predictions.groupby("category", sort=True)
    }


def evaluate_routing(predictions, args):
    records = []
    for prompt_id, group in predictions.groupby("prompt_id", sort=False):
        selected = group.loc[group["probability"].idxmax()]
        labels = group["label"].astype(float)
        records.append(
            {
                "prompt_id": int(prompt_id),
                "category": selected["category"],
                "selected_model_id": selected["model_id"],
                "selected_probability": float(selected["probability"]),
                "selected_label": float(selected["label"]),
                "selected_success": bool(selected["label"] >= args.label_threshold),
                "oracle_label": float(labels.max()),
                "oracle_success": bool(labels.max() >= args.label_threshold),
                "random_expected_label": float(labels.mean()),
                "candidate_models": int(group["model_id"].nunique()),
            }
        )
    routing = pd.DataFrame(records)

    def summarize(group):
        return pd.Series(
            {
                "prompts": int(len(group)),
                "mean_selected_label": float(group["selected_label"].mean()),
                "success_at_threshold": float(group["selected_success"].mean()),
                "oracle_mean_label": float(group["oracle_label"].mean()),
                "oracle_success_at_threshold": float(group["oracle_success"].mean()),
                "random_expected_label": float(group["random_expected_label"].mean()),
                "mean_candidate_models": float(group["candidate_models"].mean()),
            }
        )

    overall = summarize(routing).to_dict()
    by_task = {
        str(task): summarize(group).to_dict()
        for task, group in routing.groupby("category", sort=True)
    }
    return routing, overall, by_task


def build_model(args, model_map, prompt_map, prompt_embeddings, device):
    return MoEClassifier(
        num_models=len(model_map),
        num_prompts=len(prompt_map),
        num_experts=args.num_experts,
        prompt_embeddings=prompt_embeddings,
        model_embed_dim=args.model_embed_dim,
        top_k_experts=args.top_k_experts,
        expert_hidden_dim=args.expert_hidden_dim,
        expert_output_dim=args.expert_output_dim,
        shared_expert_hidden_dim=args.shared_expert_hidden_dim,
        dropout_rate=args.dropout_rate,
        embedding_noise=args.embedding_noise,
    ).to(device)


def run_validation(model, val_df, model_map, prompt_map, args, device):
    metrics, _ = evaluate_rows(model, val_df, model_map, prompt_map, args, device)
    return metrics


def train_model(train_df, val_df, model_map, prompt_map, prompt_embeddings, args, device):
    train_dataset = ClassificationDataset(train_df, model_map, prompt_map)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    model = build_model(args, model_map, prompt_map, prompt_embeddings, device)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)
    loss_fn = nn.BCEWithLogitsLoss()
    best_val_bce = math.inf
    best_epoch = 0
    epochs_no_improve = 0
    history = []
    model_path = args.output_dir.resolve() / "best_irt_router.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum, train_count = 0.0, 0
        for model_ids, prompt_ids, labels in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            model_ids = model_ids.to(device)
            prompt_ids = prompt_ids.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            a_q, b_q, gating_logits, _ = model.analyze_prompt(prompt_ids)
            theta = model.model_embedder(model_ids)
            logits = (torch.sum(a_q * theta, dim=1, keepdim=True) - b_q).squeeze(-1)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                mean_expert_logits = gating_logits.mean(dim=0)
                ideal_logit_value = mean_expert_logits.mean()
                bias_update = torch.zeros_like(model.routed_moe.bias)
                bias_update.masked_fill_(mean_expert_logits > ideal_logit_value, -args.bias_update_speed)
                bias_update.masked_fill_(mean_expert_logits < ideal_logit_value, args.bias_update_speed)
                model.routed_moe.bias.add_(bias_update)

            train_loss_sum += loss.item() * labels.numel()
            train_count += labels.numel()

        train_bce = train_loss_sum / max(train_count, 1)
        val_metrics = run_validation(model, val_df, model_map, prompt_map, args, device)
        scheduler.step(val_metrics["bce"])
        epoch_record = {
            "epoch": epoch,
            "train_bce": float(train_bce),
            "val_bce": val_metrics["bce"],
            "val_threshold_accuracy": val_metrics["threshold_accuracy"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, sort_keys=True))

        if val_metrics["bce"] < best_val_bce:
            best_val_bce = val_metrics["bce"]
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.early_stopping_patience:
                break

    model.load_state_dict(torch.load(model_path, map_location=device))
    return model, model_path, history, best_epoch, best_val_bce


def main():
    args = parse_args()
    if args.top_k_experts > args.num_experts:
        raise ValueError("--top-k-experts cannot exceed --num-experts.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    train_device = select_device(args.device)
    embedding_device = select_device(args.embedding_device)
    print(f"training device: {train_device}")
    print(f"embedding device: {embedding_device}")

    train_df, val_df, test_dfs, question_order, data_summary = prepare_data(args)
    embeddings_cpu, embedding_path = generate_or_load_embeddings(args, question_order, embedding_device)
    if args.prepare_only:
        print(f"Prepared data and embeddings under: {args.output_dir.resolve() / 'prepared'}")
        return

    model_map, prompt_map = make_maps(train_df, val_df, test_dfs, question_order)
    train_models = set(train_df["model_id"].unique())
    model_coverage = {
        name: {
            "models": sorted(df["model_id"].unique().tolist()),
            "unseen_models": sorted(set(df["model_id"].unique()) - train_models),
        }
        for name, df in test_dfs.items()
    }

    prompt_embeddings = embeddings_cpu.to(train_device)
    model_path = args.output_dir.resolve() / "best_irt_router.pt"
    history = []
    best_epoch = None
    best_val_bce = None

    if args.eval_only:
        if not model_path.exists():
            raise FileNotFoundError(f"--eval-only requested but no checkpoint exists: {model_path}")
        model = build_model(args, model_map, prompt_map, prompt_embeddings, train_device)
        model.load_state_dict(torch.load(model_path, map_location=train_device))
    else:
        model, model_path, history, best_epoch, best_val_bce = train_model(
            train_df, val_df, model_map, prompt_map, prompt_embeddings, args, train_device
        )

    metrics = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "data_summary": data_summary,
        "embedding_path": str(embedding_path),
        "embedding_shape": list(embeddings_cpu.shape),
        "model_coverage": model_coverage,
        "best_epoch": best_epoch,
        "best_val_bce": None if best_val_bce is None else float(best_val_bce),
        "history": history,
        "model_path": str(model_path),
        "tests": {},
    }

    for name, df in test_dfs.items():
        row_metrics, predictions = evaluate_rows(model, df, model_map, prompt_map, args, train_device)
        task_metrics = evaluate_by_task(predictions, args)
        routing, routing_overall, routing_by_task = evaluate_routing(predictions, args)
        predictions.to_csv(args.output_dir / f"predictions_{name}.csv", index=False)
        routing.to_csv(args.output_dir / f"routing_{name}.csv", index=False)
        metrics["tests"][name] = {
            "row_metrics": row_metrics,
            "row_metrics_by_task": task_metrics,
            "routing_overall": routing_overall,
            "routing_by_task": routing_by_task,
        }
        print(f"\n{name} row_metrics: {json.dumps(row_metrics, sort_keys=True)}")
        print(f"{name} routing_overall: {json.dumps(routing_overall, sort_keys=True)}")

    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nmetrics_path: {metrics_path}")
    print(f"model_path: {model_path}")


if __name__ == "__main__":
    main()
