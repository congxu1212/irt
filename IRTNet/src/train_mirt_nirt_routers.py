import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parents[1]
WORKSPACE_DIR = PROJECT_DIR.parent
ROUTER_DIR = WORKSPACE_DIR / "IRT-Router"
if not ROUTER_DIR.exists():
    raise FileNotFoundError(f"Expected router code at {ROUTER_DIR}")
sys.path.insert(0, str(ROUTER_DIR))

from router import MIRT, NIRT  # noqa: E402


class IRTNetRouterDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.model_ids = torch.tensor(df["model_id"].to_numpy(), dtype=torch.long)
        self.prompt_ids = torch.tensor(df["prompt_id"].to_numpy(), dtype=torch.long)
        self.category_ids = torch.tensor(df["category_id"].to_numpy(), dtype=torch.long)
        self.labels = torch.tensor(df["label"].astype("float32").to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.model_ids[idx], self.prompt_ids[idx], self.category_ids[idx], self.labels[idx]


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate MIRT/NIRT routers on IRT-Net data.")
    parser.add_argument("--train-data-path", type=Path, default=PROJECT_DIR / "data/train.csv")
    parser.add_argument("--test-data-path", type=Path, default=PROJECT_DIR / "data/test.csv")
    parser.add_argument("--question-embedding-path", type=Path, default=PROJECT_DIR / "data/question_embeddings.pth")
    parser.add_argument("--model-order-path", type=Path, default=PROJECT_DIR / "data/model_order.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_DIR / "reports/router_eval")
    parser.add_argument("--routers", nargs="+", choices=["mirt", "nirt"], default=["mirt", "nirt"])
    parser.add_argument("--mirt-epochs", type=int, default=9)
    parser.add_argument("--nirt-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--run-name", type=str, default=None)
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split(path: Path, include_text: bool = False) -> pd.DataFrame:
    usecols = ["prompt_id", "model_id", "category_id", "label"]
    if include_text:
        usecols.extend(["prompt", "model_name", "category"])
    return pd.read_csv(path, usecols=usecols)


def make_loader(df: pd.DataFrame, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        IRTNetRouterDataset(df),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def make_batch_features(batch, prompt_embeddings, num_models: int, num_categories: int, device: torch.device):
    model_ids, prompt_ids, category_ids, labels = batch
    model_ids = model_ids.to(device)
    prompt_ids = prompt_ids.to(device)
    category_ids = category_ids.to(device)
    labels = labels.to(device)
    model_vecs = F.one_hot(model_ids, num_classes=num_models).float()
    prompt_vecs = prompt_embeddings[prompt_ids]
    category_vecs = F.one_hot(category_ids, num_classes=num_categories).float()
    return model_ids, prompt_ids, category_ids, model_vecs, prompt_vecs, category_vecs, labels


def forward_router(router_name: str, cdm, batch, prompt_embeddings, num_models: int, num_categories: int, device: torch.device):
    _, _, _, model_vecs, prompt_vecs, category_vecs, labels = make_batch_features(
        batch, prompt_embeddings, num_models, num_categories, device
    )
    if router_name == "mirt":
        pred, *_ = cdm.irt_net(model_vecs, prompt_vecs)
    elif router_name == "nirt":
        pred, *_ = cdm.nirt_net(model_vecs, prompt_vecs, category_vecs)
    else:
        raise ValueError(f"Unknown router: {router_name}")
    return pred.reshape(-1), labels.reshape(-1)


def compute_metrics(labels, probs):
    labels = np.asarray(labels, dtype=np.float32)
    probs = np.clip(np.asarray(probs, dtype=np.float32), 1e-7, 1 - 1e-7)
    preds = probs >= 0.5
    metrics = {
        "bce": float(-(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)).mean()),
        "accuracy": float(accuracy_score(labels, preds)),
        "rmse": float(np.sqrt(mean_squared_error(labels, probs))),
        "mae": float(mean_absolute_error(labels, probs)),
        "label_mean": float(labels.mean()),
        "pred_mean": float(probs.mean()),
    }
    try:
        metrics["auc"] = float(roc_auc_score(labels, probs))
    except ValueError:
        metrics["auc"] = None
    return metrics


def predict(router_name: str, cdm, loader, prompt_embeddings, num_models: int, num_categories: int, device: torch.device):
    net = cdm.irt_net if router_name == "mirt" else cdm.nirt_net
    net.eval()
    all_prompt_ids, all_model_ids, all_category_ids, all_labels, all_probs = [], [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{router_name} prediction"):
            model_ids, prompt_ids, category_ids, _, _, _, labels = make_batch_features(
                batch, prompt_embeddings, num_models, num_categories, device
            )
            if router_name == "mirt":
                model_vecs = F.one_hot(model_ids, num_classes=num_models).float()
                prompt_vecs = prompt_embeddings[prompt_ids]
                probs, *_ = cdm.irt_net(model_vecs, prompt_vecs)
            else:
                model_vecs = F.one_hot(model_ids, num_classes=num_models).float()
                prompt_vecs = prompt_embeddings[prompt_ids]
                category_vecs = F.one_hot(category_ids, num_classes=num_categories).float()
                probs, *_ = cdm.nirt_net(model_vecs, prompt_vecs, category_vecs)
            all_prompt_ids.append(prompt_ids.cpu())
            all_model_ids.append(model_ids.cpu())
            all_category_ids.append(category_ids.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.reshape(-1).detach().cpu())
    return pd.DataFrame(
        {
            "prompt_id": torch.cat(all_prompt_ids).numpy(),
            "model_id": torch.cat(all_model_ids).numpy(),
            "category_id": torch.cat(all_category_ids).numpy(),
            "label": torch.cat(all_labels).numpy(),
            "pred_prob": torch.cat(all_probs).numpy(),
        }
    )


def evaluate(router_name: str, cdm, test_loader, test_df, prompt_embeddings, num_models, num_categories, device, output_dir):
    pred_df = predict(router_name, cdm, test_loader, prompt_embeddings, num_models, num_categories, device)
    prediction_path = output_dir / f"{router_name}_test_predictions.csv"
    pred_df.to_csv(prediction_path, index=False)

    metrics = compute_metrics(pred_df["label"].to_numpy(), pred_df["pred_prob"].to_numpy())

    selected_idx = pred_df.groupby("prompt_id")["pred_prob"].idxmax()
    selected = pred_df.loc[selected_idx].copy().sort_values("prompt_id")
    prompt_meta = test_df[["prompt_id", "category", "prompt"]].drop_duplicates("prompt_id")
    model_meta = test_df[["model_id", "model_name"]].drop_duplicates("model_id")
    selected = selected.merge(prompt_meta, on="prompt_id", how="left").merge(model_meta, on="model_id", how="left")
    selected.rename(columns={"label": "selected_label", "pred_prob": "selected_pred_prob"}, inplace=True)

    oracle_by_prompt = pred_df.groupby("prompt_id")["label"].max().rename("oracle_label").reset_index()
    selected = selected.merge(oracle_by_prompt, on="prompt_id", how="left")
    selected["route_success"] = selected["selected_label"].astype(float)
    selected_path = output_dir / f"{router_name}_selected_routes.csv"
    selected.to_csv(selected_path, index=False)

    per_category = (
        selected.groupby("category", dropna=False)
        .agg(
            prompts=("prompt_id", "count"),
            route_accuracy=("route_success", "mean"),
            oracle_accuracy=("oracle_label", "mean"),
            mean_selected_pred_prob=("selected_pred_prob", "mean"),
        )
        .reset_index()
        .sort_values(["route_accuracy", "prompts"], ascending=[False, False])
    )
    per_category_path = output_dir / f"{router_name}_per_category.csv"
    per_category.to_csv(per_category_path, index=False)

    metrics.update(
        {
            "router": router_name,
            "route_accuracy": float(selected["route_success"].mean()),
            "oracle_route_accuracy": float(selected["oracle_label"].mean()),
            "num_test_rows": int(len(pred_df)),
            "num_test_prompts": int(selected["prompt_id"].nunique()),
            "prediction_path": str(prediction_path),
            "selected_routes_path": str(selected_path),
            "per_category_path": str(per_category_path),
        }
    )
    return metrics


def train_router(
    router_name: str,
    cdm,
    train_loader,
    test_loader,
    test_df,
    prompt_embeddings,
    num_models,
    num_categories,
    device,
    epochs,
    lr,
    output_dir,
):
    net = cdm.irt_net if router_name == "mirt" else cdm.nirt_net
    net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.BCELoss()
    history = []
    best_route_accuracy = -1.0
    best_state = None
    start = time.time()

    for epoch in range(1, epochs + 1):
        net.train()
        train_loss_total = 0.0
        train_count = 0
        for batch in tqdm(train_loader, desc=f"{router_name} epoch {epoch}/{epochs}"):
            probs, labels = forward_router(router_name, cdm, batch, prompt_embeddings, num_models, num_categories, device)
            loss = loss_fn(probs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_n = labels.numel()
            train_loss_total += float(loss.detach().cpu()) * batch_n
            train_count += batch_n

        epoch_metrics = evaluate(
            router_name,
            cdm,
            test_loader,
            test_df,
            prompt_embeddings,
            num_models,
            num_categories,
            device,
            output_dir,
        )
        epoch_metrics["epoch"] = epoch
        epoch_metrics["train_bce"] = train_loss_total / train_count
        epoch_metrics["elapsed_seconds"] = time.time() - start
        history.append(epoch_metrics)
        print(
            f"{router_name} epoch {epoch}: train_bce={epoch_metrics['train_bce']:.6f}, "
            f"test_acc={epoch_metrics['accuracy']:.4f}, auc={epoch_metrics['auc']:.4f}, "
            f"route_acc={epoch_metrics['route_accuracy']:.4f}"
        )

        if epoch_metrics["route_accuracy"] > best_route_accuracy:
            best_route_accuracy = epoch_metrics["route_accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            torch.save(best_state, output_dir / f"{router_name}_best.pth")

    torch.save(net.state_dict(), output_dir / f"{router_name}_last.pth")
    if best_state is not None:
        net.load_state_dict(best_state)
        net.to(device)
    final_metrics = evaluate(
        router_name,
        cdm,
        test_loader,
        test_df,
        prompt_embeddings,
        num_models,
        num_categories,
        device,
        output_dir,
    )
    final_metrics["router"] = router_name
    final_metrics["best_epoch"] = int(max(history, key=lambda row: row["route_accuracy"])["epoch"])
    final_metrics["checkpoint_path"] = str(output_dir / f"{router_name}_best.pth")

    history_path = output_dir / f"{router_name}_epoch_history.csv"
    pd.DataFrame(history).to_csv(history_path, index=False)
    final_metrics["epoch_history_path"] = str(history_path)
    return final_metrics


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    run_name = args.run_name or datetime.now().strftime("mirt_nirt_%Y%m%d_%H%M%S")
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_split(args.train_data_path)
    test_df = load_split(args.test_data_path, include_text=True)
    question_embeddings = torch.load(args.question_embedding_path, map_location="cpu").float()
    max_prompt_id = max(int(train_df["prompt_id"].max()), int(test_df["prompt_id"].max()))
    if max_prompt_id >= question_embeddings.shape[0]:
        raise ValueError(
            f"Prompt id {max_prompt_id} exceeds question embedding rows {question_embeddings.shape[0]}"
        )
    question_embeddings = question_embeddings.to(device)

    if args.model_order_path.exists():
        model_order = pd.read_csv(args.model_order_path)
        num_models = int(model_order["model_id"].max()) + 1
    else:
        num_models = int(max(train_df["model_id"].max(), test_df["model_id"].max())) + 1
    num_categories = int(max(train_df["category_id"].max(), test_df["category_id"].max())) + 1
    latent_dim = args.latent_dim or num_categories

    train_loader = make_loader(train_df, args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = make_loader(test_df, args.eval_batch_size, shuffle=False, num_workers=args.num_workers)

    run_config = {
        "train_data_path": str(args.train_data_path),
        "test_data_path": str(args.test_data_path),
        "question_embedding_path": str(args.question_embedding_path),
        "output_dir": str(output_dir),
        "routers": args.routers,
        "mirt_epochs": args.mirt_epochs,
        "nirt_epochs": args.nirt_epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "lr": args.lr,
        "latent_dim": latent_dim,
        "num_models": num_models,
        "num_categories": num_categories,
        "prompt_embedding_dim": int(question_embeddings.shape[1]),
        "device": str(device),
        "seed": args.seed,
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    print(json.dumps(run_config, indent=2))

    summaries = []
    if "mirt" in args.routers:
        mirt = MIRT.MIRT(num_models, question_embeddings.shape[1], latent_dim)
        summaries.append(
            train_router(
                "mirt",
                mirt,
                train_loader,
                test_loader,
                test_df,
                question_embeddings,
                num_models,
                num_categories,
                device,
                args.mirt_epochs,
                args.lr,
                output_dir,
            )
        )
    if "nirt" in args.routers:
        nirt = NIRT.NIRT(num_models, question_embeddings.shape[1], num_categories)
        summaries.append(
            train_router(
                "nirt",
                nirt,
                train_loader,
                test_loader,
                test_df,
                question_embeddings,
                num_models,
                num_categories,
                device,
                args.nirt_epochs,
                args.lr,
                output_dir,
            )
        )

    summary_df = pd.DataFrame(summaries)
    summary_path = output_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False)
    (output_dir / "summary_metrics.json").write_text(json.dumps(summaries, indent=2))
    print("\nFinal summary")
    print(summary_df.to_string(index=False))
    print(f"\nSaved results to {output_dir}")


if __name__ == "__main__":
    main()
