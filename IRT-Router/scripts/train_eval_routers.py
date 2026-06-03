import argparse
import json
import os
import pickle
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from router import MIRT, NIRT
from test_router import config as COST_CONFIG


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def choose_device(preferred):
    if preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_records(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def records_to_matrix(records, value_key, default=None):
    max_index = max(int(record["index"]) for record in records)
    dim = len(records[0][value_key])
    if default is None:
        matrix = np.zeros((max_index + 1, dim), dtype=np.float32)
    else:
        matrix = np.tile(np.asarray(default, dtype=np.float32), (max_index + 1, 1))
    for record in records:
        matrix[int(record["index"])] = np.asarray(record[value_key], dtype=np.float32)
    return torch.tensor(matrix, dtype=torch.float32)


def load_assets(emb_name):
    llm_embeddings = records_to_matrix(load_records(f"utils/{emb_name}_embeddings/llm_embeddings.pkl"), "embedding")
    query_embeddings = records_to_matrix(load_records(f"utils/{emb_name}_embeddings/query_embeddings.pkl"), "embedding")
    train_relevance = records_to_matrix(
        load_records(f"utils/relevance/relevance_vectors_cluster_train_{emb_name}.pkl"),
        "relevance_vector",
        default=np.ones(25, dtype=np.float32),
    )
    test_relevance = records_to_matrix(
        load_records(f"utils/relevance/relevance_vectors_cluster_test_{emb_name}.pkl"),
        "relevance_vector",
        default=np.ones(25, dtype=np.float32),
    )
    cold_embeddings = records_to_matrix(
        load_records(f"utils/cold/test_avg_embeddings_{emb_name}.pkl"),
        "avg_embedding",
        default=np.zeros(query_embeddings.shape[1], dtype=np.float32),
    )
    llm_id_map = pd.read_csv("utils/map/llm.csv", index_col="name").to_dict()["index"]
    query_id_map = pd.read_csv("utils/map/query.csv", index_col="question").to_dict()["index"]
    return {
        "llm_embeddings": llm_embeddings,
        "query_embeddings": query_embeddings,
        "train_relevance": train_relevance,
        "test_relevance": test_relevance,
        "cold_embeddings": cold_embeddings,
        "llm_id_map": llm_id_map,
        "query_id_map": query_id_map,
    }


def ids_from_frame(df, assets):
    llm_ids = df["llm"].map(assets["llm_id_map"])
    query_ids = df["question"].map(assets["query_id_map"])
    valid = llm_ids.notna() & query_ids.notna()
    if not valid.all():
        missing = int((~valid).sum())
        print(f"Skipping {missing} rows with missing llm/question embeddings")
        df = df.loc[valid].copy()
        llm_ids = llm_ids.loc[valid]
        query_ids = query_ids.loc[valid]
    y = df["performance"].astype(float).to_numpy(dtype=np.float32)
    return (
        torch.tensor(llm_ids.to_numpy(dtype=np.int64), dtype=torch.long),
        torch.tensor(query_ids.to_numpy(dtype=np.int64), dtype=torch.long),
        torch.tensor(y, dtype=torch.float32),
        df,
    )


def make_loader(llm_ids, query_ids, y, batch_size, shuffle):
    return DataLoader(TensorDataset(llm_ids, query_ids, y), batch_size=batch_size, shuffle=shuffle)


def binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    y_true_bin = y_true >= 0.5
    y_pred_bin = y_pred >= 0.5
    try:
        auc = float(roc_auc_score(y_true_bin, y_pred))
    except ValueError:
        auc = None
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "auc": auc,
        "accuracy": float(accuracy_score(y_true_bin, y_pred_bin)),
    }


def train_mirt(model, train_loader, assets, device, epochs, lr):
    net = model.irt_net.to(device)
    net.train()
    llm_emb = assets["llm_embeddings"].to(device)
    query_emb = assets["query_embeddings"].to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        losses = []
        for llm_ids, query_ids, y in tqdm(train_loader, desc=f"MIRT epoch {epoch + 1}/{epochs}"):
            llm = llm_emb[llm_ids.to(device)]
            query = query_emb[query_ids.to(device)]
            target = y.to(device)
            pred, _, _, _ = net(llm, query)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses))})
        print(f"MIRT epoch {epoch + 1}: train_loss={history[-1]['train_loss']:.6f}")
    return history


def train_nirt(model, train_loader, assets, device, epochs, lr):
    net = model.nirt_net.to(device)
    net.train()
    llm_emb = assets["llm_embeddings"].to(device)
    query_emb = assets["query_embeddings"].to(device)
    relevance = assets["train_relevance"].to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        losses = []
        for llm_ids, query_ids, y in tqdm(train_loader, desc=f"NIRT epoch {epoch + 1}/{epochs}"):
            llm = llm_emb[llm_ids.to(device)]
            query = query_emb[query_ids.to(device)]
            rel = relevance[query_ids.to(device)]
            target = y.to(device)
            pred, _, _, _, _ = net(llm, query, rel)
            loss = loss_fn(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses))})
        print(f"NIRT epoch {epoch + 1}: train_loss={history[-1]['train_loss']:.6f}")
    return history


@torch.no_grad()
def evaluate_prediction(router_name, model, loader, assets, device, relevance_key="test_relevance"):
    if router_name == "mirt":
        net = model.irt_net.to(device)
    else:
        net = model.nirt_net.to(device)
    net.eval()
    llm_emb = assets["llm_embeddings"].to(device)
    query_emb = assets["query_embeddings"].to(device)
    relevance = assets[relevance_key].to(device)
    y_true, y_pred = [], []
    for llm_ids, query_ids, y in tqdm(loader, desc=f"eval {router_name}"):
        llm_ids = llm_ids.to(device)
        query_ids = query_ids.to(device)
        llm = llm_emb[llm_ids]
        query = query_emb[query_ids]
        if router_name == "mirt":
            pred, _, _, _ = net(llm, query)
        else:
            pred, _, _, _, _ = net(llm, query, relevance[query_ids])
        y_true.extend(y.cpu().numpy().tolist())
        y_pred.extend(pred.detach().cpu().numpy().tolist())
    return binary_metrics(y_true, y_pred)


@torch.no_grad()
def evaluate_routing(router_name, model, df, assets, device, a=0.8, lamda=0.3, relevance_key="test_relevance"):
    if router_name == "mirt":
        net = model.irt_net.to(device)
    else:
        net = model.nirt_net.to(device)
    net.eval()
    llm_emb = assets["llm_embeddings"].to(device)
    query_emb = assets["query_embeddings"].to(device)
    cold_emb = assets["cold_embeddings"].to(device)
    relevance = assets[relevance_key].to(device)
    llm_id_map = assets["llm_id_map"]
    query_id_map = assets["query_id_map"]

    performances = []
    costs = []
    skipped_questions = 0
    candidate_counts = []
    score_cost_weight = -(1.0 - a)

    for question, group in tqdm(df.groupby("question", sort=False), desc=f"route {router_name}"):
        query_id = query_id_map.get(question)
        if query_id is None:
            skipped_questions += 1
            continue
        candidates = []
        candidate_rows = []
        for _, row in group.drop_duplicates("llm").iterrows():
            llm_name = row["llm"]
            if llm_name not in llm_id_map or llm_name not in COST_CONFIG:
                continue
            candidates.append((llm_name, int(llm_id_map[llm_name])))
            candidate_rows.append(row)
        if not candidates:
            skipped_questions += 1
            continue

        llm_ids = torch.tensor([item[1] for item in candidates], dtype=torch.long, device=device)
        qid = torch.tensor([int(query_id)] * len(candidates), dtype=torch.long, device=device)
        query = (1.0 - lamda) * query_emb[qid] + lamda * cold_emb[qid]
        if router_name == "mirt":
            pred, _, _, _ = net(llm_emb[llm_ids], query)
        else:
            pred, _, _, _, _ = net(llm_emb[llm_ids], query, relevance[qid])
        pred = pred.detach().cpu().numpy()

        scores = []
        for idx, (llm_name, _) in enumerate(candidates):
            proxy_cost = COST_CONFIG[llm_name]["output_cost"] * 1e5
            scores.append(a * float(pred[idx]) + score_cost_weight * proxy_cost)
        best_idx = int(np.argmax(scores))
        best_row = candidate_rows[best_idx]
        best_llm = best_row["llm"]
        performances.append(float(best_row["performance"]))
        costs.append(
            float(best_row["input_tokens"]) * COST_CONFIG[best_llm]["input_cost"]
            + float(best_row["output_tokens"]) * COST_CONFIG[best_llm]["output_cost"]
        )
        candidate_counts.append(len(candidates))

    return {
        "questions_evaluated": len(performances),
        "questions_skipped": skipped_questions,
        "avg_candidate_count": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "avg_performance": float(np.mean(performances)) if performances else None,
        "total_cost": float(np.sum(costs)) if costs else None,
        "avg_cost": float(np.mean(costs)) if costs else None,
        "a": a,
        "lamda": lamda,
    }


def save_outputs(out_dir, results, histories):
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "router_eval_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    rows = []
    for router_name, router_results in results["routers"].items():
        for split_name, split_results in router_results["tests"].items():
            row = {
                "router": router_name,
                "split": split_name,
                **{f"prediction_{k}": v for k, v in split_results["prediction"].items()},
                **{f"routing_{k}": v for k, v in split_results["routing"].items()},
            }
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_dir / "router_eval_summary.csv", index=False)

    history_rows = []
    for router_name, history in histories.items():
        for record in history:
            history_rows.append({"router": router_name, **record})
    pd.DataFrame(history_rows).to_csv(out_dir / "training_history.csv", index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--emb-name", default="bert")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--mirt-epochs", type=int, default=9)
    parser.add_argument("--nirt-epochs", type=int, default=5)
    parser.add_argument("--mirt-lr", type=float, default=0.001)
    parser.add_argument("--nirt-lr", type=float, default=0.001)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--a", type=float, default=0.8)
    parser.add_argument("--lamda", type=float, default=0.3)
    parser.add_argument("--out-dir", default="reports/router_eval")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = choose_device(args.device)
    print(f"device: {device}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assets = load_assets(args.emb_name)
    train_df = pd.read_csv("data/train.csv")
    test_frames = {
        "test1": pd.read_csv("data/test1.csv"),
        "test2": pd.read_csv("data/test2.csv"),
    }

    train_llm_ids, train_query_ids, train_y, filtered_train_df = ids_from_frame(train_df, assets)
    train_loader = make_loader(train_llm_ids, train_query_ids, train_y, args.batch_size, shuffle=True)
    test_loaders = {}
    filtered_tests = {}
    for split_name, df in test_frames.items():
        llm_ids, query_ids, y, filtered_df = ids_from_frame(df, assets)
        test_loaders[split_name] = make_loader(llm_ids, query_ids, y, args.batch_size, shuffle=False)
        filtered_tests[split_name] = filtered_df

    llm_dim = int(assets["llm_embeddings"].shape[1])
    query_dim = int(assets["query_embeddings"].shape[1])
    knowledge_n = 25

    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    results = {
        "started_at": started_at,
        "device": str(device),
        "embedding": args.emb_name,
        "train_rows": int(len(filtered_train_df)),
        "tests": {name: int(len(df)) for name, df in filtered_tests.items()},
        "routers": {},
    }
    histories = {}

    mirt = MIRT.MIRT(llm_dim, query_dim, knowledge_n)
    histories["mirt"] = train_mirt(mirt, train_loader, assets, device, args.mirt_epochs, args.mirt_lr)
    mirt_path = out_dir / f"mirt_{args.emb_name}_retrained.snapshot"
    mirt.save(mirt_path)
    results["routers"]["mirt"] = {"snapshot": str(mirt_path), "tests": {}}

    nirt = NIRT.NIRT(llm_dim, query_dim, knowledge_n)
    histories["nirt"] = train_nirt(nirt, train_loader, assets, device, args.nirt_epochs, args.nirt_lr)
    nirt_path = out_dir / f"nirt_{args.emb_name}_retrained.snapshot"
    nirt.save(nirt_path)
    results["routers"]["nirt"] = {"snapshot": str(nirt_path), "tests": {}}

    for router_name, model in [("mirt", mirt), ("nirt", nirt)]:
        for split_name, loader in test_loaders.items():
            prediction = evaluate_prediction(router_name, model, loader, assets, device)
            routing = evaluate_routing(
                router_name,
                model,
                filtered_tests[split_name],
                assets,
                device,
                a=args.a,
                lamda=args.lamda,
            )
            results["routers"][router_name]["tests"][split_name] = {
                "prediction": prediction,
                "routing": routing,
            }
            print(router_name, split_name, prediction, routing)

    results["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    save_outputs(out_dir, results, histories)
    print(f"saved results to {out_dir}")


if __name__ == "__main__":
    main()
