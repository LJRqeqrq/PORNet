
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from itertools import combinations
import os
import ast
import random
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    precision_recall_curve,
)

def lsift_centers(X, y, labels_subset, r=0.1):
    X = np.asarray(X)
    y = np.asarray(y)
    y_sub = y[:, labels_subset]          
    k = len(labels_subset)
    pattern_idx = (y_sub * (2 ** np.arange(k))).sum(axis=1).astype(int)
    group = 2 ** k
    centers = []

    for g in range(group):
        idx = np.where(pattern_idx == g)[0]
        if len(idx) == 0:
            continue
        c = max(1, int(np.ceil(r * len(idx))))
        km = KMeans(n_clusters=c, n_init="auto").fit(X[idx])
        centers.append(km.cluster_centers_)

    if len(centers) == 0:
        return np.zeros((0, X.shape[1]))
    return np.vstack(centers)


def top_label_pairs(y, top_k=10):
    y = np.asarray(y)
    L = y.shape[1]
    co_scores = []
    for i in range(L):
        for j in range(i + 1, L):
            score = np.mean(y[:, i] * y[:, j])
            co_scores.append(((i, j), score))
    co_scores.sort(key=lambda x: x[1], reverse=True)
    return [p for (p, _) in co_scores[:top_k]]


def build_lsift_center_bank(X_train, y_train, label_pairs, r=0.1):
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    center_bank = {}
    for subset in label_pairs:
        centers = lsift_centers(X_train, y_train, subset, r=r)
        center_bank[tuple(subset)] = centers
    return center_bank


def lsift_transform(X, center_bank):
    X = np.asarray(X)
    N = X.shape[0]
    new_feats = []
    for subset, centers in center_bank.items():
        if centers is None or centers.shape[0] == 0:
            dists = np.zeros((N, 1), dtype=float)
        else:
            dists = np.min(pairwise_distances(X, centers), axis=1).reshape(-1, 1)
        new_feats.append(dists)
    if len(new_feats) == 0:
        return X
    return np.hstack([X] + new_feats)



class SelfAttnBlock(nn.Module):
    """简单的模态内自注意力块，输入 (B, D_in) -> 输出 (B, hidden_dim)"""

    def __init__(self, in_dim, hidden_dim, dropout=0.1, n_heads=4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)
        )

    def forward(self, x):
        h = self.proj(x)  # (B, H)
        h_seq = h.unsqueeze(1)  # (B, 1, H)
        attn_out, _ = self.attn(h_seq, h_seq, h_seq)  # (B, 1, H)
        out = self.norm(h + attn_out.squeeze(1))
        out = out + self.ff(out)
        return out  # (B, H)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, n_heads=4, dropout=0.1):
        super().__init__()
        self.cross = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, q, k, v):
        q_seq = q.unsqueeze(1)
        k_seq = k.unsqueeze(1)
        v_seq = v.unsqueeze(1)
        attn_out, _ = self.cross(q_seq, k_seq, v_seq)  # (B,1,H)
        out = q + attn_out.squeeze(1)
        out = self.norm(out)
        out = out + self.ff(out)
        return out  # (B, H)


class MultiModalSelfCrossFusion(nn.Module):
    def __init__(
        self,
        lis_dim,
        text_dim,
        doctor_dim,
        num_dim,
        pacs_dim,
        operating_doctor_dim,
        hidden_dim,
        out_dim,
        dropout=0.2,
        n_heads=4,
        cross_pairs=None,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.lis_block = SelfAttnBlock(
            lis_dim, hidden_dim, dropout=dropout, n_heads=n_heads
        )
        self.text_block = SelfAttnBlock(
            text_dim, hidden_dim, dropout=dropout, n_heads=n_heads
        )
        self.doctor_block = SelfAttnBlock(
            doctor_dim, hidden_dim, dropout=dropout, n_heads=n_heads
        )
        self.num_block = SelfAttnBlock(
            num_dim, hidden_dim, dropout=dropout, n_heads=n_heads
        )
        self.pacs_block = SelfAttnBlock(pacs_dim, hidden_dim, dropout, n_heads)
        self.operating_doctor_block = SelfAttnBlock(
            operating_doctor_dim, hidden_dim, dropout=dropout, n_heads=n_heads
        )
        self.cross_blocks = nn.ModuleDict()
        modalities = [
            "lis",
            "text",
            "doctor",
            "num",
            "pacs",
            "operating_doctor",
        ]

        if cross_pairs is None:
            cross_pairs = []
            for q in modalities:
                for kv in modalities:
                    if q != kv:
                        cross_pairs.append((q, kv))
        for q, kv in cross_pairs:
            self.cross_blocks[f"{q}_q_{kv}_kv"] = CrossAttentionBlock(
                hidden_dim, n_heads=n_heads, dropout=dropout
            )
        fusion_in_dim = hidden_dim * 6
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(
        self,
        lis_feat,
        text_feat,
        doctor_feat,
        num_feat,
        pacs_feat,
        operating_doctor_feat,
    ):
        lis_h = self.lis_block(lis_feat)
        text_h = self.text_block(text_feat)
        doctor_h = self.doctor_block(doctor_feat)
        num_h = self.num_block(num_feat)
        pacs_h = self.pacs_block(pacs_feat)
        operating_doctor_h = self.operating_doctor_block(operating_doctor_feat)

        def cross_aggregate(query_name, query_h):
            outs = []
            for kv_name in [
                "lis",
                "text",
                "doctor",
                "num",
                "pacs",
                "operating_doctor",
            ]:
                if kv_name == query_name:
                    continue
                key = f"{query_name}_q_{kv_name}_kv"
                if key in self.cross_blocks:
                    kv_h = {
                        "lis": lis_h,
                        "text": text_h,
                        "doctor": doctor_h,
                        "num": num_h,
                        "pacs": pacs_h,
                        "operating_doctor": operating_doctor_h,
                    }[kv_name]
                    outs.append(self.cross_blocks[key](query_h, kv_h, kv_h))
            if len(outs) == 0:
                return query_h
            stacked = torch.stack(outs, dim=0).mean(dim=0)
            return (query_h + stacked) / 2.0

        lis_cross = cross_aggregate("lis", lis_h)
        text_cross = cross_aggregate("text", text_h)
        doctor_cross = cross_aggregate("doctor", doctor_h)
        num_cross = cross_aggregate("num", num_h)
        pacs_cross = cross_aggregate("pacs", pacs_h)
        operating_doctor_cross = cross_aggregate("operating_doctor", operating_doctor_h)

        fused_all = torch.cat(
            [
                lis_cross,
                text_cross,
                doctor_cross,
                num_cross,
                pacs_cross,
                operating_doctor_cross,
            ],
            dim=1,
        )
        out = self.fusion(fused_all)
        return out


class PredictionHead(nn.Module):
    def __init__(self, input_dim, num_labels, dropout=0.3):
        super().__init__()
        hidden = max(8, input_dim // 2)
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_labels),
        )

    def forward(self, x):
        return self.classifier(x)


class FullModel(nn.Module):
    def __init__(self, fusion_model, pred_head):
        super().__init__()
        self.fusion = fusion_model
        self.pred_head = pred_head

    def forward(
        self,
        lis_feat,
        text_feat,
        doctor_feat,
        num_feat,
        pacs_feat,
        operating_doctor_feat,
    ):
        fused = self.fusion(
            lis_feat, text_feat, doctor_feat, num_feat, pacs_feat, operating_doctor_feat
        )
        logits = self.pred_head(fused)
        return logits
    
class LabelSmoothingBCELoss(nn.Module):
    def __init__(self, smoothing=0.05, pos_weight=None):
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight
        self.bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    def forward(self, logits, targets):
        smoothed_targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        loss = self.bce(logits, smoothed_targets)
        return loss.mean()
    
def search_best_threshold(y_true, y_probs, acc_weight=0.5):
    if y_true.ndim == 2 and y_true.shape[1] > 1:
        best_thresh = []
        for i in range(y_true.shape[1]):
            prec, rec, th = precision_recall_curve(y_true[:, i], y_probs[:, i])
            f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
            acc_scores = []
            for t in th:
                preds = (y_probs[:, i] >= t).astype(int)
                acc_scores.append(accuracy_score(y_true[:, i], preds))

            scores = (
                f1_scores[: len(acc_scores)] * (1 - acc_weight)
                + np.array(acc_scores) * acc_weight
            )
            if len(th) > 0 and len(scores) > 0:
                best_t = th[np.argmax(scores)]
            else:
                best_t = 0.5
            best_thresh.append(float(best_t))

        preds_bin = (y_probs >= np.array(best_thresh)).astype(int)
        avg_f1 = f1_score(y_true, preds_bin, average="macro", zero_division=0)
        avg_acc = accuracy_score(y_true, preds_bin)
        final_score = avg_f1 * (1 - acc_weight) + avg_acc * acc_weight
        return best_thresh, final_score, avg_f1, avg_acc
    else:
        prec, rec, th = precision_recall_curve(y_true.ravel(), y_probs.ravel())
        f1_scores = 2 * prec * rec / (prec + rec + 1e-8)

        acc_scores = []
        for t in th:
            preds = (y_probs.ravel() >= t).astype(int)
            acc_scores.append(accuracy_score(y_true.ravel(), preds))

        scores = (
            f1_scores[: len(acc_scores)] * (1 - acc_weight)
            + np.array(acc_scores) * acc_weight
        )
        if len(th) > 0 and len(scores) > 0:
            best_t = float(th[np.argmax(scores)])
            best_score = float(np.max(scores))
            best_f1 = float(f1_scores[np.argmax(scores)])
            best_acc = float(acc_scores[np.argmax(scores)])
        else:
            best_t, best_score, best_f1, best_acc = 0.5, 0.0, 0.0, 0.0
        return best_t, best_score, best_f1, best_acc

def bootstrap_ci(y_true, y_pred, metric_fn, B=1000, alpha=0.05):
    n = len(y_true)
    stats = []
    for _ in range(B):
        idx = np.random.choice(n, n, replace=True)
        stats.append(metric_fn(y_true[idx], y_pred[idx]))
    lower = np.percentile(stats, 100 * alpha / 2)
    upper = np.percentile(stats, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


def compute_metrics(y_true, y_probs, threshold=0.5):
    if isinstance(threshold, list) and y_true.ndim == 2:
        preds_bin = np.zeros_like(y_probs, dtype=int)
        for i in range(y_probs.shape[1]):
            preds_bin[:, i] = (y_probs[:, i] >= threshold[i]).astype(int)
    else:
        preds_bin = (y_probs >= threshold).astype(int)
    def macro_auc_fn(a, b):
        try:
            return roc_auc_score(a, b, average="macro", multi_class="ovr")  
        except Exception:
            return np.nan

    def micro_auc_fn(a, b):
        try:
            return roc_auc_score(a, b, average="micro", multi_class="ovr") 
        except Exception:
            return np.nan

    def macro_f1_fn(a, b):
        return f1_score(a, b, average="macro", zero_division=0)

    def micro_f1_fn(a, b):
        return f1_score(a, b, average="micro", zero_division=0)

    def macro_prec_fn(a, b):
        return precision_score(a, b, average="macro", zero_division=0)

    def micro_prec_fn(a, b):
        return precision_score(a, b, average="micro", zero_division=0)

    def macro_rec_fn(a, b):
        return recall_score(a, b, average="macro", zero_division=0)

    def micro_rec_fn(a, b):
        return recall_score(a, b, average="micro", zero_division=0)

    def sample_acc_fn(a, b):
        return accuracy_score(a, b)

    def macro_acc_fn(a, b):
        if a.ndim == 1:
            return accuracy_score(a, b)
        acc_list = [accuracy_score(a[:, i], b[:, i]) for i in range(a.shape[1])]
        return np.mean(acc_list)

    def micro_acc_fn(a, b):
        if a.ndim == 1:
            return accuracy_score(a, b)
        total_correct = (a == b).sum()  
        total_samples = a.size 
        return total_correct / total_samples if total_samples > 0 else 0.0
    macro_auc = macro_auc_fn(y_true, y_probs)
    micro_auc = micro_auc_fn(y_true, y_probs)
    macro_f1 = macro_f1_fn(y_true, preds_bin)
    micro_f1 = micro_f1_fn(y_true, preds_bin)
    macro_prec = macro_prec_fn(y_true, preds_bin)
    micro_prec = micro_prec_fn(y_true, preds_bin)
    macro_rec = macro_rec_fn(y_true, preds_bin)
    micro_rec = micro_rec_fn(y_true, preds_bin)
    sample_acc = sample_acc_fn(y_true, preds_bin)  
    macro_acc = macro_acc_fn(y_true, preds_bin)
    micro_acc = micro_acc_fn(y_true, preds_bin)
    macro_auc_ci = bootstrap_ci(y_true, y_probs, macro_auc_fn)
    micro_auc_ci = bootstrap_ci(y_true, y_probs, micro_auc_fn)
    macro_f1_ci = bootstrap_ci(y_true, preds_bin, macro_f1_fn)
    micro_f1_ci = bootstrap_ci(y_true, preds_bin, micro_f1_fn)
    macro_prec_ci = bootstrap_ci(y_true, preds_bin, macro_prec_fn)
    micro_prec_ci = bootstrap_ci(y_true, preds_bin, micro_prec_fn)
    macro_rec_ci = bootstrap_ci(y_true, preds_bin, macro_rec_fn)
    micro_rec_ci = bootstrap_ci(y_true, preds_bin, micro_rec_fn)
    sample_acc_ci = bootstrap_ci(y_true, preds_bin, sample_acc_fn)
    macro_acc_ci = bootstrap_ci(y_true, preds_bin, macro_acc_fn)
    micro_acc_ci = bootstrap_ci(y_true, preds_bin, micro_acc_fn)

    result = {
        "Macro_AUC": round(macro_auc, 4) if not np.isnan(macro_auc) else np.nan,
        "Macro_AUC_CI": macro_auc_ci,
        "Micro_AUC": round(micro_auc, 4) if not np.isnan(micro_auc) else np.nan,
        "Micro_AUC_CI": micro_auc_ci,
        # 3. F1（Macro + Micro）
        "Macro_F1": round(macro_f1, 4),
        "Macro_F1_CI": macro_f1_ci,
        "Micro_F1": round(micro_f1, 4),
        "Micro_F1_CI": micro_f1_ci,
        # 4. Precision（Macro + Micro）
        "Macro_Precision": round(macro_prec, 4),
        "Macro_Precision_CI": macro_prec_ci,
        "Micro_Precision": round(micro_prec, 4),
        "Micro_Precision_CI": micro_prec_ci,
        # 5. Recall（Macro + Micro）
        "Macro_Recall": round(macro_rec, 4),
        "Macro_Recall_CI": macro_rec_ci,
        "Micro_Recall": round(micro_rec, 4),
        "Micro_Recall_CI": micro_rec_ci,
        "Macro_Accuracy": round(macro_acc, 4),
        "Macro_Accuracy_CI": macro_acc_ci,
        "Micro_Accuracy": round(micro_acc, 4),
        "Micro_Accuracy_CI": micro_acc_ci,
        "class_metrics": [],
    }

    if y_true.ndim == 2 and y_true.shape[1] > 1:
        class_metrics = []
        for i in range(y_true.shape[1]):
            yt = y_true[:, i]
            yp = y_probs[:, i]
            pb = preds_bin[:, i]

            try:
                auc_i = float(roc_auc_score(yt, yp))
            except Exception:
                auc_i = np.nan
            f1_i = f1_score(yt, pb, zero_division=0)
            acc_i = accuracy_score(yt, pb)
            prec_i = precision_score(yt, pb, zero_division=0)
            rec_i = recall_score(yt, pb, zero_division=0)

            # single label 
            auc_ci_i = bootstrap_ci(
                yt, yp, lambda a, b: roc_auc_score(a, b) if len(set(a)) > 1 else np.nan
            )
            f1_ci_i = bootstrap_ci(yt, pb, lambda a, b: f1_score(a, b, zero_division=0))
            acc_ci_i = bootstrap_ci(yt, pb, accuracy_score)
            prec_ci_i = bootstrap_ci(
                yt, pb, lambda a, b: precision_score(a, b, zero_division=0)
            )
            rec_ci_i = bootstrap_ci(
                yt, pb, lambda a, b: recall_score(a, b, zero_division=0)
            )

            class_metrics.append(
                {
                    "class": i,
                    "AUC": auc_i,
                    "AUC_CI": auc_ci_i,
                    "F1": round(f1_i, 4),
                    "F1_CI": f1_ci_i,
                    "Accuracy": round(acc_i, 4),
                    "Accuracy_CI": acc_ci_i,
                    "Precision": round(prec_i, 4),
                    "Precision_CI": prec_ci_i,
                    "Recall": round(rec_i, 4),
                    "Recall_CI": rec_ci_i,
                }
            )

        result["class_metrics"] = class_metrics

    return result


def format_float_cols(df):
    for col in df.columns:
        if df[col].dtype == float:
            df[col] = df[col].map(lambda x: f"{x:.4f}")
    return df


# ================= train & eval =================
def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=40,
    lr=5e-4,
    patience=8,
    weight_decay=1e-5,
    min_delta=1e-4,
    smooth_window=3,
    min_lr=1e-6,
    acc_weight=0.6,
):
    all_labels = []
    for batch in train_loader:
        _, _, _, _, _, _, y = batch
        all_labels.append(y.numpy())
    all_labels = np.vstack(all_labels)

    if all_labels.ndim == 1:
        pos_weight = torch.tensor(
            [(len(all_labels) - all_labels.sum()) / (all_labels.sum() + 1e-8)],
            device=device,
        )
    else:
        pos_weight = torch.tensor(
            (all_labels.shape[0] - all_labels.sum(axis=0))
            / (all_labels.sum(axis=0) + 1e-8),
            device=device,
        )

    criterion = LabelSmoothingBCELoss(smoothing=0.05, pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=2
    )

    try:
        scaler = torch.amp.GradScaler(
            device_type="cuda" if torch.cuda.is_available() else "cpu"
        )
    except Exception:
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    best_val_score = -float("inf")
    best_val_f1, best_val_acc, best_val_loss = 0.0, 0.0, float("inf")
    best_state, best_threshold = None, None
    patience_counter = 0
    val_score_history, val_loss_history = [], []

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, total=len(train_loader), leave=True)
        loop.set_description(f"Epoch [{epoch+1}/{num_epochs}]")

        for batch in loop:
            lis, text, doctor, num, pacs, operating_doctor, y = [
                t.to(device, non_blocking=True) for t in batch
            ]
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with torch.amp.autocast(
                    device_type="cuda" if torch.cuda.is_available() else "cpu"
                ):
                    logits = model(lis, text, doctor, num, pacs, operating_doctor)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(lis, text, doctor, num, pacs, operating_doctor)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            train_loss += float(loss.item())
            loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])

        avg_train_loss = train_loss / max(1, len(train_loader))

        # val
        model.eval()
        all_preds, all_labels, val_loss = [], [], 0.0
        with torch.no_grad():
            for batch in val_loader:
                lis, text, doctor, num, pacs, operating_doctor, y = [
                    t.to(device, non_blocking=True) for t in batch
                ]
                if scaler is not None:
                    with torch.amp.autocast(
                        device_type="cuda" if torch.cuda.is_available() else "cpu"
                    ):
                        logits = model(lis, text, doctor, num, pacs, operating_doctor)
                        loss = criterion(logits, y)
                        preds = torch.sigmoid(logits).cpu().numpy()
                else:
                    logits = model(lis, text, doctor, num, pacs, operating_doctor)
                    loss = criterion(logits, y)
                    preds = torch.sigmoid(logits).cpu().numpy()
                val_loss += float(loss.item())
                all_preds.append(preds)
                all_labels.append(y.cpu().numpy())

        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        avg_val_loss = val_loss / max(1, len(val_loader))

        best_t, val_score, val_f1, val_acc = search_best_threshold(
            all_labels, all_preds, acc_weight=acc_weight
        )
        scheduler.step(epoch + val_score)

        val_score_history.append(val_score)
        val_loss_history.append(avg_val_loss)
        score_improved = val_score > best_val_score + min_delta
        loss_improved = avg_val_loss < best_val_loss - min_delta

        if score_improved or loss_improved:
            best_val_score = max(best_val_score, val_score)
            best_val_f1 = max(best_val_f1, val_f1)
            best_val_acc = max(best_val_acc, val_acc)
            best_val_loss = min(best_val_loss, avg_val_loss)
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            best_threshold = best_t
            patience_counter = 0
        else:
            patience_counter += 1

        if len(val_score_history) >= smooth_window + 1:
            recent_score_mean = np.mean(val_score_history[-smooth_window:])
            prev_score_mean = np.mean(val_score_history[-smooth_window - 1 : -1])
            recent_loss_mean = np.mean(val_loss_history[-smooth_window:])
            prev_loss_mean = np.mean(val_loss_history[-smooth_window - 1 : -1])
            if (recent_score_mean < prev_score_mean - min_delta) and (
                recent_loss_mean > prev_loss_mean + min_delta
            ):
                patience_counter += 1

        current_lr = optimizer.param_groups[0]["lr"]
        if current_lr <= min_lr:
            break
        if patience_counter >= patience:
            logging.info("early stopping triggered")
            break


    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_threshold


def evaluate_model(model, data_loader, threshold, dataset_name="", target_cols=None):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            lis, text, doctor, num, pacs, operating_doctor, y = [
                t.to(device, non_blocking=True) for t in batch
            ]
            with torch.amp.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu"
            ):
                preds = (
                    torch.sigmoid(model(lis, text, doctor, num, pacs, operating_doctor))
                    .cpu()
                    .numpy()
                )
            all_preds.append(preds)
            all_labels.append(y.cpu().numpy())
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    metrics = compute_metrics(all_labels, all_preds, threshold=threshold)

    if metrics["class_metrics"]:
        for i, m in enumerate(metrics["class_metrics"]):
            label_name = (
                target_cols[i]
                if target_cols is not None and i < len(target_cols)
                else f"label_{i}"
            )
    return metrics


