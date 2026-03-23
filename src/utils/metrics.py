"""
工具函数
"""
import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from typing import Dict, List


def compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算 AUC"""
    try:
        return roc_auc_score(y_true, y_pred)
    except:
        return 0.0


def compute_recall_at_k(y_true: List[int], y_pred: List[int], k: int = 10) -> float:
    """计算 Recall@K"""
    y_true_set = set(y_true)
    y_pred_k = y_pred[:k]
    hits = len(set(y_pred_k) & y_true_set)
    return hits / min(len(y_true_set), k)


def compute_ndcg_at_k(y_true: List[int], y_pred: List[int], k: int = 10) -> float:
    """计算 NDCG@K"""
    y_pred_k = y_pred[:k]
    dcg = sum([1.0 / np.log2(i + 2) if item in y_true else 0.0
               for i, item in enumerate(y_pred_k)])

    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(y_true), k))])

    return dcg / idcg if idcg > 0 else 0.0


def save_embeddings(model, data_loader, save_path: str, embedding_type: str = 'user'):
    """保存用户或物品的嵌入向量"""
    model.eval()
    embeddings = []
    ids = []

    with torch.no_grad():
        for batch in data_loader:
            if embedding_type == 'user':
                user_inputs, _ = batch
                emb = model.encode_user(user_inputs)
                ids.extend(user_inputs['user_id'].cpu().numpy())
            else:
                _, item_inputs = batch
                emb = model.encode_item(item_inputs)
                ids.extend(item_inputs['item_id'].cpu().numpy())

            embeddings.append(emb.cpu().numpy())

    embeddings = np.vstack(embeddings)
    np.savez(save_path, ids=np.array(ids), embeddings=embeddings)
    print(f"保存 {embedding_type} embeddings 到 {save_path}")


def load_embeddings(load_path: str):
    """加载嵌入向量"""
    data = np.load(load_path)
    return data['ids'], data['embeddings']
