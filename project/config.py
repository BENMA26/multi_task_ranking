import torch

CONFIG = {
    # 数据
    "batch_size": 4096,
    "num_workers": 4,

    # Embedding
    "embedding_dim": 16,

    # Expert（MMoE/PLE/AdaFTR）
    "num_experts": 8,
    "expert_hidden_dim": 256,

    # Task-specific Tower
    "tower_dims": [256, 128, 64],

    # PLE 专用
    "num_specific_experts": 2,
    "num_shared_experts": 2,

    # 训练
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "epochs": 10,
    "optimizer": "adam",

    # AdaFTR 专用
    "alpha": 0.1,
    "tau_min": 0.05,
    "tau_max": 0.50,
    "relatedness_hidden_dim": 64,

    # 其他
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "dropout": 0.1,
}
