#只做参数的存放，并无其他用处
from types import SimpleNamespace
import torch
# trajectory：轨迹；
# batch：经验批次
# memory：经验池的一批数据
config = SimpleNamespace(
    grid_h=20,
    grid_w=20,
    n_amrs=8,
    n_pickers=5,
    cnn_feat_dim=256,
    action_pair_dim=4,
    atn_embed_dim=128,
    gamma=0.99,
    lam=0.95,
    actor_lr=3e-4,
    critic_lr=1e-3,
    ppo_clip=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    epochs=3,
    minibatch_size=64,
    device='cuda' if torch.cuda.is_available() else 'cpu',
)