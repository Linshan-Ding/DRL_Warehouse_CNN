#只做参数的存放，并无其他用处
from types import SimpleNamespace
import torch
# trajectory：轨迹；
# batch：经验批次
# memory：经验池的一批数据
config = SimpleNamespace(
    cnn_output_dim = 256,
    gamma=0.99,
    lam=0.95,
    actor_lr=1e-4,
    critic_lr=3e-5,
    ppo_clip=0.2,
    value_coef=0.2,
    entropy_coef=0.01,
    max_grad_norm=0.5,
    epochs=2,
    batch_size=64,
    dt_scale = 10.0,
    a3c_num_workers = 2,
    device='cuda' if torch.cuda.is_available() else 'cpu',
)