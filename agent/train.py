# train.py
import torch
from environment.env_test import WarehouseEnv
from MAPPO import MAPPO, Memory
from conj import config

# Visdom初始化 - 使用时间戳确保唯一性
try:
    from visdom import Visdom

    # 创建唯一的环境名
    env_name = "warehouse_test"
    viz = Visdom(port=8097, env=env_name)
    viz_enabled = True

    # 先关闭可能存在的旧窗口
    viz.close(win="reward_curve")

    # 创建新窗口
    viz.line(
        [0], [0],
        win="reward_curve",
        opts=dict(
            title=f"Training Reward - {env_name}",
            xlabel='Episode',
            ylabel='Reward',
            showlegend=True
        )
    )
    print(f"Visdom enabled with env: {env_name}")
except Exception as e:
    viz_enabled = False
    print(f"Visdom not available: {e}")
    reward_history = []


def main():
    env = WarehouseEnv()
    agent = MAPPO(config)
    memory = Memory(config.n_amrs, config.n_pickers)
    max_episodes = 2000
    max_steps = 200

    for ep in range(max_episodes):
        obs, _ = env.reset()
        obs = torch.tensor(obs, dtype=torch.float32)
        memory.clear()

        ep_reward = 0.0

        # rollout
        for step in range(max_steps):
            amr_pairs, picker_pairs = env.get_action_pairs()
            result = agent.select_actions(obs, amr_pairs, picker_pairs)

            action_list = result["amr_actions"] + result["picker_actions"]
            next_obs, reward, terminated, truncated, _ = env.step(action_list)
            next_obs = torch.tensor(next_obs, dtype=torch.float32)

            # store into memory
            memory.states.append(obs.clone().detach().cpu())
            memory.rewards.append(float(reward))
            memory.amr_pairs.append(amr_pairs.clone().detach().cpu())
            memory.picker_pairs.append(picker_pairs.clone().detach().cpu())
            memory.amr_actions.append(result["amr_actions"])
            memory.picker_actions.append(result["picker_actions"])
            memory.amr_log_probs.append(result["amr_log_probs"])
            memory.picker_log_probs.append(result["picker_log_probs"])
            memory.amr_values.append(result["amr_values"])
            memory.picker_values.append(result["picker_values"])

            obs = next_obs
            ep_reward += reward
            if terminated or truncated:
                break

        # after rollout: compute returns & advantages (GAE)
        with torch.no_grad():
            state_tensor = torch.stack(memory.states).to(agent.device)
            state_values = agent.critic(agent.cnn(state_tensor)).squeeze(-1)
            state_values = state_values.cpu().numpy().tolist()

        state_values = state_values[:len(memory.rewards)]
        returns, advs = agent.compute_gae_returns(memory.rewards, state_values, last_value=0.0)
        memory.returns = returns
        memory.advantages = advs

        agent.update(memory)
        memory.clear()

        print(f"EP {ep}, Reward={ep_reward:.4f}")

        # 更新图表 - 确保数据格式正确
        if viz_enabled:
            try:
                viz.line(
                    [ep_reward], [ep],
                    win="reward_curve",
                    update='append',
                    name='Episode Reward'
                )
            except Exception as e:
                print(f"Visdom update failed: {e}")
                viz_enabled = False


if __name__ == "__main__":
    main()