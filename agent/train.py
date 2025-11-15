# train.py - 只使用 Visdom
import torch
from environment.env_test import WarehouseEnv
from MAPPO import MAPPO, Memory
from conj import config


def main():
    try:
        from visdom import Visdom
        env_name = "warehouse_test"
        viz = Visdom(port=8097, env=env_name)

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
        print(f"✅ Visdom enabled with env: {env_name}")

    except Exception as e:
        print(f"❌ Visdom disabled: {e}")
        return

    # 环境初始化
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

        for step in range(max_steps):
            amr_pairs, picker_pairs = env.get_action_pairs()
            result = agent.select_actions(obs, amr_pairs, picker_pairs)

            action_list = result["amr_actions"] + result["picker_actions"]
            next_obs, reward, terminated, truncated, _ = env.step(action_list)
            next_obs = torch.tensor(next_obs, dtype=torch.float32)

            # 存储经验
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

        # 计算GAE和returns
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

        # 更新 Visdom 图表
        viz.line(
            [ep_reward], [ep],
            win="reward_curve",
            update='append',
            name='Episode Reward'
        )

        # 保存模型
        if ep % 100 == 0:
            torch.save(agent.state_dict(), f"checkpoints/mappo_ep{ep}.pth")


if __name__ == "__main__":
    main()