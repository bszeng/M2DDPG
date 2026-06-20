import argparse
import os

import numpy as np
import torch

import DDPG_Buffer
import utils

import environment
from datetime import datetime

def whiten(state):
    return (state - np.mean(state)) / np.std(state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Choose the type of the experiment
    parser.add_argument('--experiment_type', default='custom', choices=['custom', 'power', 'rsi_elements', 'learning_rate', 'decay'],
                        help='Choose one of the experiment types to reproduce the learning curves given in the paper')

    # Training-specific parameters
    parser.add_argument("--policy", default="DDPG", help='Algorithm (default: DDPG)')
    parser.add_argument("--env", default="RIS_MISO", help='OpenAI Gym environment name')
    parser.add_argument("--seed", default=0, type=int, help='Seed number for PyTorch and NumPy (default: 0)')
    parser.add_argument("--gpu", default="0", type=int, help='GPU ordinal for multi-GPU computers (default: 0)')
    parser.add_argument("--start_time_steps", default=32, type=int, metavar='N', help='Number of exploration time steps sampling random actions (default: 0)')
    parser.add_argument("--buffer_size", default=500, type=int, help='Size of the experience replay buffer (default: 100000)')
    parser.add_argument("--batch_size", default=16, metavar='N', help='Batch size (default: 16)')
    parser.add_argument("--save_model", action="store_true", help='Save model and optimizer parameters')
    parser.add_argument("--load_model", default="", help='Model load file name; if empty, does not load')

    # Environment-specific parameters
    parser.add_argument("--num_antennas", default=4, type=int, metavar='N', help='Number of antennas in the BS')
    parser.add_argument("--num_RIS_elements", default=4, type=int, metavar='N', help='Number of RIS elements')
    parser.add_argument("--num_users", default=4, type=int, metavar='N', help='Number of users')
    parser.add_argument("--power_t", default=30, type=float, metavar='N', help='Transmission power for the constrained optimization in dB')
    parser.add_argument("--num_time_steps_per_eps", default=1000, type=int, metavar='N', help='Maximum number of steps per episode (default: 20000)')
    parser.add_argument("--num_eps", default=10000, type=int, metavar='N', help='Maximum number of episodes (default: 5000)')
    parser.add_argument("--awgn_var", default=1e-2, type=float, metavar='G', help='Variance of the additive white Gaussian noise (default: 0.01)')
    parser.add_argument("--channel_est_error", default=False, type=bool, help='Noisy channel estimate? (default: False)')

    # Algorithm-specific parameters
    parser.add_argument("--exploration_noise", default=0.0, metavar='G', help='Std of Gaussian exploration noise')
    parser.add_argument("--discount", default=0.99, metavar='G', help='Discount factor for reward (default: 0.99)')
    parser.add_argument("--tau", default=0.01, type=float, metavar='G',  help='Learning rate in soft/hard updates of the target networks (default: 0.001)')
    parser.add_argument("--lr", default=1e-3, type=float, metavar='G', help='Learning rate for the networks (default: 0.001)')
    parser.add_argument("--decay", default=1e-5, type=float, metavar='G', help='Decay rate for the networks (default: 0.00001)')

    args = parser.parse_args()

    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    file_name = f"{args.num_antennas}_{args.num_RIS_elements}_{args.num_users}_{args.power_t}_{args.lr}_{args.decay}"

    if not os.path.exists(f"./Learning Curves/{args.experiment_type}"):
        os.makedirs(f"./Learning Curves/{args.experiment_type}")

    if args.save_model and not os.path.exists("./Models"):
        os.makedirs("./Models")

    env = environment.RIS_MISO(args.num_antennas, args.num_RIS_elements, args.num_users, AWGN_var=args.awgn_var)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.state_dim
    action_dim = env.action_dim
    max_action = 1

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "power_t": args.power_t,
        "max_action": max_action,
        "M": args.num_antennas,
        "N": args.num_RIS_elements,
        "K": args.num_users,
        "actor_lr": args.lr,
        "critic_lr": args.lr,
        "actor_decay": args.decay,
        "critic_decay": args.decay,
        "device": device,
        "discount": args.discount,
        "tau": args.tau
    }

    # Initialize the algorithm
    agent = DDPG_Buffer.DDPG(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        agent.load(f"./models/{policy_file}")

    replay_buffer = utils.MultiTaskReplayBuffer(args.num_eps, state_dim, action_dim, max_size=args.buffer_size)

    # Initialize the instant rewards recording array
    instant_rewards = []

    max_reward = 0

    for eps in range(int(args.num_eps)):
        agent = DDPG_Buffer.DDPG(**kwargs)
        state, done = env.reset(), False
        episode_reward = 0
        episode_num = 0
        episode_time_steps = 0

        state = whiten(state)

        eps_rewards = []

        action_selection = 'Random'

        for t in range(int(args.num_time_steps_per_eps)):

            # Choose action from the policy
            if args.start_time_steps > t:                
                action = np.random.uniform(-1,1,size=action_dim)
            else:
                action = agent.select_action(np.array(state))
                action_selection = 'Selection'
            
            # Take the selected action
            next_state, reward, done, _ = env.step(action)

            done = 1.0 if t == args.num_time_steps_per_eps - 1 else float(done)

            # Store data in the experience replay buffer
            replay_buffer.add(eps, state, action, next_state, reward, done)

            state = next_state
            episode_reward += reward

            state = whiten(state)

            if reward > max_reward:
                max_reward = reward

            if replay_buffer.size > args.batch_size:
                # Train the agent
                agent.update_parameters(eps, replay_buffer, args.batch_size)

            print(f"{datetime.now()} | Episode Num: {eps + 1}/{args.num_eps} Time step: {t + 1} Reward: {reward:.3f} Buffer Size: {replay_buffer.size} Action: {action_selection}")

            eps_rewards.append(reward)

            episode_time_steps += 1

            if done:
                print(f"\nTotal T: {t + 1} Episode Num: {episode_num + 1} Episode T: {episode_time_steps} Max. Reward: {max_reward:.3f}\n")

                # Reset the environment
                state, done = env.reset(), False
                episode_reward = 0
                episode_time_steps = 0
                episode_num += 1

                state = whiten(state)

                instant_rewards.append(eps_rewards)

    replay_buffer.save(f"./MultiTask Buffer/replay_buffer_{file_name}_{args.num_eps}_{args.num_time_steps_per_eps}_{args.buffer_size}.pkl")

    if args.save_model:
        agent.save(f"./Models/{file_name}")
