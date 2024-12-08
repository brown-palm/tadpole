import warnings
warnings.filterwarnings('ignore')
import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import torch
import numpy as np
import gym
gym.logger.set_level(40)
import time
import random
from pathlib import Path
from cfg import parse_cfg
from env import make_env
from algorithm.tdmpc import TDMPC
from algorithm.helper import StridedOverlappingEpisode, ReplayBuffer
import logger
torch.backends.cudnn.benchmark = True
__CONFIG__, __LOGS__ = 'cfgs', 'logs'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def evaluate(env, agent, num_episodes, step, env_step, video):
    """Evaluate a trained agent and optionally save a video."""
    episode_rewards = []
    for i in range(num_episodes):
        obs, done, ep_reward, t = env.reset(), False, 0, 0
        if video: video.init(env, enabled=(i==0))
        while not done:
            action = agent.plan(obs, eval_mode=True, step=step, t0=t==0)
            obs, reward, done, _ = env.step(action.cpu().numpy())
            ep_reward += reward
            if video: video.record(env)
            t += 1
        episode_rewards.append(ep_reward)
        if video: video.save(env_step)
    return np.nanmean(episode_rewards)


def train(cfg):
    """Training script for (Video-)TADPoLe TD-MPC. Requires a CUDA-enabled device."""
    assert torch.cuda.is_available()
    set_seed(cfg.seed)
    work_dir = Path().cwd() / __LOGS__ / cfg.task / cfg.modality / cfg.exp_name / str(cfg.seed)
    env, agent, buffer = make_env(cfg), TDMPC(cfg), ReplayBuffer(cfg)
    device = torch.device('cuda')

    domain, task = cfg.task.replace('-', '_').split('_', 1)

    camera_id = dict(quadruped=2).get(domain, 0)
    dim = dict(dog=512).get(domain, 480)
    render_kwargs = dict(height=dim, width=dim, camera_id=camera_id)

    def render_env():
        rendered = torch.Tensor(env.render(**render_kwargs).transpose(2,0,1).copy())[None,:].to(device)     
        if cfg.fp16:
            rendered = rendered.half()
        return rendered

    noise_level_base = cfg.noise_level_base
    noise_level_range = cfg.noise_level_range
    align_scale = cfg.align_scale
    recon_scale = cfg.recon_scale

    context_size = cfg.context_size
    stride = cfg.stride

    if cfg.tadpole_type == 'tadpole':
        from utils import TADPoLe
        guidance = TADPoLe(device, cfg.text_prompt, fp16=cfg.fp16)
        assert context_size == stride == 1
    else:
        assert cfg.tadpole_type == 'video-tadpole'
        from utils import VideoTADPoLe
        guidance = VideoTADPoLe(device, cfg.text_prompt, fp16=cfg.fp16)

    # Run training
    L = logger.Logger(work_dir, cfg)
    episode_idx, start_time = 0, time.time()
    for step in range(0, cfg.train_steps+cfg.episode_length, cfg.episode_length):
        # Collect trajectory
        obs = env.reset()
        episode = StridedOverlappingEpisode(cfg, obs, context_size, stride=stride)

        latent = guidance.encode_imgs(render_env())
        latent_history = latent.repeat(context_size, 1, 1, 1)
        timestep = torch.randint(noise_level_base, noise_level_base + noise_level_range, [1], dtype=torch.long, device=device)
        while not episode.done:
            action = agent.plan(obs, step=step, t0=episode.first)
            obs, gt_reward, done, _ = env.step(action.cpu().numpy())
            latent_history = latent_history.roll(-1, 0)
            latent_history[-1] = guidance.encode_imgs(render_env())

            reward = torch.zeros([context_size], dtype=torch.float32).to(device)
            if ((episode._idx + 2 - context_size) % stride == 0 or episode._idx == cfg.episode_length - 1) and episode._idx >= (context_size-2):
                reward += guidance.get_reward(latent_history,
                                              timestep,
                                              align_scale=align_scale,
                                              recon_scale=recon_scale)

            episode += (obs, action, reward, gt_reward, done)
        assert len(episode) == cfg.episode_length
        buffer += episode

        # Update model
        train_metrics = {}
        if step >= cfg.seed_steps:
            num_updates = cfg.seed_steps if step == cfg.seed_steps else cfg.episode_length
            for i in range(num_updates):
                train_metrics.update(agent.update(buffer, step+i))

        # Log training episode
        episode_idx += 1
        env_step = int(step*cfg.action_repeat)
        common_metrics = {
            'episode': episode_idx,
            'step': step,
            'env_step': env_step,
            'total_time': time.time() - start_time,
            'episode_reward': episode.cumulative_reward,
            'episode_gt_reward': episode.cumulative_gt_reward}
        train_metrics.update(common_metrics)
        L.log(train_metrics, category='train', agent=agent)

        # Evaluate agent periodically
        if env_step % cfg.eval_freq == 0:
            eval_episode_gt_reward = evaluate(env, agent, cfg.eval_episodes, step, env_step, L.video)
            eval_metrics = {
                'env_step': env_step,
                'episode_gt_reward': eval_episode_gt_reward,
            }
            L.log(eval_metrics, category='eval', agent=agent)

    L.finish(agent)
    print('Training completed successfully')


if __name__ == '__main__':
    train(parse_cfg(Path().cwd() / __CONFIG__))
