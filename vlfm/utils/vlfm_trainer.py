# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

import os
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import torch
import tqdm
from habitat import VectorEnv, logger
from habitat.config import read_write
from habitat.config.default import get_agent_config
from habitat.tasks.rearrange.rearrange_sensors import GfxReplayMeasure
from habitat.tasks.rearrange.utils import write_gfx_replay
from habitat_baselines import PPOTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
)
from habitat_baselines.rl.ddppo.algo import DDPPO  # noqa: F401.
from habitat_baselines.rl.ppo.single_agent_access_mgr import (  # noqa: F401.
    SingleAgentAccessMgr,
)
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    get_action_space_info,
    inference_mode,
    is_continuous_action_space,
)
from habitat_baselines.utils.info_dict import (
    extract_scalars_from_info as extract_scalars_from_info_habitat,
)
from omegaconf import OmegaConf


###TODO Added
from habitat_baselines.rl.ppo.policy import PolicyActionData
###


###TODO Added: Get current habitat position for each env
def get_curr_hab_pos(vec_envs):

    curr_hab_pos = {}
    for i in range(vec_envs.num_envs):

        agent_state = vec_envs.call_at(index = i,
                                       function_name = "get_curr_state")
        curr_hab_pos[i] = agent_state.position
    #print(f" Current Hab position: {agent_state.position}\n Current Hab Rotation: {agent_state.rotation}")
    return curr_hab_pos
###

def extract_scalars_from_info(info: Dict[str, Any]) -> Dict[str, float]:
    info_filtered = {k: v for k, v in info.items() if not isinstance(v, list)}
    return extract_scalars_from_info_habitat(info_filtered)


@baseline_registry.register_trainer(name="vlfm")
class VLFMTrainer(PPOTrainer):
    envs: VectorEnv

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Some configurations require not to load the checkpoint, like when using
        # a hierarchial policy
        if self.config.habitat_baselines.eval.should_load_ckpt:
            # map_location="cpu" is almost always better than mapping to a CUDA device.
            ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
            step_id = ckpt_dict["extra_state"]["step"]
            print(step_id)
        else:
            ckpt_dict = {"config": None}

        config = self._get_resume_state_config_or_new_config(ckpt_dict["config"])

        with read_write(config):
            config.habitat.dataset.split = config.habitat_baselines.eval.split

        if len(self.config.habitat_baselines.eval.video_option) > 0:
            agent_config = get_agent_config(config.habitat.simulator)
            agent_sensors = agent_config.sim_sensors
            extra_sensors = config.habitat_baselines.eval.extra_sim_sensors
            with read_write(agent_sensors):
                agent_sensors.update(extra_sensors)
            with read_write(config):
                if config.habitat.gym.obs_keys is not None:
                    for render_view in extra_sensors.values():
                        if render_view.uuid not in config.habitat.gym.obs_keys:
                            config.habitat.gym.obs_keys.append(render_view.uuid)
                config.habitat.simulator.debug_render = True

        if config.habitat_baselines.verbose:
            logger.info(f"env config: {OmegaConf.to_yaml(config)}")

        ###TODO Added: Enforce extra_args, and initialize Log Dir
        os.environ["ZSOS_LOG_DIR"] = os.path.join(os.getcwd(), config.extra_args.log_dir)
        os.makedirs(os.environ["ZSOS_LOG_DIR"], exist_ok=True)
        print(f"\nSetting Logging Directory to be: {os.environ['ZSOS_LOG_DIR']}\n")

        assert config.extra_args.save_frames_as in ['final', 'all', 'gif'], "Please provide a valid option for save_frames_as arg. Possible Options are: final, all, gif"
        ###

        ###TODO Added
        #Change directory to initialize env using data in habitat-lab
        curr_dir = os.getcwd()
        hab_dir = os.path.join(curr_dir, 'habitat-lab')
        print(f"Chaning directory to : {hab_dir}")
        os.chdir(hab_dir)
        ###

        self._init_envs(config, is_eval=True)

        ###TODO Added
        print(f"Changing back directory to: {curr_dir}")
        os.chdir(curr_dir)
        ###

        self._agent = self._create_agent(None)
        action_shape, discrete_actions = get_action_space_info(self._agent.policy_action_space)

        if self._agent.actor_critic.should_load_agent_state:
            self._agent.load_state_dict(ckpt_dict)

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device="cpu")

        ###TODO Changed: Replacing hidden_state_shape with placeholder shape
        # This is fine since we are not using an RL policy in VLFM
        test_recurrent_hidden_states = torch.zeros(
            (
                self.config.habitat_baselines.num_environments,
                *(1, 512), #*self._agent.hidden_state_shape,
            ),
            device=self.device,
        )
        ###

        prev_actions = torch.zeros(
            self.config.habitat_baselines.num_environments,
            *action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            self.config.habitat_baselines.num_environments,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[Any, Any] = {}  # dict of dicts that stores stats per episode
        ep_eval_count: Dict[Any, int] = defaultdict(lambda: 0)

        rgb_frames: List[List[np.ndarray]] = [[] for _ in range(self.config.habitat_baselines.num_environments)]
        if len(self.config.habitat_baselines.eval.video_option) > 0:
            os.makedirs(self.config.habitat_baselines.video_dir, exist_ok=True)

        number_of_eval_episodes = self.config.habitat_baselines.test_episode_count
        evals_per_ep = self.config.habitat_baselines.eval.evals_per_ep
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            # if total_num_eps is negative, it means the number of evaluation episodes is unknown
            if total_num_eps < number_of_eval_episodes and total_num_eps > 1:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes, dataset only has {{total_num_eps}}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps
            else:
                assert evals_per_ep == 1
        assert number_of_eval_episodes > 0, "You must specify a number of evaluation episodes with test_episode_count"

        pbar = tqdm.tqdm(total=number_of_eval_episodes * evals_per_ep)
        self._agent.eval()

        from vlfm.utils.habitat_visualizer import HabitatVis

        num_successes = 0
        num_total = 0
        hab_vis = HabitatVis()
        while len(stats_episodes) < (number_of_eval_episodes * evals_per_ep) and self.envs.num_envs > 0:
            current_episodes_info = self.envs.current_episodes()

            ###TODO Added: Check if logged file already exists. If so, then skip the episode
            curr_scene_name = os.path.basename(current_episodes_info[0].scene_id).split(".")[0]
            curr_episode_id = current_episodes_info[0].episode_id

            #If Episode was done
            if not not_done_masks[0][0]:

                print(f"\n\nCurrent Scene Name: {curr_scene_name}")
                print(f"Current Episode ID: {curr_episode_id}")

                #If Logged file already exists, then skip the current episode.
                log_file_name = f"{curr_episode_id}_{curr_scene_name}.json"        
                log_file_path = os.path.join(self.config.extra_args.log_dir, log_file_name)

                print(f"\nLog File path: ", log_file_path)
                skip_episode = os.path.exists(log_file_path)

                if skip_episode:
                    print(f"\n\n----Logged File already exists at: {log_file_path}.\nSkipping Episode {curr_episode_id} for Scene {curr_scene_name}...")
            ###


            with inference_mode():

                ###TODO Changed: Accounting for skip_episode. 
                # If false, then implements the usual case of obtaining the action.
                if not skip_episode:
                    action_data = self._agent.actor_critic.act(
                        batch,
                        test_recurrent_hidden_states,
                        prev_actions,
                        not_done_masks,
                        deterministic=False,
                    )
                else:
                    stop_action = torch.tensor([[0]], dtype=torch.long)
                    
                    action_data = PolicyActionData(
                                    actions=stop_action,
                                    rnn_hidden_states=test_recurrent_hidden_states,
                                    # policy_info=[{"step_action": {}}],
                                )


                if "VLFM_RECORD_ACTIONS_DIR" in os.environ:
                    action_id = action_data.actions.cpu()[0].item()
                    filepath = os.path.join(
                        os.environ["VLFM_RECORD_ACTIONS_DIR"],
                        "actions.txt",
                    )
                    # If the file doesn't exist, create it
                    if not os.path.exists(filepath):
                        open(filepath, "w").close()
                    with open(filepath, "a") as f:
                        f.write(f"{action_id}\n")

                if action_data.should_inserts is None:
                    test_recurrent_hidden_states = action_data.rnn_hidden_states
                    prev_actions.copy_(action_data.actions)  # type: ignore
                else:
                    for i, should_insert in enumerate(action_data.should_inserts):
                        if should_insert.item():
                            test_recurrent_hidden_states[i] = action_data.rnn_hidden_states[i]
                            prev_actions[i].copy_(action_data.actions[i])  # type: ignore
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            if is_continuous_action_space(self._env_spec.action_space):
                # Clipping actions to the specified limits
                step_data = [
                    np.clip(
                        a.numpy(),
                        self._env_spec.action_space.low,
                        self._env_spec.action_space.high,
                    )
                    for a in action_data.env_actions.cpu()
                ]
            else:
                step_data = [a.item() for a in action_data.env_actions.cpu()]

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [list(x) for x in zip(*outputs)]
            policy_infos = self._agent.actor_critic.get_extra(action_data, infos, dones)
            for i in range(len(policy_infos)):
                infos[i].update(policy_infos[i])


            ###TODO ADDED: Save the agent position to txt file
            if (self._agent.actor_critic._num_steps > 0) and \
                (self._agent.actor_critic._num_steps < self.config.habitat.environment.max_episode_steps):
                
                save_traj_path = os.path.join(self.config.extra_args.log_dir, "trajectory", f"{curr_episode_id}_{curr_scene_name}.txt")
                os.makedirs(os.path.dirname(save_traj_path), exist_ok = True)

                curr_hab_pos = get_curr_hab_pos(self.envs)[0]
                num_steps = self._agent.actor_critic._num_steps
                
                with open(save_traj_path, "a" if num_steps > 1 else "w") as f:
                    f.write(f"{self._agent.actor_critic._num_steps}, {curr_hab_pos[0]}, {curr_hab_pos[1]}, {curr_hab_pos[2]}\n")
                    # print(f"Saved Trajectory at step {num_steps} to {save_traj_path}")
            ###

            batch = batch_obs(  # type: ignore
                observations,
                device=self.device,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(rewards_l, dtype=torch.float, device="cpu").unsqueeze(1)
            current_episode_reward += rewards
            next_episodes_info = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    ep_eval_count[
                        (
                            next_episodes_info[i].scene_id,
                            next_episodes_info[i].episode_id,
                        )
                    ]
                    == evals_per_ep
                ):
                    envs_to_pause.append(i)
                elif int(next_episodes_info[i].episode_id) == 123123123:
                    envs_to_pause.append(i)

                ###TODO Changed: Adding the skip_episode check
                if (not skip_episode) and (len(self.config.habitat_baselines.eval.video_option) > 0):
                    hab_vis.collect_data(batch, infos, action_data.policy_info)
                ###

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {"reward": current_episode_reward[i].item()}
                    episode_stats.update(extract_scalars_from_info(infos[i]))
                    current_episode_reward[i] = 0
                    k = (
                        current_episodes_info[i].scene_id,
                        current_episodes_info[i].episode_id,
                    )
                    ep_eval_count[k] += 1
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[(k, ep_eval_count[k])] = episode_stats

                    if episode_stats["success"] == 1:
                        num_successes += 1
                    num_total += 1
                    print(f"Success rate: {num_successes / num_total * 100:.2f}% ({num_successes} out of {num_total})")


                    ###TODO Added: skip_episode check
                    if (not skip_episode):
                    ###

                        from vlfm.utils.episode_stats_logger import (
                            log_episode_stats,
                        )

                        ###TODO Added: Adding relevant info in infos
                        infos[i]["num_steps"] = self._agent.actor_critic._num_steps
                        infos[i]["final_pos"] = [float(coord) for coord in get_curr_hab_pos(self.envs)[i]]
                        ###

                        ###TODO Changed: Adding a try-except clause inside this, instead of out in vlfm_trainer
                        # This ensures that the episode is logged even when the failure is unknown
                        # try:
                        #     failure_cause = log_episode_stats(
                        #         current_episodes_info[i].episode_id,
                        #         current_episodes_info[i].scene_id,
                        #         infos[i],
                        #     )
                        # except Exception:
                        #     failure_cause = "Unknown"

                        failure_cause = log_episode_stats(
                            current_episodes_info[i].episode_id,
                            current_episodes_info[i].scene_id,
                            infos[i],
                        )

                        ###

                        if len(self.config.habitat_baselines.eval.video_option) > 0:
                            rgb_frames[i] = hab_vis.flush_frames(failure_cause)

                            ###TODO Changed: Instead of a video, save the last frame or all the frames
                            # generate_video(
                            #     video_option=self.config.habitat_baselines.eval.video_option,
                            #     video_dir=self.config.habitat_baselines.video_dir,
                            #     images=rgb_frames[i],
                            #     episode_id=current_episodes_info[i].episode_id,
                            #     checkpoint_idx=checkpoint_index,
                            #     metrics=extract_scalars_from_info(infos[i]),
                            #     fps=self.config.habitat_baselines.video_fps,
                            #     tb_writer=writer,
                            #     keys_to_include_in_name=self.config.habitat_baselines.eval_keys_to_include_in_name,
                            # )

                            from PIL import Image
                            images = np.array(rgb_frames[i])                        
                            curr_scene_name = os.path.basename(current_episodes_info[i].scene_id).split(".")[0]

                            frames_root_dir = os.path.join(config.extra_args.log_dir, 'gifs')
                            if not os.path.exists(frames_root_dir): os.makedirs(frames_root_dir, exist_ok=True)
                            
                            
                            #Save only Final Frame
                            if config.extra_args.save_frames_as == "final":
                                final_file_path = os.path.join(frames_root_dir, f'scene_{curr_scene_name}_ep_{current_episodes_info[i].episode_id}.png')
                                final_frame = Image.fromarray(images[-1])
                                final_frame.save(final_file_path, format="PNG")
                                print(f"Created Last Frame Image at: {final_file_path}")

                            #Save all the Frames
                            elif config.extra_args.save_frames_as == "all":

                                save_all_dir = os.path.join(frames_root_dir, f"{curr_scene_name}/{current_episodes_info[i].episode_id}")
                                os.makedirs(save_all_dir, exist_ok=True)

                                print(f"Saving all the frames at : {save_all_dir}")
                                for i in tqdm.tqdm(range(len(images))):

                                    pad_i = str(i).zfill(3)
                                    frame_save_path = os.path.join(save_all_dir, f'vis_{pad_i}.png')

                                    curr_frame = Image.fromarray(images[i])
                                    curr_frame.save(frame_save_path, format="PNG")

                            #Save as GIF
                            elif config.extra_args.save_frames_as == "gif":
                                gif_file_path = os.path.join(frames_root_dir, f'scene_{curr_scene_name}_ep_{current_episodes_info[i].episode_id}.gif')
                                gif_images = list(map(Image.fromarray, images))
                                gif_images[0].save(gif_file_path, save_all=True, append_images=gif_images[1:], duration=200, loop=0)
                                print(f'Created gif at : {gif_file_path}')
                            ###

                            rgb_frames[i] = []

                    gfx_str = infos[i].get(GfxReplayMeasure.cls_uuid, "")
                    if gfx_str != "":
                        write_gfx_replay(
                            gfx_str,
                            self.config.habitat.task,
                            current_episodes_info[i].episode_id,
                        )

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        pbar.close()

        if "ZSOS_DONE_PATH" in os.environ:
            # Create an empty file at ZSOS_DONE_PATH to signal that the
            # evaluation is done
            done_path = os.environ["ZSOS_DONE_PATH"]
            with open(done_path, "w") as f:
                f.write("")

        assert (
            len(ep_eval_count) >= number_of_eval_episodes
        ), f"Expected {number_of_eval_episodes} episodes, got {len(ep_eval_count)}."

        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = np.mean([v[stat_key] for v in stats_episodes.values()])

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalar("eval_reward/average_reward", aggregated_stats["reward"], step_id)

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        self.envs.close()
