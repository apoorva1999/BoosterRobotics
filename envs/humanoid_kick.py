import torch
from isaacgym.torch_utils import quat_rotate_inverse, torch_rand_float, get_euler_xyz

from .t1 import T1


class HumanoidKick(T1):
    """
    A kicking-oriented variant of T1.
    - Keeps the same robot/controls as T1.
    - Adds a "virtual ball" target (no extra actor) defined in world space.
    - Rewards approaching the ball with a swing foot and imparting forward velocity toward a goal direction.
    """

    def __init__(self, cfg):
        # Run standard T1 setup (sim, robot, buffers, reward prep, etc.)
        super().__init__(cfg)

        # Kick-specific state
        self.swing_foot_index = 1  # use the second foot as the default swing/kick foot
        self.ball_targets = torch.zeros(self.num_envs, 3, device=self.device)
        self.goal_yaws = torch.zeros(self.num_envs, device=self.device)

    # ---------------------------------------------------------------------- #
    # Command sampling                                                       #
    # ---------------------------------------------------------------------- #
    def _resample_commands(self):
        """Sample a ball position (relative to env origin) and a goal yaw."""
        env_ids = (self.episode_length_buf == self.cmd_resample_time).nonzero(as_tuple=False).flatten()
        if len(env_ids) == 0:
            return

        # Sample ball offsets in the world frame (x forward, y lateral)
        ball_x = torch_rand_float(
            self.cfg["commands"]["ball_x"][0], self.cfg["commands"]["ball_x"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        ball_y = torch_rand_float(
            self.cfg["commands"]["ball_y"][0], self.cfg["commands"]["ball_y"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)
        ball_z = torch.zeros_like(ball_x)
        self.ball_targets[env_ids, :] = torch.stack((ball_x, ball_y, ball_z), dim=1) + self.env_origins[env_ids]

        # Sample goal yaw for desired kick direction
        self.goal_yaws[env_ids] = torch_rand_float(
            self.cfg["commands"]["goal_yaw"][0], self.cfg["commands"]["goal_yaw"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)

        # Store into commands buffer for logging/obs consistency
        self.commands[env_ids, 0] = ball_x
        self.commands[env_ids, 1] = ball_y
        self.commands[env_ids, 2] = self.goal_yaws[env_ids]

        # Gait frequency is kept for timing/swing shaping
        self.gait_frequency[env_ids] = torch_rand_float(
            self.cfg["commands"]["gait_frequency"][0], self.cfg["commands"]["gait_frequency"][1], (len(env_ids), 1), device=self.device
        ).squeeze(1)

        still_envs = env_ids[torch.randperm(len(env_ids))[: int(self.cfg["commands"]["still_proportion"] * len(env_ids))]]
        self.commands[still_envs, :] = 0.0
        self.gait_frequency[still_envs] = 0.0

        # Resample interval
        self.cmd_resample_time[env_ids] += torch.randint(
            int(self.cfg["commands"]["resampling_time_s"][0] / self.dt),
            int(self.cfg["commands"]["resampling_time_s"][1] / self.dt),
            (len(env_ids),),
            device=self.device,
        )

    # ---------------------------------------------------------------------- #
    # Observations                                                           #
    # ---------------------------------------------------------------------- #
    def _compute_observations(self):
        """Observation includes ball relative position and goal direction."""
        commands_scale = torch.tensor(
            [1.0, 1.0, 1.0],  # keep raw for logging; scaling already in ranges
            device=self.device,
        )

        # Ball position relative to base frame
        ball_rel_world = self.ball_targets - self.base_pos
        ball_rel_base = quat_rotate_inverse(self.base_quat, ball_rel_world)

        goal_yaw = self.goal_yaws
        goal_dir = torch.stack((torch.cos(goal_yaw), torch.sin(goal_yaw)), dim=1)

        self.obs_buf = torch.cat(
            (
                # Base state
                self.projected_gravity * self.cfg["normalization"]["gravity"],
                self.base_ang_vel * self.cfg["normalization"]["ang_vel"],
                # Ball + goal
                ball_rel_base,
                goal_dir,
                # Phase
                (torch.cos(2 * torch.pi * self.gait_process) * (self.gait_frequency > 1.0e-8).float()).unsqueeze(-1),
                (torch.sin(2 * torch.pi * self.gait_process) * (self.gait_frequency > 1.0e-8).float()).unsqueeze(-1),
                # Joints and actions
                (self.dof_pos - self.default_dof_pos) * self.cfg["normalization"]["dof_pos"],
                self.dof_vel * self.cfg["normalization"]["dof_vel"],
                self.actions,
            ),
            dim=-1,
        )

        self.privileged_obs_buf = torch.cat(
            (
                self.base_lin_vel * self.cfg["normalization"]["lin_vel"],
                (self.base_pos[:, 2] - self.terrain.terrain_heights(self.base_pos)).unsqueeze(-1),
                ball_rel_base,
                goal_dir,
            ),
            dim=-1,
        )
        self.extras["privileged_obs"] = self.privileged_obs_buf

    # ---------------------------------------------------------------------- #
    # Reward terms                                                           #
    # ---------------------------------------------------------------------- #
    def _reward_ball_approach(self):
        """Reward getting the swing foot close to the ball target."""
        foot_pos = self.feet_pos[:, self.swing_foot_index]
        dist = torch.norm(self.ball_targets - foot_pos, dim=-1)
        return torch.exp(-dist / 0.2)

    def _reward_kick_velocity(self):
        """Reward foot velocity toward goal direction when near the ball."""
        foot_vel = (self.feet_pos[:, self.swing_foot_index] - self.last_feet_pos[:, self.swing_foot_index]) / self.dt
        goal_dir = torch.stack((torch.cos(self.goal_yaws), torch.sin(self.goal_yaws), torch.zeros_like(self.goal_yaws)), dim=1)
        forward_speed = torch.sum(foot_vel * goal_dir, dim=-1)

        # Gate by proximity to ball
        foot_pos = self.feet_pos[:, self.swing_foot_index]
        dist = torch.norm(self.ball_targets - foot_pos, dim=-1)
        near_ball = (dist < 0.25).float()
        return torch.relu(forward_speed) * near_ball

    def _reward_goal_align(self):
        """Encourage base yaw alignment with the desired goal direction."""
        _, _, base_yaw = get_euler_xyz(self.base_quat)
        yaw_err = (base_yaw - self.goal_yaws + torch.pi) % (2 * torch.pi) - torch.pi
        return torch.exp(-torch.square(yaw_err) / 0.5)

    def _reward_feet_clearance(self):
        """Encourage swing foot clearance during forward swing."""
        foot_height = self.feet_pos[:, self.swing_foot_index, 2] - self.terrain.terrain_heights(self.feet_pos[:, self.swing_foot_index])
        phase = self.gait_process
        swing = (torch.abs(phase - 0.5) < 0.25).float()
        return torch.relu(foot_height - 0.05) * swing
