from gym.envs.registration import register

register(
    id='DisableAnt-v0',
    entry_point='envs.mujoco.ant:DisableAntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='HeavyAnt-v0',
    entry_point='envs.mujoco.ant:HeavyAntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='LightAnt-v0',
    entry_point='envs.mujoco.ant:LightAntEnv',
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

register(
    id='DisableSwimmer-v0',
    entry_point='envs.mujoco.swimmer:DisableSwimmerEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,
)
register(
    id='LightSwimmer-v0',
    entry_point='envs.mujoco.swimmer:LightSwimmerEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,
)
register(
    id='HeavySwimmer-v0',
    entry_point='envs.mujoco.swimmer:HeavySwimmerEnv',
    max_episode_steps=1000,
    reward_threshold=360.0,
)