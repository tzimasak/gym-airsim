from gym.envs.registration import register

register(
    id='AirSimDiscrt-v0',
    entry_point='gym_airsim.envs:AirSimDiscrtEnv',
)