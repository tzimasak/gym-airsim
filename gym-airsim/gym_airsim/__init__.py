from gym.envs.registration import register

# FOLLOW TARGET
register(
    id='AirSimFollowTargetCamOnlyEnv-v0',
    entry_point='gym_airsim.envs.follow_target:AirSimFollowTargetCamOnlyEnv',
)

# FRONTAL SHOT

# Single Person
register(
    id='AirSimFrontalShotDroneOnlyEnv-v0',
    entry_point='gym_airsim.envs.frontal_shot:AirSimFrontalShotDroneOnlyEnv',
)

register(
    id='AirSimFrontalShotContEnv-v0',
    entry_point='gym_airsim.envs.frontal_shot:AirSimFrontalShotContEnv',
)

register(
    id='AirSimFrontalShotDroneEnv-v0',
    entry_point='gym_airsim.envs.frontal_shot.dbl_agent:AirSimFrontalShotDroneEnv',
)

register(
    id='AirSimFrontalShotCameraEnv-v0',
    entry_point='gym_airsim.envs.frontal_shot.dbl_agent:AirSimFrontalShotCameraEnv',
)

# Multiple People
register(
    id='AirSimFrontalShotDroneOnlyMultiEnv-v0',
    entry_point='gym_airsim.envs.frontal_shot:AirSimFrontalShotDroneOnlyMultiEnv',
)

register(
    id='AirSimFrontalShotContMultiEnv-v0',
    entry_point='gym_airsim.envs.frontal_shot:AirSimFrontalShotContMultiEnv',
)

