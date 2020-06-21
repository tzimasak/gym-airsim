# gym-airsim
OpenAI Gym-compatible environment of AirSim for multirotor control in RL problems

# gym Environments
**FollowTargetCamOnly**
	Corresponding Unreal Engine Environmet: StaticCube
    Description:
        The drone performs a random trajectory and the agent has to keep
        track of the red cube using only the drone's camera.
    Observation:
        A (256)x(144) color image taken from the bottom_center camera of the
        AirSim drone. The resolution of the observation image is determined
        in the AirSim settings.
    Actions:
        Type: Discrete(5)
        Num Action
        0   Do nothing
        1   Tilt drone camera up
        2   Tilt drone camera down
        3   Pan drone camera left
        4   Pan drone camera right
    Reward:
        Reward depends on the distance of the red cube from the center of the
        camera's frame.
    Starting State:
        Drone starts in a random position in the air with the camera pointed
        at the direction of the red cube.
    Episode Termination:
        Cube moves out of the camera frame.
        Drone completes the trajectory and stops moving.

**FrontalShotDroneOnly**

**FrontalShotCont**

**FrontalShotDroneOnlyMulti**

**FrontalShotContMulti**
