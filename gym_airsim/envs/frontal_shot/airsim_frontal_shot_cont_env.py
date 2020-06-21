import airsim
import gym
from gym import spaces
from gym.utils import seeding

import os
import pprint
import time
import numpy as np
import cv2

class AirSimFrontalShotContEnv(gym.Env):
    """
    Corresponding Unreal Engine Environmet: StaticHuman

    Description:
        The agent has to move the drone in front of the person's face in
        order to get a frontal close-up shot.

    Observation:
        A (200)x(200) color image taken from the bottom_center camera of the
        AirSim drone. The resolution of the observation image is determined
        in the AirSim settings.

    Actions:
        Type: Box(5)
        Num	Action
        0   Drone forwards/backwards
        1   Drone left/right
        2   Drone up/down
        3   Camera tilt up/down
        4   Camera pan left/right

    Reward:
        The reward depends on the drone's distance from the frontal position
        and the angle between the target's direction and the drone's look
        direction.

    Starting State:
        The drone is placed in a random position in front of the target
        human facing towards him.

    Episode Termination:
        The person's face moves out of the camera frame.
        The drone moves far away from the person's face.
        The drone collides with another object.
        10 seconds pass from the beggining of the episode.
    """

    def __init__(self):
        # Connect to the AirSim simulator
        print("Connecting...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Set segmentation IDs for items we don't care
        success = self.client.simSetSegmentationObjectID("Floor",-1);
        success = self.client.simSetSegmentationObjectID("SkySphere",-1);
        # Set segmentation ID for target (ID17 -> RGB[196, 30, 8])
        success = self.client.simSetSegmentationObjectID("RUST_3d_Low1",17)

        self.targetPos = self.client.simGetObjectPose("TargetPos").position
        self.frontalPos = self.client.simGetObjectPose("FrontalPos").position

        # Time
        self.episode_duration = 10
        self.start_time = None

        # Maximum possible speed
        self.maxSpeed = 1
        # Maximum possible camera angle change
        self.maxAngle = 0.1

        # Movement duration
        self.duration = 0.3

        # Orientation of bottom_center camera
        self.qCam = None

        # Action space
        # TODO np.float32?
        self.action_space = spaces.Box(low=np.array([-self.maxSpeed,-self.maxSpeed,-self.maxSpeed,-self.maxAngle,-self.maxAngle]), high=np.array([+self.maxSpeed,+self.maxSpeed,+self.maxSpeed,+self.maxAngle,+self.maxAngle]), dtype=np.float64)
        # Observation space
        responses = self.client.simGetImages([airsim.ImageRequest("bottom_center", airsim.ImageType.Scene, False, False)])
        self.observation_space = spaces.Box(low=0, high=255, shape=(responses[0].height, responses[0].width, 3), dtype=np.uint8)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Take action
        self._take_action(action)
        time.sleep(0.02) # ClockSpeed = 50

        # Calculate reward
        reward, done = self._get_reward(action)

        # Get an observation of the environment
        observation = self._observe()

        # Check if episode duration is passed
        if self.episode_duration < time.time() - self.start_time:
            done = True

        #print(reward, done)
        cv2.imshow('Observation',observation)
        cv2.waitKey(5)
        return observation, reward, done, {}

    def _take_action(self, action):
        # Baselines stuff
        #action = np.clip(action, [-self.maxSpeed,-self.maxSpeed,-self.maxSpeed,-self.maxAngle,-self.maxAngle], [+self.maxSpeed,+self.maxSpeed,+self.maxSpeed,+self.maxAngle,+self.maxAngle])
        # Baselines stuff
        action = np.float64(action)

        #print(action)

        # Drone
        # action[0] -> forwards/backwards
        # action[1] -> left/right
        # action[2] -> up/down
        self.client.moveByVelocityAsync(action[0], action[1], action[2]/2, self.duration, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode()).join()
        z = self.client.simGetGroundTruthKinematics().position.z_val
        self.client.moveByVelocityZAsync(0, 0, z, self.duration).join()

        # Camera
        # action[3] -> up/down
        # action[4] -> left/right
        qRot = airsim.to_quaternion(action[3], 0, 0)
        self.qCam = self.qCam * qRot
        qRot = airsim.to_quaternion(0, action[4], 0)
        self.qCam = qRot * self.qCam
        self.client.simSetCameraOrientation("bottom_center", self.qCam)

    def _observe(self):
        responses = self.client.simGetImages([airsim.ImageRequest("bottom_center", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        # Get numpy array
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        # Reshape array to 4 channel image array H X W X 4
        img_rgba = img1d.reshape(response.height, response.width, 4)
        # Covnert from rgb to bgr for OpenCV
        img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR);
        # Write observation image to file
        #cv2.imwrite("./images/test" + ".png", img_rgba)

        return img_bgr

    def _get_reward(self, action):
        # Drone reward
        cameraPos = self.client.simGetCameraInfo("bottom_center").pose.position
        # Calculate Euclidean distance from target position
        dist = self.frontalPos.distance_to(cameraPos)

        # Camera reward
        # Get target direction
        targetDirection = self.targetPos - cameraPos
        # Get current camera direction vector
        defaultDirection = airsim.Vector3r(1,0,0)
        defaultDirection = defaultDirection.to_Quaternionr()
        ori = self.client.simGetCameraInfo("bottom_center").pose.orientation
        lookDirection = ori.conjugate() * defaultDirection * ori
        lookDirection = airsim.Vector3r(lookDirection.x_val,-lookDirection.y_val,-lookDirection.z_val)
        # Calculate the angle between the two
        theta = np.arccos(targetDirection.dot(lookDirection) / (targetDirection.get_length() * lookDirection.get_length()))

        # Calculate reward
        done = False
        reward = (1 - dist)

        # Detect target reached
        if dist < 0.5 and theta < 0.2:
            reward = reward + 100

        action_clipped = np.clip(action, [-self.maxSpeed,-self.maxSpeed,-self.maxSpeed,-self.maxAngle,-self.maxAngle], [+self.maxSpeed,+self.maxSpeed,+self.maxSpeed,+self.maxAngle,+self.maxAngle])
        if np.array_equal(action, action_clipped) == False:
            reward = -100
            done = True

        # Detect out of bounds
        if dist > 2 or theta > 0.7:
            reward = -100
            done = True

        # Detect collision
        if self.client.simGetCollisionInfo().object_id != -1:
            reward = -100
            done = True


        return reward, done

    def _look_at_target(self):
        # Get the direction of the target object
        cameraPos = self.client.simGetCameraInfo("bottom_center").pose.position
        targetDirection = self.targetPos - cameraPos

        if targetDirection.get_length() > 0:
            # The default camera pitch
            defaultDirection = airsim.Vector3r(0,0,1)
            # Calculate the angle between the two
            pitchTheta = np.arccos(targetDirection.dot(defaultDirection) / (targetDirection.get_length() * defaultDirection.get_length()))

            #targetDirection.z_val = 0 # for yaw vector
            # The default camera yaw
            defaultDirection = airsim.Vector3r(1,0,0)
            # Calculate the angle between the two
            yawTheta = np.arccos(targetDirection.dot(defaultDirection) / (targetDirection.get_length() * defaultDirection.get_length()))
            if targetDirection.y_val < defaultDirection.y_val:
                yawTheta = -yawTheta
        else:
            pitchTheta = 0
            yawTheta = 0

        # Set camera pitch
        self.qCam = airsim.to_quaternion(pitchTheta, 0, 0)
        # Set camera yaw
        self.qCam = airsim.to_quaternion(0, yawTheta, 0) * self.qCam
        self.client.simSetCameraOrientation("bottom_center", self.qCam)

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Spawn the drone at random position around frontal position
        offset = airsim.Vector3r(np.random.uniform(-1,0), np.random.uniform(-2,2), np.random.uniform(-0.5,0.3))
        dronePos = self.frontalPos + offset
        self.client.simSetVehiclePose(airsim.Pose(dronePos, airsim.to_quaternion(0, 0, 0)), True)
        self.client.moveByVelocityZAsync(0, 0, dronePos.z_val, self.duration).join()
        time.sleep(0.05)

        self._look_at_target()

        self.start_time = time.time()
        return self._observe()
