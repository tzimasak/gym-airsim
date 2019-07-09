import airsim
import gym
from gym import spaces
from gym.utils import seeding

import os
import pprint
import time
import numpy as np
import cv2

class AirSimFollowTargetCamOnlyEnv(gym.Env):
    """
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
        success = self.client.simSetSegmentationObjectID("Cube",17)

        self.targetPos = self.client.simGetObjectPose("Cube").position

        # Orientation of bottom_center camera
        self.qCam = None
        # Number of points in a trajectory
        self.points = 3

        # Action space
        self.action_space = spaces.Discrete(5)
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

        # Get an observation of the environment
        observation = self._observe()

        # Calculate reward
        reward, done, dist = self._get_reward()

        # End the episode if the drone is not moving
        velocity = self.client.simGetGroundTruthKinematics().linear_velocity
        if abs(velocity.x_val) < 0.1 and abs(velocity.y_val) < 0.1 and abs(velocity.z_val) < 0.1:
            done = True

        #print(reward, done)
        cv2.imshow('Observation',observation)
        cv2.waitKey(5)

        info = {'distance':dist}
        return observation, reward, done, info

    def _take_action(self, action):
        if action == 0:
            # Do nothing
            pass
        elif action == 1:
            # Tilt drone camera up
            qRot = airsim.to_quaternion(0.0872, 0, 0)
            self.qCam = self.qCam * qRot
        elif action == 2:
            # Tilt drone camera down
            qRot = airsim.to_quaternion(-0.0872, 0, 0)
            self.qCam = self.qCam * qRot
        elif action == 3:
            # Pan drone camera left
            qRot = airsim.to_quaternion(0, -0.0872, 0)
            self.qCam = qRot * self.qCam
        elif action == 4:
            # Pan drone camera right
            qRot = airsim.to_quaternion(0, 0.0872, 0)
            self.qCam = qRot * self.qCam

        self.client.simSetCameraOrientation("bottom_center", self.qCam)

    def _observe(self):
        # Very small pause to let the simulator catch up with the code
        time.sleep(0.009)
        responses = self.client.simGetImages([airsim.ImageRequest("bottom_center", airsim.ImageType.Scene, False, False)])
        response = responses[0]
        # Get numpy array
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        # Reshape array to 4 channel image array H X W X 4
        img_rgba = img1d.reshape(response.height, response.width, 4)
        # Covnert from rgb to bgr for OpenCV
        img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR);
        # Write observation image to file
        #cv2.imwrite("test.png", img_bgr)

        return img_bgr

    def _get_reward(self):
        responses = self.client.simGetImages([airsim.ImageRequest("bottom_center", airsim.ImageType.Segmentation, False, False)])
        response = responses[0]
        # Get numpy array
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        # Reshape array to 4 channel image array H X W X 4
        img_rgba = img1d.reshape(response.height, response.width, 4)
        # Covnert from rgb to bgr for OpenCV
        img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR);

        # Detects positions where the segmentation image has the wanted colour
        # that correspondes to the ID from AirSim/docs/seg_rgbs.txt
        where = np.where(img_bgr[:,:,2] == 196)

        dist = 70
        if len(where[0]) > 0:
            # Find the center of the frame
            frameCenter = np.array((img_bgr[:,:,2].shape[1]//2, img_bgr[:,:,2].shape[0]//2))

            # Find target start and end in both axes
            x_start = min(where[1])
            x_end = max(where[1])

            y_start = min(where[0])
            y_end = max(where[0])

            # Find target center in frame
            targetCenter = np.array(((x_end+x_start)//2, (y_end+y_start)//2))

            # Calculate Euclidean distance
            dist = np.linalg.norm(frameCenter-targetCenter)

            # Calculate reward
            if dist < 25:
                reward = 50
            else:
                reward = -1

            done = False
        else:
            reward = -100

            done = True

        return reward, done, dist

    def _run_trajectory(self):
        path = []
        for i in range(self.points):
            path.append(airsim.Vector3r(np.random.randint(0,16),np.random.randint(-10,11),np.random.randint(-5,-3)))

        self.client.moveOnPathAsync(path, 2, 3e+38, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(), -1, 0)

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Spawn the drone at random position
        self.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(np.random.randint(0,16),np.random.randint(-10,11),np.random.randint(-5,-3)), airsim.to_quaternion(0, 0, 0)), True)
        time.sleep(0.05)

        # Get the direction of the target object
        cameraPos = self.client.simGetCameraInfo("bottom_center").pose.position
        targetDirection = self.targetPos - cameraPos
        #targetDirection.z_val = 0 # for yaw vector

        if targetDirection.get_length() > 0:
            # The default camera pitch
            defaultDirection = airsim.Vector3r(0,0,1)
            # Calculate the angle between the two
            pitchTheta = np.arccos(targetDirection.dot(defaultDirection) / (targetDirection.get_length() * defaultDirection.get_length()))

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

        # Perform trajectory
        self._run_trajectory()
        return self._observe()
