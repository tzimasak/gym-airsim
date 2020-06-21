import airsim
import gym
from gym import spaces
from gym.utils import seeding

import os
import pprint
import time
import numpy as np
import cv2

class AirSimFrontalShotDroneOnlyEnv(gym.Env):
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
        Type: Discrete(9)
        Num	Action
        0   Drone forwards
        1   Drone backwards
        2   Drone left
        3   Drone right
        4   Drone up
        5   Drone down
        6   Drone yaw left
        7   Drone yaw right
        8   Do nothing

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
        success = self.client.simSetSegmentationObjectID("TargetActor",17)

        # Info about the position of the target model
        self.initialPose = self.client.simGetObjectPose("TargetActor")
        self.targetPos = self.client.simGetObjectPose("TargetPos").position
        self.frontalPos = self.client.simGetObjectPose("FrontalPos").position

        # Orientation of bottom_center camera
        self.qCam = None

        # Speed
        self.speed = 1
        self.duration = 0.3

        # Time
        self.episode_duration = 10
        self.start_time = None

        # Action space
        self.action_space = spaces.Discrete(9)
        # Observation space
        responses = self.client.simGetImages([airsim.ImageRequest("bottom_center", airsim.ImageType.Scene, False, False)])
        self.observation_space = spaces.Box(low=0, high=255, shape=(responses[0].height, responses[0].width, 3), dtype=np.uint8)

        self.image_counter = 0
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Take action
        self._take_action(action)
        #time.sleep(0.7)  # ClockSpeed = 1
        time.sleep(0.02) # ClockSpeed = 50

        # Calculate reward
        reward, done, dist, theta = self._get_reward()

        # Get an observation of the environment
        observation = self._observe()

        # Check if episode duration is passed
        if self.episode_duration < time.time() - self.start_time:
            done = True

        #print(reward, done)
        cv2.imshow('Observation',observation)
        cv2.waitKey(5)

        info = {'distance':dist, 'angle':theta}
        return observation, reward, done, info

    def _transform_to_frame(self, vector, q):
        # Create a pure quaternion p out of vector
        vector = vector.to_Quaternionr()
        # q is the vector's orientation with regard to the world frame
        # Pre-multiply vector with q and post-multiply it with the conjugate q*
        qv = q*vector*q.conjugate()
        return airsim.Vector3r(qv.x_val, qv.y_val, qv.z_val)

    def _take_action(self, action):
        # Get the drone's orientation
        ori = self.client.simGetGroundTruthKinematics().orientation

        if action == 0:
            # Drone forwards
            v = self._transform_to_frame(airsim.Vector3r(self.speed, 0, 0), ori)
            z = self.client.simGetGroundTruthKinematics().position.z_val
            self.client.moveByVelocityZAsync(v.x_val, v.y_val, z, self.duration, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode()).join()
            self.client.moveByVelocityZAsync(0, 0, z, self.duration).join()
        elif action == 1:
            # Drone backwards
            v = self._transform_to_frame(airsim.Vector3r(-self.speed, 0, 0), ori)
            z = self.client.simGetGroundTruthKinematics().position.z_val
            self.client.moveByVelocityZAsync(v.x_val, v.y_val, z, self.duration, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode()).join()
            self.client.moveByVelocityZAsync(0, 0, z, self.duration).join()
        elif action == 2:
            # Drone left
            v = self._transform_to_frame(airsim.Vector3r(0, -self.speed, 0), ori)
            z = self.client.simGetGroundTruthKinematics().position.z_val
            self.client.moveByVelocityZAsync(v.x_val, v.y_val, z, self.duration, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode()).join()
            self.client.moveByVelocityZAsync(0, 0, z, self.duration).join()
        elif action == 3:
            # Drone right
            v = self._transform_to_frame(airsim.Vector3r(0, self.speed, 0), ori)
            z = self.client.simGetGroundTruthKinematics().position.z_val
            self.client.moveByVelocityZAsync(v.x_val, v.y_val, z, self.duration, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode()).join()
            self.client.moveByVelocityZAsync(0, 0, z, self.duration).join()
        elif action == 4:
            # Drone up
            self.client.moveByVelocityAsync(0, 0, -self.speed/2, self.duration, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode()).join()
            z = self.client.simGetGroundTruthKinematics().position.z_val
            self.client.moveByVelocityZAsync(0, 0, z, self.duration).join()
        elif action == 5:
            # Drone down
            self.client.moveByVelocityAsync(0, 0, self.speed/2, self.duration, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode()).join()
            z = self.client.simGetGroundTruthKinematics().position.z_val
            self.client.moveByVelocityZAsync(0, 0, z, self.duration).join()
        elif action == 6:
            # Yaw left
            z = self.client.simGetGroundTruthKinematics().position.z_val
            self.client.rotateByYawRateAsync(-35,self.duration).join()
            self.client.moveByVelocityZAsync(0, 0, z, self.duration).join()
        elif action == 7:
            # Yaw right
            z = self.client.simGetGroundTruthKinematics().position.z_val
            self.client.rotateByYawRateAsync(35,self.duration).join()
            self.client.moveByVelocityZAsync(0, 0, z, self.duration).join()
        elif action == 8:
            pass

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
        #cv2.imwrite("./images/test"+str(self.image_counter)+".png", img_bgr)
        self.image_counter = self.image_counter + 1

        return img_bgr

    def _get_reward(self):
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

        #print(dist,theta)
        # Calculate reward
        done = False
        reward = (1 - dist)

        # Detect target reached
        if dist < 0.5 and theta < 0.2:
            reward = reward + 100

        # Detect out of bounds
        if dist > 2.3 or theta > 0.7:
            reward = -100
            done = True

        # Detect collision
        if self.client.simGetCollisionInfo().object_id != -1:
            reward = -100
            done = True

        return reward, done, dist, theta

    def _look_at_target(self):
        # Get the direction of the target object
        cameraPos = self.client.simGetCameraInfo("bottom_center").pose.position
        targetDirection = self.targetPos - cameraPos

        if targetDirection.get_length() > 0:
            targetDirection.z_val = 0 # for yaw vector
            # The default camera yaw
            defaultDirection = airsim.Vector3r(1,0,0)
            # Calculate the angle between the two
            yawTheta = np.arccos(targetDirection.dot(defaultDirection) / (targetDirection.get_length() * defaultDirection.get_length()))
            if targetDirection.y_val < defaultDirection.y_val:
                yawTheta = -yawTheta
        else:
            yawTheta = 0

        self.client.simSetVehiclePose(airsim.Pose(self.client.simGetVehiclePose().position, airsim.to_quaternion(0, 0, yawTheta)), True)

    def _move_target(self, name):
        # Get target's current position
        pose = self.client.simGetObjectPose(name)
        # Change target's position
        #pose.position = pose.position + airsim.Vector3r(np.random.uniform(-0.3,0.3), np.random.uniform(-0.3,0.3), 0) # For testing
        pose.position = pose.position + airsim.Vector3r(np.random.uniform(-4,4), np.random.uniform(-4,4), 0)
        # Change target's orientation
        #pose.orientation = pose.orientation * airsim.to_quaternion(0, 0, np.random.uniform(-0.3,0.3))
        pose.orientation = pose.orientation * airsim.to_quaternion(0, 0, np.random.uniform(-3.14,3.14))
        success = self.client.simSetObjectPose(name, pose, True)

        # Update targetPos variable
        self.targetPos = self.client.simGetObjectPose("TargetPos").position
        self.frontalPos = self.client.simGetObjectPose("FrontalPos").position
        # Fail-safe because AirSim API fails some times and returns NaN pose
        while np.isnan(self.frontalPos.x_val) or np.isnan(self.targetPos.x_val):
            #print("FAIL")
            self.targetPos = self.client.simGetObjectPose("TargetPos").position
            self.frontalPos = self.client.simGetObjectPose("FrontalPos").position

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Move the TargetActor to initial position
        self.client.simSetObjectPose("TargetActor",self.initialPose, True)
        # Change initial position
        self._move_target("TargetActor")

        # Spawn the drone at random position around frontal position
        offset = airsim.Vector3r(np.random.uniform(-2,2), np.random.uniform(0,1), np.random.uniform(-0.5,0.3))
        ori = self.client.simGetObjectPose("TargetActor").orientation
        while np.isnan(ori.w_val):
            #print("FAIL")
            ori = self.client.simGetObjectPose("TargetActor").orientation
        offset = self._transform_to_frame(offset, ori)
        dronePos = self.frontalPos + offset
        self.client.simSetVehiclePose(airsim.Pose(dronePos, airsim.to_quaternion(0, 0, 0)), True)
        self.client.moveByVelocityZAsync(0, 0, dronePos.z_val, self.duration).join()
        time.sleep(0.05)

        # Look at target direction
        self._look_at_target()
        self.client.simSetCameraOrientation("bottom_center", airsim.to_quaternion(1.40, 0, 0))

        # Reset variables
        self.start_time = time.time()
        return self._observe()
