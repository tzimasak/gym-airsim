import airsim
import gym
from gym import spaces
from gym.utils import seeding

import os
import pprint
import time
import numpy as np
import cv2

class AirSimFrontalShotContMultiEnv(gym.Env):
    """
    Corresponding Unreal Engine Environmet: MultiHumanCity

    Description:
        The agent has to move the drone in front of the person's face in
        order to get a frontal close-up shot. In this environment there
        are 9 different people.

    Observation:
        A (200)x(200) color image taken from the front_center camera of the
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
        The drone is placed in a random position in front of a random target
        human facing towards him.

    Episode Termination:
        The person's face moves out of the camera frame.
        The drone moves far away from the person's face.
        The drone collides with another object.
        20 seconds pass from the beggining of the episode.
    """

    def __init__(self):
        # Connect to the AirSim simulator
        print("Connecting...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Number of people meshes in the scene
        self.numPeople = 9

        self.initialPose = []
        # Info about the initial pose of all target models
        for i in range(self.numPeople):
            pose = self._safe_simGetObjectPose("TargetActor"+str(i))
            self.initialPose.append(pose)

        # Currently active target person mesh
        self.activeMesh = -1

        # The position of the face of current target model
        self.targetPos = None

        # The target position in which the camera should be placed for a frontal shot
        self.frontalPos = None

        # Orientation of front_center camera
        self.qCam = None

        # Speed
        self.duration = 0.3

        # Maximum possible speed
        self.maxSpeed = 0.5
        # Maximum possible camera angle change
        self.maxAngle = 0.1

        # Time
        self.episode_duration = 20
        self.start_time = None

        # Times the target is reached
        self.targetReached = 0

        # Action space
        self.action_space = spaces.Box(low=np.array([-self.maxSpeed,-self.maxSpeed,-self.maxSpeed,-self.maxAngle,-self.maxAngle]), high=np.array([+self.maxSpeed,+self.maxSpeed,+self.maxSpeed,+self.maxAngle,+self.maxAngle]), dtype=np.float64)
        # Observation space
        responses = self.client.simGetImages([airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)])
        self.observation_space = spaces.Box(low=0, high=255, shape=(responses[0].height, responses[0].width, 3), dtype=np.uint8)

        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Take action
        action = np.float64(action)
        self._take_action(action)
        #time.sleep(0.07)  # ClockSpeed = 1
        time.sleep(0.05) # ClockSpeed = 50

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

    def _transform_to_frame(self, vector, q):
        # Create a pure quaternion p out of vector
        vector = vector.to_Quaternionr()
        # q is the vector's orientation with regard to the world frame
        # Pre-multiply vector with q and post-multiply it with the conjugate q*
        qv = q*vector*q.conjugate()
        return airsim.Vector3r(qv.x_val, qv.y_val, qv.z_val)

    def _take_action(self, action):
        # Baselines stuff
        action = np.clip(action, [-self.maxSpeed,-self.maxSpeed,-self.maxSpeed,-self.maxAngle,-self.maxAngle], [+self.maxSpeed,+self.maxSpeed,+self.maxSpeed,+self.maxAngle,+self.maxAngle])

        # Drone
        # action[0] -> forwards/backwards
        # action[1] -> left/right
        # action[2] -> up/down
        v = airsim.Vector3r(action[0], action[1], action[2])
        # Get the drone's orientation
        #ori = self.client.simGetGroundTruthKinematics().orientation
        ori = self.client.simGetCameraInfo("front_center").pose.orientation
        v = self._transform_to_frame(v, ori)
        self.client.moveByVelocityAsync(v.x_val, v.y_val, v.z_val-0.09, self.duration, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode()).join()
        z = self.client.simGetGroundTruthKinematics().position.z_val
        self.client.moveByVelocityZAsync(0, 0, z, self.duration).join()

        # Camera
        # action[3] -> up/down
        # action[4] -> left/right
        qRot = airsim.to_quaternion(action[3], 0, 0)
        self.qCam = self.qCam * qRot
        qRot = airsim.to_quaternion(0, 0, action[4])
        self.qCam = qRot * self.qCam
        self.client.simSetCameraOrientation("front_center", self.qCam)


    def _observe(self):
        responses = self.client.simGetImages([airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)])
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
        cameraPos = self.client.simGetCameraInfo("front_center").pose.position
        # Calculate Euclidean distance from target position
        dist = self.frontalPos.distance_to(cameraPos)

        # Camera reward
        # Get target direction
        targetDirection = self.targetPos - cameraPos
        # Get current camera direction vector
        defaultDirection = airsim.Vector3r(1,0,0)
        ori = self.client.simGetCameraInfo("front_center").pose.orientation
        lookDirection = self._transform_to_frame(defaultDirection, ori)
        # Calculate the angle between the two
        theta = np.arccos(targetDirection.dot(lookDirection) / (targetDirection.get_length() * lookDirection.get_length()))

        #print(dist,theta)

        # Calculate reward
        done = False
        reward = (1 - dist)

        action_clipped = np.clip(action, [-self.maxSpeed,-self.maxSpeed,-self.maxSpeed,-self.maxAngle,-self.maxAngle], [+self.maxSpeed,+self.maxSpeed,+self.maxSpeed,+self.maxAngle,+self.maxAngle])
        if np.array_equal(action, action_clipped) == False:
            reward = reward - 10

        if dist < 0.15 and theta < 0.1 and all(i < 0.1 for i in action_clipped):
            reward = reward + 100
        elif dist < 0.15 and theta < 0.1:
            reward = reward + 10

        # Detect out of bounds
        if dist > 1.13 or theta > 0.4:
            reward = -100
            done = True

        # Detect collision
        #if self.client.simGetCollisionInfo().object_id != -1:
        if self.client.simGetCollisionInfo().has_collided != False:
            reward = -100
            done = True

        return reward, done

    def _look_at_target(self):
        # Get the direction of the target object
        cameraPos = self.client.simGetGroundTruthKinematics().position
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

        self.qCam = airsim.to_quaternion(0, 0, 0)
        self.client.simSetCameraOrientation("front_center", self.qCam)

    def _move_target(self, name):
        # Get target's current position
        pose = self._safe_simGetObjectPose(name)

        # Change target's orientation
        #pose.orientation = pose.orientation * airsim.to_quaternion(0, 0, np.random.uniform(-0.3,0.3))
        pose.orientation = pose.orientation * airsim.to_quaternion(0, 0, np.random.uniform(-3.14,3.14))
        success = self.client.simSetObjectPose(name, pose, True)

        # Update targetPos variable
        self.targetPos = self._safe_simGetObjectPose("TargetPos"+str(self.activeMesh)).position
        # Update frontalPos variable
        offset = airsim.Vector3r(0, 0.5, -0.06)
        ori = self._safe_simGetObjectPose("TargetActor"+str(self.activeMesh)).orientation
        offset = self._transform_to_frame(offset, ori)
        self.frontalPos = self.targetPos + offset

    def reset(self):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # Move the currently active TargetActor to its initial pose
        self.client.simSetObjectPose("TargetActor"+str(self.activeMesh),self.initialPose[self.activeMesh], True)

        # Pick a new TargetActor for this episode
        self.activeMesh = np.random.randint(0,self.numPeople-3) # Train
        #self.activeMesh = np.random.randint(6,self.numPeople) # Test

        # Change TargetActor pose
        self._move_target("TargetActor"+str(self.activeMesh))

        # Spawn the drone at random position around frontal position
        offset = airsim.Vector3r(np.random.uniform(-1,1), np.random.uniform(0,0.5), np.random.uniform(-0.1,0.1))
        #offset = airsim.Vector3r(0, 0, 0)
        ori = self._safe_simGetObjectPose("TargetActor"+str(self.activeMesh)).orientation
        offset = self._transform_to_frame(offset, ori)
        dronePos = self.frontalPos + offset
        self.client.simSetVehiclePose(airsim.Pose(dronePos, airsim.to_quaternion(0, 0, 0)), True)
        self.client.moveByVelocityZAsync(0, 0, dronePos.z_val, self.duration).join()
        time.sleep(0.05)

        # Look at target direction
        self._look_at_target()

        # Reset variables
        self.targetReached = 0
        self.start_time = time.time()
        return self._observe()

    def _safe_simGetObjectPose(self, name):
        # Fail-safe function because AirSim API fails some times and returns NaN pose
        pose = self.client.simGetObjectPose(name)
        while np.isnan(pose.position.x_val):
            #print("FAIL")
            pose = self.client.simGetObjectPose(name)
        return pose
