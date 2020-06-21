import airsim   
import gym
from gym import spaces
from gym.utils import seeding

import os
import pprint
import time
import numpy as np
import cv2

class AirSimFrontalShotCameraEnv(gym.Env):
    """
    Description:
        OpenAI Gym-compatible environment of AirSim for multirotor control in RL problems
        
    Observation:
        TODO
    
    Actions:
        Type: Discrete(4)
        Num	Action
        0   Move straight
        1	Yaw to the left
        2	Yaw to the right  
        3
        4
        
    Reward:
        TODO
        
    Starting State:
        TODO
        
    Episode Termination:    
        TODO    
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
        
        self.targetPos = self.client.simGetObjectPose("Target").position        
        
        # Orientation of bottom_center camera
        self.qCam = None       
        
        # Speed
        self.velocity = None
        self.speed = 0.4
        self.duration = 0.3#3e+38
              
        # Time
        self.episode_duration = 10
        self.start_time = None
        
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
        reward, done = self._get_reward()         
            
        #print(reward, done)
        """cv2.imshow('Observation',observation)
        cv2.waitKey(0)"""      
        return observation, reward, done, {}
        
    def _take_action(self, action):  
        #TO_DO ZOOM        
        """camera_info = self.client.simGetCameraInfo("bottom_center")
        print("CameraInfo %s: %s" % ("bottom_center", pprint.pformat(camera_info)))  
        drone_info = self.client.simGetGroundTruthKinematics()
        print("DroneInfo %s: %s" % ("drone", pprint.pformat(drone_info)))"""
    
        if action == 0:
            pass
        elif action == 1:
            # Camera up
            qRot = airsim.to_quaternion(0.0872, 0, 0)  
            self.qCam = self.qCam * qRot
        elif action == 2:
            # Camera down
            qRot = airsim.to_quaternion(-0.0872, 0, 0)  
            self.qCam = self.qCam * qRot
        elif action == 3:
            # Camera left
            qRot = airsim.to_quaternion(0, -0.0872, 0)  
            self.qCam = qRot * self.qCam
        elif action == 4:
            # Camera right
            qRot = airsim.to_quaternion(0, 0.0872, 0)  
            self.qCam = qRot * self.qCam
            
        self.client.simSetCameraOrientation("bottom_center", self.qCam) 
            
        
    def _observe(self):
        # Very small pause to let the simulator catch up with the code
        time.sleep(0.009)
        #self.client.simPause(True)
        #self.client.simPause(False)
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
        
    def _get_reward(self): 
        done = False         
       
        cameraPos = self.client.simGetCameraInfo("bottom_center").pose.position 
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
        
        # Calculate final reward
        reward = (0.4 - theta)
                   
        # Detect target reached    
        if self.episode_duration < (time.time()-self.start_time) and theta < 0.2:
            reward = 100
            #done = True
        
        # Detect out of bounds
        if theta > 0.7:
            reward = -100
            done = True
        
        # Detect collision
        if self.client.simGetCollisionInfo().object_id != -1:        
            #reward = -100
            done = True
        
        return reward, done
    
    
    def _move_to_target(self):
        frontalPos = self.targetPos - airsim.Vector3r(1,0,0.4)
        self.client.moveToPositionAsync(frontalPos.x_val, frontalPos.y_val, frontalPos.z_val, self.speed, 3e+38, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode(), -1, 0) 
        
    def render(self):        
        """self.client.simPause(True)
        time.sleep(1)
        self.client.simContinueForTime(1)"""        
        #self.client.simSetObjectPose("Cube",airsim.Pose(airsim.Vector3r(np.random.randint(0,2),np.random.randint(-6,6),0), airsim.to_quaternion(0, 0, 0)), True) 
    
    def reset(self):
        self.client.reset()   
        self.client.enableApiControl(True)
        self.client.armDisarm(True) 
        
        # Spawn the drone at random position        
        self.velocity = airsim.Vector3r(0,0,0)
        self.client.simSetVehiclePose(airsim.Pose(airsim.Vector3r(np.random.uniform(-1,0),np.random.uniform(-1,1),np.random.uniform(-3,-1)), airsim.to_quaternion(0, 0, 0)), True) 
        self.client.moveByVelocityAsync(self.velocity.x_val, self.velocity.y_val, self.velocity.z_val, 1, airsim.DrivetrainType.MaxDegreeOfFreedom, airsim.YawMode())
        time.sleep(0.05)
        
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
        
        self.start_time = time.time()
        
        #self._move_to_target()        
        return self._observe()
