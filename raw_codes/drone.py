from AirSimClient import *
from skimage.transform import resize
import utilities as util
from IPython.core.debugger import Pdb

class environment_2(object):

    z_velocity = 10

    pitch_val = 0
    roll_val = 0

    x_val = 0
    y_val = 0
    z_val = 0

    TIME_MAXIMUM = 50

    collision_count = 0

    box_dist = 9
    dist_to_origin = 15
    dist_threshold = 2 
    boundary_dist = 0
    

    side_boundary_dist = 6 + 20 # the first number is the range of boxes 

    # Initialize a client talking with AirSim 
    def __init__(self, max_steps, box_num, image_size="small", box_hit_reward=1):
        
        self.client = MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)      
        self.max_steps = max_steps
        self.reward = box_hit_reward
        self.image_size = image_size
        self.box_num = box_num

        self.boundary_dist = self.box_num * self.box_dist + self.dist_to_origin + self.dist_threshold

        # Used to detect when the cube is out of view of the camera
        if self.image_size == "small":
            self.mean_blank = np.load("mean_blank_small.npy")
        else:
            self.mean_blank = np.load("mean_blank_large.npy")
            
    def reset(self):
        # self.client.moveToPosition(self.spawnPos.x_val, self.spawnPos.y_val, self.height, 10)
        # self.client.goHome()
        # self.client.moveToPosition(0, 0, self.initial_height, self.z_velocity)
        # time.sle+ep(10)
        self.client.simSetPose(Pose(Vector3r(0, 0, 0), AirSimClientBase.toQuaternion(0, 0, 0)), True)
        self.time_step = 0
        self.collision_count = 0
        state = self.getImage()
        return state

    def step(self, action):
        self.time_step += 1

        # movement controlls 
        pose = []

        cur_position = self.client.simGetPose()
        for key in cur_position:
            for key_2 in cur_position[key]:
                pose.append(cur_position[key][key_2]) 
        
        x_val = pose[0]
        y_val = pose[1]
        z_val = pose[2]

        if action == 0:
            self.pitch_val = 1
            self.roll_val = 0
            self.up_val = 0
        elif action == 1:
            self.pitch_val = 0
            self.roll_val = -1
            self.up_val = 0
        elif action == 2:
            self.pitch_val = 0
            self.roll_val = 1
            self.up_val = 0

        # self.client.moveByAngle(self.pitch_val, self.roll_val, self.initial_height, 0, 0.1)
        
        self.client.simSetPose(Pose(Vector3r(x_val + self.pitch_val, y_val + self.roll_val, z_val + self.up_val), AirSimClientBase.toQuaternion(0, 0, 0)), True)

        state = self.getImage()

        #Pdb().set_trace()

        collision_info = self.client.getCollisionInfo()    

        reward = 0

        if collision_info.has_collided:
            self.collision_count += 1
            reward = 1
            print("Current collected box num: " + str(self.collision_count))
            time.sleep(0.5) #avoid multiple counts for one collision
            
        # Detect if cube is out of view of camera - we don't care if it accidentally collided
        #print(np.linalg.norm(self.mean_blank-state))
        
        # if np.linalg.norm(self.mean_blank-state) < 10:
        if y_val > self.side_boundary_dist or y_val < -self.side_boundary_dist: 
            print("Ep. failed. Cube out of camera view")
            done = 1
        elif self.collision_count == self.box_num:
            print("Ep. success! All boxes are collected")
            print("Took " + str(self.time_step) + " steps")
            done = 1
        elif x_val > self.boundary_dist:
            print("Ep. failed. Out of boundary.")
            print("Box collected: " + str(self.collision_count))
            done = 1
        elif self.time_step == self.max_steps:
            print("Reach max steps")
            done = 1
        else:
            done = 0

        final_collected_box = 0
        if done == 1:
            final_collected_box = self.collision_count

        return state, reward, done, final_collected_box

    def getImage(self):
        #scene vision image in uncompressed RGBA array
        rawData = self.client.simGetImages([ImageRequest(1, AirSimImageType.Scene, False, False)])
        observation = rawData[0] # use the only one element in the list 
        observation = np.fromstring(observation.image_data_uint8, dtype=np.uint8)
        observation = observation.reshape(rawData[0].height, rawData[0].width, 4) #reshape array to 4 channel image array H X W X 
        observation = observation[:,:,:3] # remove alpha channel
        observation = observation[20:120,:,:] # trim vertical
        observation = observation/255 # convert range to [0,1]
        observation = util.convert_numpy_to_grey(observation)
        
        if self.image_size == "small":
            observation = resize(observation, (25,64),  mode="constant") # shrink image!
        elif self.image_size == "large":
            pass

        return observation

###################################################################################################################

class environment(object):
    
    z_velocity = 10

    pitch_val = 0
    roll_val = 0
    initial_height = 0

    x_val = 0
    y_val = 0
    z_val = 0

    TIME_MAXIMUM = 50
    
    # Initialize a client talking with AirSim 
    def __init__(self, max_steps, image_size="small", box_hit_reward=1):
        
        self.client = MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)      
        self.max_steps = max_steps
        self.reward = box_hit_reward
        self.image_size = image_size

        # Used to detect when the cube is out of view of the camera
        if self.image_size == "small":
            self.mean_blank = np.load("mean_blank_small.npy")
        else:
            self.mean_blank = np.load("mean_blank_large.npy")

        print("Drone is ready")

    def reset(self):
        # self.client.moveToPosition(self.spawnPos.x_val, self.spawnPos.y_val, self.height, 10)
        # self.client.goHome()
        # self.client.moveToPosition(0, 0, self.initial_height, self.z_velocity)
        # time.sle+ep(10)
        self.client.simSetPose(Pose(Vector3r(0, 0, 0), AirSimClientBase.toQuaternion(0, 0, 0)), True)
        self.time_step = 0
        state = self.getImage()
        return state

    def step(self, action):
        self.time_step += 1

        pose = []

        cur_position = self.client.simGetPose()
        for key in cur_position:
            for key_2 in cur_position[key]:
                pose.append(cur_position[key][key_2]) 
        
        x_val = pose[0]
        y_val = pose[1]
        z_val = pose[2]

        if action == 0:
            self.pitch_val = 1
            self.roll_val = 0
            self.up_val = 0
        elif action == 1:
            self.pitch_val = 0
            self.roll_val = -1
            self.up_val = 0
        elif action == 2:
            self.pitch_val = 0
            self.roll_val = 1
            self.up_val = 0

        # self.client.moveByAngle(self.pitch_val, self.roll_val, self.initial_height, 0, 0.1)
        
        self.client.simSetPose(Pose(Vector3r(x_val + self.pitch_val, y_val + self.roll_val, z_val + self.up_val), AirSimClientBase.toQuaternion(0, 0, 0)), True)

        state = self.getImage()

        #Pdb().set_trace()

        collision_info = self.client.getCollisionInfo()        

        # Detect if cube is out of view of camera - we don't care if it accidentally collided
        #print(np.linalg.norm(self.mean_blank-state))
        if np.linalg.norm(self.mean_blank-state) < 10:
            print("Ep. failed. Cube out of camera view")
            reward = 0
            done = 1
        elif collision_info.has_collided:
            print("Ep. success!")
            reward = self.reward
            done = 1
        elif self.time_step > self.max_steps:
            print("Ep. failed. Reached max_steps")
            reward = 0
            done = 1
        else:
            reward = 0
            done = 0

        return state, reward, done

    def getImage(self):
        #scene vision image in uncompressed RGBA array
        rawData = self.client.simGetImages([ImageRequest(1, AirSimImageType.Scene, False, False)])
        observation = rawData[0] # use the only one element in the list 
        observation = np.fromstring(observation.image_data_uint8, dtype=np.uint8)
        observation = observation.reshape(rawData[0].height, rawData[0].width, 4) #reshape array to 4 channel image array H X W X 
        observation = observation[:,:,:3] # remove alpha channel
        observation = observation[20:120,:,:] # trim vertical
        observation = observation/255 # convert range to [0,1]
        observation = util.convert_numpy_to_grey(observation)
        
        if self.image_size == "small":
            observation = resize(observation, (25,64),  mode="constant") # shrink image!
        elif self.image_size == "large":
            pass

        return observation