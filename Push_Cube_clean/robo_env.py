"""
This python file does:


define a pybullet environment that spawns a red cube which then has to be pushed in the correct translucent goal area


Segmentation: "generate_color_map", "hsv_to_rgb", "segmentation_to_rgb" help with image segmentation

PandaSim:
    - reset(self) - The cubes get randomly positioned within self.x_min, self.x_max and within self.y_min, self.y_max & Robot is reset
    - closeEnv - closes environment
    Vis- create_spheres - creates the visualization sphere
    Vis- updatePathSpehere - repositions the visualization spheres to the next given 8 positions
    - createCube - creates a cube object   
    - get_object_position - gets the position and orientation of an object id
    - get_cube_pos - gets the position and orientation of the red cube
    - is_close_to_Goal - checks if a given object is within the given radius around a position
    - isDone - checks if the goal is reached: eg. the cube is close to the goal
    - load_robot - loads the robot
    - set_joint_angles - sets joint angles of each robot joint
    - get_joint_angles - gets joint angles of each robot joint
    - check_collisions - checks on collision points. this is mainly unused
    - calculate_ik - calculates inverse kinematics for the panda URDF
    - add_gui_sliders - adds the moving sliders for the user to controll the robot
    - read_gui_sliders - reads the values of these sliders
    - get_current_pose - gets the current EE-position and orientation
    - get_camera_image - gets the camera image either from a static or a EE-perspective, segmented or not
    - get_camera_orientation - gets the orientation of the camera
    - create_drag_handle - there was an idea to use a drag handle on the robot to control it's EE, but this did not work
    - register_mouse_events - removes the gui sliders and only leaves the gripper slider
    - process_events - react to mouse moves and keyboard inouts to controll the robot
    - handle_keyboard - handles keyboad inputs
    - handle_mouse_click - handles mouse click inputs
    - handle_mouse_move - handles mouse move inputs to controll the robot
    - screen_to_world - transforms the position of the mouse pointer to world coordinates to controll the robot's EE

"""
# IMPORTS

import numpy as np
import pybullet as p
import math
import os
from datetime import datetime
import pybullet_data
from collections import namedtuple
from addict import Dict
import random
import time

## used parts from https://github.com/josepdaniel/ur5-bullet/tree/master

# Set the correct path to your URDF file UR5 has to be added to folder: pybullet_data.getDataPath()
ROBOT_URDF_PATH = os.path.join(os.getcwd(), "franka_panda/panda.urdf")
TABLE_URDF_PATH = os.path.join(pybullet_data.getDataPath(), "table/table.urdf")
print(ROBOT_URDF_PATH)



def generate_color_map(max_id=1000):
    """
    Generate a deterministic color map for object IDs.
    
    Args:
    max_id (int): The maximum object ID to generate colors for.

    Returns:
    dict: A dictionary mapping object IDs to RGB colors.
    """
    color_map = {0: [0, 0, 0]}  # Black for background
    color_map[-1] = [128, 128, 128]  # Gray for undefined areas
    for i in range(1, max_id + 1):
        hue = (i * 0.618033988749895) % 1  # Golden ratio method
        rgb = np.array(hsv_to_rgb(hue, 0.9, 0.9)) * 255
        color_map[i] = rgb.astype(int).tolist()
    return color_map

def hsv_to_rgb(h, s, v):
    """
    Convert HSV color to RGB.
    
    Args:
    h, s, v (float): Hue, Saturation, Value components.

    Returns:
    tuple: RGB values (0-1 range).
    """
    if s == 0.0:
        return (v, v, v)
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    if i == 0:
        return (v, t, p)
    if i == 1:
        return (q, v, p)
    if i == 2:
        return (p, v, t)
    if i == 3:
        return (p, q, v)
    if i == 4:
        return (t, p, v)
    if i == 5:
        return (v, p, q)

# Generate a color map once
GLOBAL_COLOR_MAP = generate_color_map()

def segmentation_to_rgb(segmentation_mask):
    """
    Convert a 2D segmentation mask to a 3-channel RGB image with consistent colors.
    
    Args:
    segmentation_mask (np.array): 2D array of shape (height, width) containing object IDs

    Returns:
    np.array: 3D array of shape (height, width, 3) containing the colored segmentation mask
    """
    height, width = segmentation_mask.shape
    rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Get unique object IDs
    unique_ids = np.unique(segmentation_mask)
    
    # Apply colors to the mask
    for id in unique_ids:
        if id in GLOBAL_COLOR_MAP:
            rgb_mask[segmentation_mask == id] = GLOBAL_COLOR_MAP[id]
        else:
            # If we encounter an ID not in our color map, assign a new color
            new_color = generate_color_map(max(abs(id), 1000))[abs(id)]
            GLOBAL_COLOR_MAP[id] = new_color
            rgb_mask[segmentation_mask == id] = new_color
    
    return rgb_mask

class PushCubeEnv():
  
    def __init__(self, width = 640, height = 480, fov = 90, near = 0.01, far = 5, cameraEyePosition=[0.8, 0, 0.4], cameraTargetPosition=[0.0, 0, 0.0], startPose=[0.5, 0, 0.135, math.pi, 0, 0.744, 0.5], useMouse = False, visualize_path = True):

        # get parameters
        self.cubeZ = 0.015 # z start pos of cubes
        self.width = width
        self.height = height
        self.fov = fov
        self.near = near
        self.far = far
        self.cameraEyePosition = cameraEyePosition
        self.cameraTargetPosition = cameraTargetPosition
        self.startPose = startPose
        self.goalPos = [0.5, 0.0, 0.01]
        self.cubeSideLength = 0.03
        # mouse controller
        self.useMouse = useMouse # use it?
        self.is_dragging = False
        self.last_mouse_pos = None
        self.current_z = startPose[2]  # Initial Z height
        self.current_orientation = [math.pi, 0, 0.744]  # Default orientation
        self.visualize_path = visualize_path # show 8 path spheres?
        

        p.connect(p.GUI)
        if visualize_path == True:
            self.existing_spheres = self.create_spheres()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Register mouse events if needed

        p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 0) # mouse picking not allowed!
        if useMouse:
            self.register_mouse_events()
            
            
        else:
             # add sliders
            self.add_gui_sliders()

        #disable rendering during creation.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        

        
        #p.configureDebugVisualizer(p.COV_ENABLE_PLANAR_REFLECTION, 1)
        #print('pybullet data path: ',pybullet_data.getDataPath())
        
        # Define the bounds of the rectangle in x and y axes
        self.x_min, self.x_max = 0.28, 0.4  # Example: Rectangle width in x-axis
        self.y_min, self.y_max = -0.35, 0.35  # Example: Rectangle height in y-axis


        self.end_effector_index = 7
        self.panda = self.load_robot()
        

        if self.panda:      
        
            self.control_joints = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"]
            self.joint_type_list = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
            self.joint_info = namedtuple("jointInfo", ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"])
            self.num_joints = p.getNumJoints(self.panda)
            self.joints = Dict()
            self.current_joint_angles = [-0.6002407739360265, 0.515839628272284, 0.419641777995558, -2.329040114381972, -0.558347742443123, 2.752384990431231, -0.4498838863576673] # vor ik
            for i in range(self.num_joints):
                info = p.getJointInfo(self.panda, i)
                jointID = info[0]
                jointName = info[1].decode("utf-8")
                jointType = self.joint_type_list[info[2]]
                jointLowerLimit = info[8]
                jointUpperLimit = info[9]
                jointMaxForce = info[10]
                jointMaxVelocity = info[11]
                controllable = True if jointName in self.control_joints else False
                info = self.joint_info(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
                if info.type == "REVOLUTE":
                    p.setJointMotorControl2(self.panda, info.id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
                self.joints[info.name] = info     
        
             
            # Reset the debug visualizer camera
            camera_distance = 1.5        # Distance from the target
            camera_yaw = 90              # Yaw angle (left-right rotation)
            camera_pitch = -56          # Pitch angle (up-down rotation)
            camera_target_position = [0, 0, 0]  # The target point of the camera in world space (where the camera looks)

            p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)

            # add gravity
            p.setGravity(0, 0, -9.81)
            # Load cubes to interact with
                        
            self.red_cube_id = self.create_cube(pos=[0.3, -0.15, 0.015],cube_half_extents=[self.cubeSideLength, self.cubeSideLength, self.cubeSideLength],rgbaColor=[1, 0, 0, 1],baseMass=1, hasCollision=True)
            #self.blue_cube_id = self.create_cube(pos=[0.3, 0.15, 0.015],cube_half_extents=[0.03, 0.03, 0.03],rgbaColor=[0, 0, 1, 1],baseMass=1, hasCollision=True)

            # create red goal area
            self.goal_area_red_id = self.create_cube(pos=self.goalPos,cube_half_extents=[self.cubeSideLength, self.cubeSideLength, 0.01],rgbaColor=[1, 0, 0, 0.3],baseMass=0, hasCollision=False)
            

            # enable realtimesim now
            p.setRealTimeSimulation(True)
            #set pose
            self.set_joint_angles([-0.6002407739360265, 0.515839628272284, 0.419641777995558, -2.329040114381972, -0.558347742443123, 2.752384990431231, -0.4498838863576673, 0.0, 0.0, 0.01999988588968471])
                       
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            time.sleep(0.2)

            # create drag handle
            #if useMouse: self.create_drag_handle()


    def reset(self): 
        # Reset sliders
        p.removeAllUserParameters()
        if self.useMouse:
            self.register_mouse_events()
            #reset handle
            #p.resetBasePositionAndOrientation(self.handle_id, [self.startPose[0], self.startPose[1],self.startPose[2] + 0.25], p.getQuaternionFromEuler([0, 0, 0]))
                
        else:
            # add sliders
            self.add_gui_sliders()
        # reset cube
        # Define cube size and a safe distance threshold (e.g., slightly larger than cube size)
        

        # Generate random position for the red cube
        red_cube_pos = [
            random.uniform(self.x_min, self.x_max),
            random.uniform(self.y_min, self.y_max),
            self.cubeZ
        ]
        #reset robot
        self.set_joint_angles([-0.6002407739360265, 0.515839628272284, 0.419641777995558, -2.329040114381972, -0.558347742443123, 2.752384990431231, -0.4498838863576673, 0.0, 0.0, 0.01999988588968471])
        time.sleep(0.8)
        # Set the positions for cube
        p.resetBasePositionAndOrientation(self.red_cube_id, red_cube_pos, p.getQuaternionFromEuler([0, 0, 0]))
        #p.resetBasePositionAndOrientation(self.blue_cube_id, blue_cube_pos, p.getQuaternionFromEuler([0, 0, 0]))
    
    def closeEnv(self):
        # Close the PyBullet environment
        p.disconnect()

    def create_cube(self, pos, cube_half_extents, rgbaColor, baseMass, hasCollision):
        #cube_half_extents =   # This creates a cube with 0.1m sides

        # Create the cube's collision shape
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=cube_half_extents
        )

        # Create a visual shape for the red cube 
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=cube_half_extents,
            rgbaColor=rgbaColor#[1, 0, 0, 1]  # Red cube
        )

        # Create the cube body
        return (p.createMultiBody(
            baseMass=baseMass,  # Mass of the cube
            baseCollisionShapeIndex=collision_shape_id if hasCollision else -1,
            baseVisualShapeIndex=visual_shape_id,
            basePosition=pos  # Initial position of the cube
        ))
    
    def update_path_spheres(self, path):
        """
        Create spheres to visualize a path in PyBullet, removing previous spheres.
        
        Args:
        - path (np.ndarray): Array of 3D positions to visualize        
        """
        
        #for sphere_id in existing_spheres:
        #    p.removeBody(sphere_id) 
            
        
        # Ensure path is numpy array and has at least 2 dimensions
        path = np.array(path)
        
        if path.ndim == 0 or (path.ndim == 1 and path.size == 0):
            return []
        # move existing spheres if present
        if self.existing_spheres:
            for i in range(len(self.existing_spheres)):
                #p.removeBody(sphere_id)
                p.resetBasePositionAndOrientation(self.existing_spheres[i], path[i], p.getQuaternionFromEuler([0, 0, 0]))
            
    def create_spheres(self):
        # Create new spheres
        radius=0.008
        color=[0, 0, 1, 0.8]
        path = np.array([
            [ 0, 0,  -1],
            [ 0, 0,  -1],
            [ 0, 0,  -1],
            [ 0, 0,  -1],
            [ 0, 0,  -1],
            [ 0, 0,  -1],
            [ 0, 0,  -1]
        ])
        sphere_ids = []
        for position in path:
            sphere_id = p.createMultiBody(
                baseMass=0,  # No mass
                baseCollisionShapeIndex=-1,  # No collision
                baseVisualShapeIndex=p.createVisualShape(
                    shapeType=p.GEOM_SPHERE, 
                    radius=radius,
                    rgbaColor=color
                ),
                basePosition=position
            )
            sphere_ids.append(sphere_id)                
        return sphere_ids  
            

    def get_object_position(self, object_id):
        """
        Get the position of an object in PyBullet simulation.
        
        Args:
            object_id: The PyBullet object ID
            
        Returns:
            position: [x, y, z] coordinates
        """
        position, orientation = p.getBasePositionAndOrientation(object_id)
        return position
    
    def get_cube_pos(self):
        return(self.get_object_position(self.red_cube_id))
    
    def is_close_to_Goal(self, position, radius):
        """
        Check if an object's position is within a circular area on the XY plane.
        
        Args:
            position: [x, y, z] coordinates of the object
            circle_center: [x, y] coordinates of circle center
            radius: radius of the circle
            
        Returns:
            bool: True if object is within circle with radius around goal, False otherwise
        """
        # Only consider x and y coordinates
        object_x, object_y = position[0], position[1]
        center_x, center_y, center_z = self.goalPos
        
        # Calculate distance using Pythagorean theorem
        distance = np.sqrt((object_x - center_x)**2 + (object_y - center_y)**2)
        
        return distance <= radius
    
    
    def isDone(self, radius):
        return self.is_close_to_Goal(self.get_object_position(self.red_cube_id), radius)

    def load_robot(self):

        # Load the URDF file
        
        flags = p.URDF_USE_SELF_COLLISION
        

        # place floor

        floor_collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.6, 0.6, 0.01]
        )

        # Create a visual shape for the red cube 
        floor_shape_id = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.6, 0.6, 0.01],
            rgbaColor=[139/255, 115/255, 85/255, 1]
        )

        floor = p.createMultiBody(
            baseMass=0,  # Mass of the cube
            baseCollisionShapeIndex=floor_collision_shape_id,
            baseVisualShapeIndex=floor_shape_id,
            basePosition=[0.3, 0.0, -0.01]  # Initial position of the mirror
        )

        robot = p.loadURDF(ROBOT_URDF_PATH, [0, 0, 0], [0, 0, 0, 1], flags=flags, useFixedBase=True)
        return robot
    

    def set_joint_angles(self, joint_angles):
        #print(f"joint_angles: {joint_angles}")
        poses = []
        indexes = []
        forces = []
        open_amount = joint_angles[9] # get degree of openness of finger
        joint_angles = joint_angles[:7]  # Taking only the first 7 values        
        poses = []
        indexes = []
        forces = []

        for i, name in enumerate(self.control_joints[:7]):  # Only iterate over the first 7 control joints
            joint = self.joints[name]
            poses.append(joint_angles[i])
            indexes.append(joint.id)
            forces.append(joint.maxForce)

        # Set the joint motor controls for the Panda's 7 DOF joints
        p.setJointMotorControlArray(
            self.panda, indexes,
            p.POSITION_CONTROL,
            targetPositions=poses,  # Use 'poses' which is already aligned with the 7 joint angles
            targetVelocities=[0] * len(poses),
            positionGains=[0.04] * len(poses),
            forces=forces
        )

        
        """
        Control the Panda gripper.
        open_amount: [0, 0.04] for each finger
        """
                
        # The Panda gripper has a range of [0, 0.04] for each finger
        finger_position = open_amount
        
        p.setJointMotorControl2(self.panda, 9, p.POSITION_CONTROL, finger_position)
        p.setJointMotorControl2(self.panda, 10, p.POSITION_CONTROL, finger_position) 


    def get_joint_angles(self):
        return [p.getJointState(self.panda, i)[0] for i in range(7)]  # Get angles for the first 7 joints
    

    def check_collisions(self):
        collisions = p.getContactPoints()
        if len(collisions) > 0:
            print("[Collision detected!] {}".format(datetime.now()))
            return True
        return False


    def calculate_ik(self, position, orientation):
        quaternion = p.getQuaternionFromEuler(orientation)

        joint_damping = [0.5] * 9  # Increased damping

        joint_angles = p.calculateInverseKinematics(
            self.panda, self.end_effector_index, position, quaternion,
            jointDamping=joint_damping,
            solver=p.IK_DLS,
            maxNumIterations=100,
            residualThreshold=1e-5
        )

        # Smooth the transition
        smoothed_angles = joint_angles[:7] 
        self.current_joint_angles = smoothed_angles

        # Handle finger position
        if self.useMouse == False:
            finger_openstate = p.readUserDebugParameter(self.sliders[6]) * 0.04
        else:
            finger_openstate = p.readUserDebugParameter(self.gripper_slider) * 0.04

        smoothed_angles = list(smoothed_angles)
        smoothed_angles.append(0)
        smoothed_angles.append(0)
        smoothed_angles.append(finger_openstate)

        return smoothed_angles

    def add_gui_sliders(self):
        self.sliders = []
        self.sliders.append(p.addUserDebugParameter("X", 0, 1, self.startPose[0]))
        self.sliders.append(p.addUserDebugParameter("Y", -1, 1, self.startPose[1]))
        self.sliders.append(p.addUserDebugParameter("Z", 0.12, 0.15, self.startPose[2]))
        self.sliders.append(p.addUserDebugParameter("Rx", -math.pi, math.pi, self.startPose[3]))
        self.sliders.append(p.addUserDebugParameter("Ry", -math.pi/2, math.pi/2, self.startPose[4]))
        self.sliders.append(p.addUserDebugParameter("Rz", -math.pi/2, math.pi/2, self.startPose[5]))
        self.sliders.append(p.addUserDebugParameter("Rz", 0, 1, self.startPose[6])) # fingers


    def read_gui_sliders(self):
        x = p.readUserDebugParameter(self.sliders[0])
        y = p.readUserDebugParameter(self.sliders[1])
        z = p.readUserDebugParameter(self.sliders[2])
        Rx = p.readUserDebugParameter(self.sliders[3])
        Ry = p.readUserDebugParameter(self.sliders[4])
        Rz = p.readUserDebugParameter(self.sliders[5])
        finger = p.readUserDebugParameter(self.sliders[6])
        return [x, y, z, Rx, Ry, Rz, finger]
    
    def read_gui_mouse(self):
        x = self.get_current_pose()[0][0]
        y = self.get_current_pose()[0][1]
        z = self.get_current_pose()[0][2] 
        Rx = self.startPose[3]
        Ry = self.startPose[4]
        Rz = self.startPose[5]
        finger = p.readUserDebugParameter(self.gripper_slider)
        return [x, y, z, Rx, Ry, Rz, finger]
        
    def get_current_pose(self):
        linkstate = p.getLinkState(self.panda, self.end_effector_index, computeForwardKinematics=True)
        position, orientation = linkstate[0], linkstate[1]
        return (position, orientation)
    

    def get_camera_image(self, fixed=True, segmentation=False):
        """
        Captures an image from the virtual fixed camera or the one located at the EE.
        If segmentation is True, returns the segmentation mask instead of the RGB image.
        """
        
        # Fixed camera setup
        if fixed:
            viewMatrix = p.computeViewMatrix(
                cameraEyePosition=self.cameraEyePosition,
                cameraTargetPosition=self.cameraTargetPosition,
                cameraUpVector=[0, 0, 1]
            )
        else:
            # End effector camera setup
            linkState = p.getLinkState(self.panda, 11)
            linkPos = linkState[0]
            cameraPos = [linkPos[0], linkPos[1], linkPos[2] + 0.04]
            cameraTarget = [linkPos[0], linkPos[1], linkPos[2] - 1]
            viewMatrix = p.computeViewMatrix(cameraPos, cameraTarget, [0, 1, 0])

        projectionMatrix = p.computeProjectionMatrixFOV(self.fov, self.width/self.height, self.near, self.far)
        
        img_arr = p.getCameraImage(self.width, self.height, viewMatrix, projectionMatrix, 
                                renderer=p.ER_BULLET_HARDWARE_OPENGL)
        
        if segmentation:
            # Extract segmentation mask
            segmentation_mask = np.reshape(img_arr[4], (self.height, self.width))
            #print('segmentation_to_rgb(segmentation_mask).shape: ', segmentation_to_rgb(segmentation_mask).shape)
            return segmentation_to_rgb(segmentation_mask)
        else:
            # Extract RGB image
            rgb_image = np.reshape(img_arr[2], (self.height, self.width, 4))[:, :, :3]
            if rgb_image.dtype == np.int64:
                rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
            return rgb_image

    

    def get_camera_orientation(self, position, target):
        # Compute forward vector
        forward = np.array(target) - np.array(position)
        forward = forward / np.linalg.norm(forward)
        
        # Compute up vector (assuming world up is [0, 0, 1])
        world_up = np.array([0, 0, 1])
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        
        # Construct rotation matrix
        rotation_matrix = np.column_stack((right, up, -forward))
        
        # Convert rotation matrix to quaternion
        trace = np.trace(rotation_matrix)
        if trace > 0:
            S = np.sqrt(trace + 1.0) * 2
            qw = 0.25 * S
            qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / S
            qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / S
            qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / S
        elif rotation_matrix[0, 0] > rotation_matrix[1, 1] and rotation_matrix[0, 0] > rotation_matrix[2, 2]:
            S = np.sqrt(1.0 + rotation_matrix[0, 0] - rotation_matrix[1, 1] - rotation_matrix[2, 2]) * 2
            qw = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / S
            qx = 0.25 * S
            qy = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / S
            qz = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / S
        elif rotation_matrix[1, 1] > rotation_matrix[2, 2]:
            S = np.sqrt(1.0 + rotation_matrix[1, 1] - rotation_matrix[0, 0] - rotation_matrix[2, 2]) * 2
            qw = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / S
            qx = (rotation_matrix[0, 1] + rotation_matrix[1, 0]) / S
            qy = 0.25 * S
            qz = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / S
        else:
            S = np.sqrt(1.0 + rotation_matrix[2, 2] - rotation_matrix[0, 0] - rotation_matrix[1, 1]) * 2
            qw = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / S
            qx = (rotation_matrix[0, 2] + rotation_matrix[2, 0]) / S
            qy = (rotation_matrix[1, 2] + rotation_matrix[2, 1]) / S
            qz = 0.25 * S
        
        return [qx, qy, qz, qw]
 
            
    def create_drag_handle(self):
        # Create a small visual and collision sphere as the drag handle
        visual_shape_id = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.02,
            rgbaColor=[0, 1, 0, 0.5]  # Semi-transparent green
        )
        
        collision_shape_id = p.createCollisionShape(
            shapeType=p.GEOM_SPHERE,
            radius=0.02
        )
        
        # Get initial end effector position
        ee_pos = self.get_current_pose()[0]
        
        # Create the handle as a dynamic object
        self.handle_id = p.createMultiBody(
            baseMass=0.1,
            baseVisualShapeIndex=visual_shape_id,
            baseCollisionShapeIndex=collision_shape_id,
            basePosition=[ee_pos[0], ee_pos[1], ee_pos[2] + 0.25]
        )
        
        # Enable the handle to be directly manipulated by the user
        p.changeDynamics(self.handle_id, -1, linearDamping=0.1, angularDamping=0.1)
        #p.enableJointForceTorqueSensor(self.handle_id, -1, 1)   
    
        
    def register_mouse_events(self):
        # Remove existing sliders as we'll use mouse control instead
        if hasattr(self, 'sliders'):
            p.removeAllUserDebugParameters()
            
        # Add a single slider for gripper control
        self.gripper_slider = p.addUserDebugParameter("Gripper", 0, 1, 0.5)
        

    def process_events(self):
        # Get mouse events
        events = p.getMouseEvents()
        keys = p.getKeyboardEvents()
        
        for e in events:
            if e[0] == 1:  # Mouse move
                self.handle_mouse_move(e)
            elif e[0] == 2:  # Mouse button
                self.handle_mouse_click(e)
        
        # Process keyboard for Z height control
        self.handle_keyboard(keys)
        
        # Always update positions
        handle_pos = self.get_current_pose()[0]
        # make handle float
        #p.resetBasePositionAndOrientation(self.handle_id, [handle_pos[0], handle_pos[1], 0.5], p.getQuaternionFromEuler([0, 0, 0]))          
    
                
    def handle_keyboard(self, keys):
        z_step = 0.005
        for key in keys:
            if key == p.B3G_UP_ARROW and (keys[key] & p.KEY_IS_DOWN):
                self.current_z = min(self.current_z + z_step, 0.15)
                # Get the current handle position
                handle_pos = self.get_current_pose()[0]
                # Calculate and set the joint angles based on the new end effector position
                joint_angles = self.calculate_ik(
                    [handle_pos[0], handle_pos[1],self.current_z],
                    self.current_orientation
                )
                self.set_joint_angles(joint_angles)
            elif key == p.B3G_DOWN_ARROW and (keys[key] & p.KEY_IS_DOWN):
                self.current_z = max(self.current_z - z_step, 0.12)
                # Get the current handle position
                handle_pos = self.get_current_pose()[0]
                # Calculate and set the joint angles based on the new end effector position
                joint_angles = self.calculate_ik(
                    [handle_pos[0], handle_pos[1],self.current_z],
                    self.current_orientation
                )
                self.set_joint_angles(joint_angles)
                    
    def handle_mouse_click(self, event):
        button = event[4]
        #state = event[3]  # 1 for pressed, 0 for released
        
        #print(f'Mouse click event: {event}')
       
        if button == 3:  # Left click
            #print("Click start")
            self.is_dragging = True
        if button == 4:
            #print("Click end")
            self.is_dragging = False  
            

    def handle_mouse_move(self, event):
        # Update the handle position directly based on mouse events
        
        if self.is_dragging: 
            mouse_pos_real_world = self.screen_to_world(event[1], event[2])
            #print(f"Mouse move - screen_x: {event[1]} screen_y: {event[2]} -Mouse cords: {mouse_pos_real_world}")

            # Limit movement to reasonable workspace
            #mouse_pos_real_world[0] = np.clip(mouse_pos_real_world[0], 0.2, 0.8) # x
            #mouse_pos_real_world[1] = np.clip(mouse_pos_real_world[1], -0.4, 0.4) # y
            # Calculate and set the joint angles based on the new end effector position
            joint_angles = self.calculate_ik(
                [mouse_pos_real_world[0], mouse_pos_real_world[1], self.current_z],
                self.current_orientation
            )
            self.set_joint_angles(joint_angles)
        
    
    def screen_to_world(self, mouse_x, mouse_y):
        """
        Convert screen coordinates to world coordinates.
        
        Parameters:
        mouse_x (float): X-coordinate of the mouse pointer on the screen.
        mouse_y (float): Y-coordinate of the mouse pointer on the screen.
        
        Returns:
        numpy.ndarray: 2D array containing the world x and y coordinates.
        """
        # Get screen width and height
        width, height = p.getDebugVisualizerCamera()[0], p.getDebugVisualizerCamera()[1]
        #print(f"cam: {p.getDebugVisualizerCamera()}")

        #print(f"mouse_x: {mouse_x}")
        #print(f"mouse_y: {mouse_y}")
        
        # Adjust for PyBullet's top-left origin
        norm_x = (mouse_x / width) * 2 - 1
        norm_y = (mouse_y / height) * 2 - 1

        #print(f"norm_x: {norm_x}")
        #print(f"norm_y: {norm_y}")
        
        
        # Get camera data for conversion
        cam_data = p.getDebugVisualizerCamera()
        view_matrix = np.array(cam_data[2]).reshape(4, 4)
        proj_matrix = np.array(cam_data[3]).reshape(4, 4)
        
        # Get the camera view and projection matrices
        # Get the camera position and orientation
               
        # Depth (set close to 1.0 for objects close to the far clipping plane)
        depth = 1.0 # p.getDebugVisualizerCamera()[10] # dist
       
        # Transform screen coordinates to homogeneous clip coordinates
        clip_coords = np.array([norm_x, norm_y, depth, 1.0])
        
        # Convert clip coordinates to camera space
        inv_proj_matrix = np.linalg.inv(proj_matrix)
        cam_coords = inv_proj_matrix @ clip_coords
        cam_coords /= cam_coords[3]  # Perspective divide
        
        # Convert camera coordinates to world coordinates
        inv_view_matrix = np.linalg.inv(view_matrix)
        world_coords = inv_view_matrix @ cam_coords

       

        #print(f"list(world_coord): {world_coords}") 
        return [norm_y, norm_x]#world_coords[:3]
    
   
