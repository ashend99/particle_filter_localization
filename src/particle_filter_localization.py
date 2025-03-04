#!/usr/bin/env python3

# import libraries
import rospy
import random
import time
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, TransformStamped, Twist
from nav_msgs.srv import GetMap
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from tf import TransformListener
from tf.transformations import quaternion_matrix, inverse_matrix, concatenate_matrices, quaternion_from_matrix
import tf2_ros
import numpy as np
import cv2 as cv
import math
from threading import Lock
import copy
from plotter import Plotter
import os 

# object to save particles including pose and weight
class Particle():
    def __init__(self, x, y, theta, weight):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight

# Partcle Filter
class ParticleFilter():
    def __init__(self):
        self.filepath = os.path.dirname(__file__)
        self.images_dir_path = self.filepath + "/../images/"

        # ROS parameters
        self.sampling_method = rospy.get_param("~sampling_method", "uniform")   # Sampling method for particle initialization
        self.num_particles = rospy.get_param("~num_particles", 250) # Number of particles
        self.num_motion_updates = rospy.get_param("~num_motion_updates", 2) # Number of motion updates before a sensor update
        self.num_scan_rays = rospy.get_param("~num_scan_rays", 20) # Number of scan rays to evaluate
        self.num_sensing_updates = rospy.get_param("~num_sensing_updates", 10) # Number of sensing updates before resampling
        self.motion_distance_noise_stddev = rospy.get_param("~motion_distance_noise_stddev", 0.01) # Standard deviation of distance noise for motion update
        self.motion_rotation_noise_stddev = rospy.get_param("~motion_rotation_noise_stddev", math.pi / 120) # Standard deviation of rotation noise for motion update
        self.sensing_noise_stddev = rospy.get_param("~sensing_noise_stddev", 0.5) # Standard deviation of sensing noise

        self.pre_vel_time = 0           # time when last velocity message arrived

        self.prev_odom_msg = 0

        self.motion_update_count = 0    # number of motion updates since last sensor update
        self.sensing_update_count = 0   # number of sensor updates since last resampling
        self.estimated_pose_valid = False       # whether the estimated pose is valid or not

        self.estimated_pose = Pose()    # estimated pose
        self.lock = Lock()              # lock for acquaring and releasing when particles array is updated in many functions
        self.plotter = Plotter()        # visualize the weights distribution

        # ROS publishers and subscribers
        self.particlesPub = rospy.Publisher("/particlecloud", PoseArray, queue_size=10)
        rospy.Subscriber("/scan", LaserScan, self.scanCallback, queue_size=1)
        rospy.Subscriber("/odom", Odometry, self.odomCallback, queue_size=1)

        self.particles_sequence_number = 0      # publishing sequence number of particles array
        self.particles_pub_timer = rospy.Timer(rospy.Duration(0.5), self.publishParticles)      # publish particles at 0.1 second intervals

        self.estimated_pose_pub = rospy.Publisher('estimated_pose', PoseStamped, queue_size=1)
        self.estimated_pose_sequence_number = 0     # estimated pose sequence number for publishing
        self.estimated_pose_pub_timer = rospy.Timer(rospy.Duration(0.5), self.publishEstimatedPose)     # publish the estimated pose at 0.1 second intervals
        
        self.transform_broadcaster = tf2_ros.TransformBroadcaster()     # sending transformations
        self.transform_sequence_number = 0          # transformation sequence number for publishing


        # First, get the static map and initialize the particles
        self.lock.acquire()

        self.getMap()
        self.initializeParticles()

        self.lock.release()

    # get the static map
    def getMap(self):
        # call the map service
        rospy.loginfo("Waiting for static map service.")
        rospy.wait_for_service("static_map")
        try:
            get_map = rospy.ServiceProxy('/static_map', GetMap)
            response = get_map()
            self.map = response.map     # get the map
            rospy.loginfo("Map received.")
        except rospy.ServiceException as e:
            print("Service call failed:", e)
        
        # map width and height
        self.map_width = self.map.info.width
        self.map_height = self.map.info.height

        # map origin
        self.map_origin = self.map.info.origin

        # get an image of the map
        self.map_image = np.reshape(self.map.data, (self.map_height, self.map_width)).astype(np.uint8)
    
        # dilated map image
        kernel = np.ones((3, 3), np.uint8)   # kernel for dilation
        threshold_value = 90
        _, binary_image = cv.threshold(self.map_image.astype(np.uint8), threshold_value, 255, cv.THRESH_BINARY)     # thresholding
        binary_image = np.flip(binary_image, axis=0)    # flip the image since it has a flipped x axis in the RViz map
        self.dilated_map_image = cv.dilate(binary_image, kernel, iterations=1)   # dilate
        
        # define the map boundaries
        self.map_x_min = self.map_origin.position.x + 10
        self.map_x_max = self.map_x_min + 5
        self.map_y_min = self.map_origin.position.y + 10
        self.map_y_max = self.map_y_min + 5

        # Preprocess the distance transform for fast ray casting
        self.map_image_distance_transform = self.distance_transform(self.map_image)


    # Initialize the particles
    def initializeParticles(self):

        # empty array
        particles = []

        # get particles
        for p in range(self.num_particles):
            while True:
                # random particles using defined sampling 
                # method (uniform or normal)
                particle = self.getParticles()   
                if self.isValidParticle(particle):
                    particles.append(particle)
                    break

        self.particles_array = particles

        # normalize particle weights
        self.normalizeWeights()

    # get velocity commands for motion update
    def velocityCallback(self, vel_msg):
        cur_time = rospy.Time.now()     # current time

        # check whether the first message is
        if self.pre_vel_time == 0:
            self.pre_vel_time = rospy.Time.now()
            return

        time_diff = (cur_time - self.pre_vel_time).to_sec()     # time difference in seconds

        # moved distance and rotation during the time interval
        _, _, yaw = euler_from_quaternion([self.estimated_pose.orientation.x, self.estimated_pose.orientation.y,
                                            self.estimated_pose.orientation.z, self.estimated_pose.orientation.w])
        delta_x = vel_msg.linear.x * math.cos(yaw) * time_diff
        delta_y = vel_msg.linear.x * math.sin(yaw) * time_diff

        delta_theta = vel_msg.angular.z * time_diff
        distance = (delta_x**2 + delta_y**2)**0.5

        # return, if no motion
        if distance == 0 and delta_theta == 0:
            return 
        
        # add the distance and rotation for every particle
        for p in self.particles_array:
            distance_noise = self.normalSampling(0, self.motion_distance_noise_stddev)
            rotation_noise = self.normalSampling(0, self.motion_rotation_noise_stddev)
            p.x = p.x + (distance + distance_noise) * math.cos(p.theta)
            p.y = p.y + (distance + distance_noise) * math.sin(p.theta)
            theta_ = p.theta + delta_theta + rotation_noise
            p.theta = self.wrap_angle(theta_)
    
        # After updating, keep only the particles that are inside the map
        old_particles = copy.deepcopy(self.particles_array)
        particles_array = []

        for p in old_particles:
            if self.isValidParticle(p):
                particles_array.append(p)
        
        self.particles_array = particles_array

        # Normalise particle weights
        self.normalizeWeights()

        # If the estimated pose is valid move it too
        if self.estimated_pose_valid:
            estimated_pose_theta = 2. * math.acos(self.estimated_pose.orientation.w)

            if self.estimated_pose.orientation.z < 0.:
                estimated_pose_theta = -estimated_pose_theta

            self.estimated_pose.position.x += math.cos(estimated_pose_theta) * distance
            self.estimated_pose.position.y += math.sin(estimated_pose_theta) * distance

            estimated_pose_theta = self.wrap_angle(estimated_pose_theta + delta_theta)

            self.estimated_pose.orientation.w = math.cos(estimated_pose_theta / 2.)
            self.estimated_pose.orientation.z = math.sin(estimated_pose_theta / 2.)

        # Increment the motion update counter
        self.motion_update_count = self.motion_update_count + 1

        # save the time
        self.pre_vel_time = cur_time

    def odomCallback(self, odom_msg):
        # Skip the first call since we are looking for movements
        if not self.prev_odom_msg:
            self.prev_odom_msg = odom_msg
            return

        # # Distance moved since the previous odometry message
        global_delta_x = odom_msg.pose.pose.position.x - self.prev_odom_msg.pose.pose.position.x
        global_delta_y = odom_msg.pose.pose.position.y - self.prev_odom_msg.pose.pose.position.y

        distance = math.sqrt(math.pow(global_delta_x, 2) + math.pow(global_delta_y, 2))

        # Previous robot orientation
        prev_theta = 2 * math.acos(self.prev_odom_msg.pose.pose.orientation.w)

        if self.prev_odom_msg.pose.pose.orientation.z < 0:
            prev_theta = -prev_theta

        # Figure out if the direction is backward
        if (prev_theta < 0 and global_delta_y > 0) or (prev_theta > 0 and global_delta_y < 0):
            distance = -distance

        # Current orientation
        theta = 2 * math.acos(odom_msg.pose.pose.orientation.w)

        if odom_msg.pose.pose.orientation.z < 0:
            theta = -theta

        # Rotation since the previous odometry message
        rotation = theta - prev_theta

        # Return if the robot hasn't moved
        if abs(distance) < 0.01 and abs(rotation) < 0.05:
            # print("removing: ", distance, rotation)
            return

        # self.lock.acquire()

        # print("odom: ", odom_msg.header.stamp, "   now: ", rospy.Time.now())

        # print("robot is moving: ", distance, rotation)

        # update particle values with motion updates
        for p in self.particles_array:
            distance_noise = self.normalSampling(0, self.motion_distance_noise_stddev)
            rotation_noise = self.normalSampling(0, self.motion_rotation_noise_stddev)
            p.x = p.x + (distance + distance_noise) * math.cos(p.theta)
            p.y = p.y + (distance + distance_noise) * math.sin(p.theta)
            theta_ = p.theta + rotation + rotation_noise
            p.theta = self.wrap_angle(theta_)


        # Overwrite the previous odometry message
        self.prev_odom_msg = odom_msg

        # Delete any particles outside of the map
        old_particles = copy.deepcopy(self.particles_array)
        self.particles_array = []

        for p in old_particles:
            # if not(p.x < self.map_x_min or p.x > self.map_x_max or p.y < self.map_y_min or p.y > self.map_y_max):
            if self.isValidParticle(p):
                # Keep it
                self.particles_array.append(p)

        # Normalise particle weights because particles have been deleted
        self.normalizeWeights()

        # If the estimated pose is valid move it too
        # if self.estimated_pose_valid_:
        estimated_pose_theta = 2. * math.acos(self.estimated_pose.orientation.w)

        if self.estimated_pose.orientation.z < 0.:
            estimated_pose_theta = -estimated_pose_theta

        self.estimated_pose.position.x += math.cos(estimated_pose_theta) * distance
        self.estimated_pose.position.y += math.sin(estimated_pose_theta) * distance

        estimated_pose_theta = self.wrap_angle(estimated_pose_theta + rotation)

        self.estimated_pose.orientation.w = math.cos(estimated_pose_theta / 2.)
        self.estimated_pose.orientation.z = math.sin(estimated_pose_theta / 2.)

        # Increment the motion update counter
        self.motion_update_count = self.motion_update_count + 1

        # self.lock.release()
    

    # callback function for laser data from the LiDAR sensor
    def scanCallback(self, msg):
        # Corrections after defined number of motion updates
        if self.motion_update_count < self.num_motion_updates:
            return
    
        # select only few lidar rays to check the likelihood in order to reduce the computational complexity
        # calculate a step count to skip the lidar rays
        step = int(math.floor(float(len(msg.ranges))/self.num_scan_rays))

        # self.lock.acquire()

        # Correction step: get the likelihood and multiply with particle weights
        for p in self.particles_array:
            likelihood = 1      # starting likelihood

            for i in range(0, len(msg.ranges), step):

                scan_range = msg.ranges[i]      # laser ray
                laser_angle = msg.angle_increment * i + msg.angle_min       # laser angle
                global_angle = self.wrap_angle(p.theta + laser_angle)

                # get the obstacles distances around the robot
                particle_range = self.hit_scan(p.x, p.y, global_angle, 10.0) 

                # calculate the likelihood per ray
                ray_likelihood = 1/math.sqrt(2*math.pi*math.pow(self.sensing_noise_stddev, 2))*math.exp(-math.pow((particle_range-scan_range),2)/(2*math.pow(self.sensing_noise_stddev,2)))
                
                # update the likelihood 
                likelihood *= ray_likelihood

            # update the particle weight with the calculated likelihood
            p.weight *= likelihood

        # normalize the weights
        self.normalizeWeights()

        # estimate the robot pose
        self.estimate_pose()

        self.sensing_update_count = self.sensing_update_count + 1

        ## Resampling
        # get the N effective size
        N_eff = self.Neff()
        print(f"N effective: {N_eff}")

        # check the N effective is less than the threshold or else resample after defined number of sensor updates
        if N_eff < 300 or self.sensing_update_count > self.num_sensing_updates:
            self.resample_particles()
            self.sensing_update_count = 0

        self.motion_update_count = 0
        # self.lock.release()
        

    # calculate the N effective size    
    def Neff(self):
        sum_of_weights_squared = sum([p.weight**2 for p in self.particles_array])
        N_eff = 1/sum_of_weights_squared

        return N_eff

    # Resampling
    def resample_particles(self):

        # Copy old particles
        old_particles = copy.deepcopy(self.particles_array)
        particles = []

        # Iterator for old_particles
        old_particles_i = 0

        print("resampling started.")

        # Find a new set of particles by randomly stepping through the old set, biased by their probabilities
        while len(particles) < self.num_particles:

            value = self.uniformSampling(0.0, 1.0)
            sum = 0.0

            # Loop until a particle is found
            particle_found = False
            while not particle_found:

                # If the random value is between the sum and the sum + the weight of the particle
                if value > sum and value < (sum + old_particles[old_particles_i].weight):

                    # Add the particle to the array
                    particles.append(copy.deepcopy(old_particles[old_particles_i]))

                    # Add some noise to the particle
                    particles[-1].x = particles[-1].x + self.normalSampling(0, 0.02)
                    particles[-1].y = particles[-1].y + self.normalSampling(0, 0.02)
                    particles[-1].theta = self.wrap_angle(particles[-1].theta + self.normalSampling(0, math.pi / 120))
                    
                    # Break out of the loop
                    particle_found = True

                # Add particle weight to sum and increment the iterator
                sum = sum + old_particles[old_particles_i].weight
                old_particles_i = old_particles_i + 1

                # If the iterator passes number of particles, loop back to the beginning
                if old_particles_i >= len(old_particles):
                    old_particles_i = 0
        
        print("resampling is over.")
        
        # resampled particle array
        self.particles_array = particles

        # Normalise the new particles
        self.normalizeWeights()

        # Don't use the estimated pose just after resampling
        self.estimated_pose_valid = False

        # Induce a sensing update
        self.motion_update_count = self.num_motion_updates  

    # estimate the robot pose
    def estimate_pose(self):

        # weighted average pose
        x = 0
        y = 0
        theta_sin = 0
        theta_cos = 0
        for p in self.particles_array:
            x += p.x
            y += p.y
            theta_sin += math.sin(p.theta)
            theta_cos += math.cos(p.theta)
        
        n = len(self.particles_array)
        estimated_pose_x = x/n
        estimated_pose_y = y/n
        estimated_pose_theta = math.atan2(theta_sin/n, theta_cos/n)


        # Set the estimated pose message
        self.estimated_pose.position.x = estimated_pose_x
        self.estimated_pose.position.y = estimated_pose_y

        self.estimated_pose.orientation.w = math.cos(estimated_pose_theta / 2)
        self.estimated_pose.orientation.z = math.sin(estimated_pose_theta / 2)

        self.estimated_pose_valid = True
    

    # Find the nearest obstacle from position start_x, start_y (in meters) in direction theta
    def hit_scan(self, start_x, start_y, theta, max_range):        

        # Start point in map occupancy grid coordinates
        start_point = [int(round((start_x - self.map.info.origin.position.x) / self.map.info.resolution)),
                             int(round((start_y - self.map.info.origin.position.y) / self.map.info.resolution))]

        # End point in real coordinates
        end_x = start_x + math.cos(theta) * max_range
        end_y = start_y + math.sin(theta) * max_range

        # End point in occupancy grid coordinates
        end_point = [int(round((end_x - self.map.info.origin.position.x) / self.map.info.resolution)),
                           int(round((end_y - self.map.info.origin.position.y) / self.map.info.resolution))]

        # Find the first "hit" along scan
        hit = self.find_obstacles(self.map_image_distance_transform, start_point, end_point)

        # Convert hit back to world coordinates
        hit_x = hit[0] * self.map.info.resolution + self.map.info.origin.position.x
        hit_y = hit[1] * self.map.info.resolution + self.map.info.origin.position.y

        return math.sqrt(math.pow(start_x - hit_x, 2) + math.pow(start_y - hit_y, 2))

    # check whether there are obstacles on a line between two points
    def find_obstacles(self, img, p1, p2):
        # coordinates of points
        x1 = float(p1[0])
        y1 = float(p1[1])
        x2 = float(p2[0])
        y2 = float(p2[1])

        # x and y differences
        dx = x2 - x1
        dy = y2 - y1

        # calculate the distance between the two points and normalize
        l = (dx**2 + dy**2)**0.5
        dx = dx / l
        dy = dy / l

        step = 1.0 # pixels
        min_step = 3 # pixels -- too large risks jumping over obstacles
        max_steps = int(l)

        # check obstacles
        dist = 0
        while dist < max_steps:

            # Get the next pixel
            x = int(round(x1 + dx*dist))
            y = int(round(y1 + dy*dist))

            # Check if it's outside of the image
            if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
                return [x, y]

            current_distance = img[y, x]    # distance to the nearest obstacle from the next point

            # Check for osbtacles
            if current_distance <= 0:
                return [x, y]

            # Otherwise, adjust the step size according to the distance transform
            step = current_distance - 1

            # keep the minimum step size
            if step < min_step:
                step = min_step

            # Move along the ray
            dist += step

        # No obstacles found between the two points
        return p2

    # check whether the particle is inside the map, on an obstacle or outside the map
    def isValidParticle(self, particle):
        # get the x and y pixel values for x and y distance vakues
        p_x = self.map_origin.position.x - particle.x
        p_y = self.map_origin.position.y - particle.y
        x_pixel = int(abs(p_x) / self.map.info.resolution)
        y_pixel = self.map_height - int(abs(p_y) / self.map.info.resolution)
        
        # check the particle with the dilated map
        if self.dilated_map_image[y_pixel][x_pixel] != 255:
            return True
        return False
    
    # wrap an angle between 0 and 2*Pi
    def wrap_angle(self, angle):
        while angle < 0.0:
            angle = angle + 2 * math.pi

        while angle > 2 * math.pi:
            angle = angle - 2 * math.pi

        return angle
    
    # normalize the particle weights
    def normalizeWeights(self):
        # sum of all weights
        sum_of_weights = sum([p.weight for p in self.particles_array])

        if sum_of_weights == 0:
            print("weights are zero.", len(self.particles_array))
            return 
        
        # normalize
        for p in self.particles_array:
            p.weight = p.weight/sum_of_weights
        
        # plot the weights
        x = [p.x for p in self.particles_array]
        y = [p.weight for p in self.particles_array]
        self.plotter.set_data(x, y)
        self.plotter.update_plot()
    
    # get random particles using defined sampling method
    def getParticles(self):

        # uniform sampling
        if self.sampling_method == "uniform":
            x = self.uniformSampling(self.map_x_min, self.map_x_max)
            y = self.uniformSampling(self.map_y_min, self.map_y_max)
            theta = self.uniformSampling(-math.pi, math.pi)
            
            particle = Particle(x, y, theta, 1.0)
        
        # normal (gaussian) sampling
        elif self.sampling_method == "normal":
            x = self.normalSampling(self.uniformSampling(self.map_x_min, self.map_x_max), 0.05)
            y = self.normalSampling(self.uniformSampling(self.map_y_min, self.map_y_max), 0.05)
            theta = self.normalSampling(self.uniformSampling(-math.pi, math.pi), math.pi/60)
            particle = Particle(x, y, theta, 1.0)

        return particle

    # random value using uniform sampling
    def uniformSampling(self, a, b):
        return random.uniform(min(a, b), max(a, b))

    # random value using normal sampling
    def normalSampling(self, mean, std):
        return np.random.normal(mean, std)

    # publish the particles
    def publishParticles(self, event):
        # self.lock.acquire()

        # arrange the particles into a PoseArray
        self.pose_array = PoseArray()
        self.pose_array.header.frame_id = "map"
        self.pose_array.header.stamp = rospy.Time.now()
        self.pose_array.header.seq = self.particles_sequence_number
        self.particles_sequence_number += 1

        c = 0
        for p in self.particles_array:
            pose = Pose()
            pose.position.x = p.x
            pose.position.y = p.y
            _, _, z, w = quaternion_from_euler(0, 0, p.theta)
            pose.orientation.z = z
            pose.orientation.w = w
            self.pose_array.poses.append(pose)
            c += 1
        # self.pose_array.poses = [p.pose for p in self.particles_array]
        self.particlesPub.publish(self.pose_array)

        # self.lock.release()
    
    # publish the estimated pose and transformations
    def publishEstimatedPose(self, event): 
        # publish only if the estimated pose is valid       
        # if not self.estimated_pose_valid:
            # return

        # self.lock.acquire()

        # Publish the estimated pose
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.seq = self.estimated_pose_sequence_number
        self.estimated_pose_sequence_number = self.estimated_pose_sequence_number + 1

        pose_stamped.pose = self.estimated_pose

        self.estimated_pose_pub.publish(pose_stamped)


        #  publish the transformation from map to base_footprint
        transform = TransformStamped()

        transform.header.frame_id = "map"
        transform.header.stamp = rospy.Time.now()
        transform.header.seq = self.transform_sequence_number
        self.transform_sequence_number = self.transform_sequence_number + 1

        transform.child_frame_id = "base_footprint"

        transform.transform.translation.x = self.estimated_pose.position.x
        transform.transform.translation.y = self.estimated_pose.position.y

        transform.transform.rotation.w = self.estimated_pose.orientation.w
        transform.transform.rotation.z = self.estimated_pose.orientation.z
        

        # get odom to base_footprint tf
        listener = TransformListener()
        while not rospy.is_shutdown():
            try:
                # print("getting odom")        
                listener.waitForTransform("/odom", "/base_footprint", rospy.Time(), rospy.Duration(10.0))
                (odom_to_base_trans,odom_to_base_rot) = listener.lookupTransform('/odom', '/base_footprint',  rospy.Time())
                
                break
            except Exception as e:
                continue
        
        # calculate map to odom tf using map to base_footprint estimated pose and obtained odom to base_footprint tf
        map_to_base_trans = [self.estimated_pose.position.x, self.estimated_pose.position.y, self.estimated_pose.position.z]
        map_to_base_rot = [self.estimated_pose.orientation.x, self.estimated_pose.orientation.y, self.estimated_pose.orientation.z, self.estimated_pose.orientation.w]

        map_to_base_matrix = quaternion_matrix(map_to_base_rot)
        map_to_base_matrix[:3, 3] = map_to_base_trans

        odom_to_base_matrix = quaternion_matrix(odom_to_base_rot)
        odom_to_base_matrix[:3, 3] = odom_to_base_trans

        odom_to_base_inverse = inverse_matrix(odom_to_base_matrix)

        map_to_odom_matrix = concatenate_matrices(map_to_base_matrix, odom_to_base_inverse)

        map_to_odom_rot = quaternion_from_matrix(map_to_odom_matrix)
        map_to_odom_trans = map_to_odom_matrix[:3, 3]

        # create the tf
        map_to_odom_transform = TransformStamped()
        map_to_odom_transform.header.stamp = rospy.Time.now()
        map_to_odom_transform.header.frame_id = "map"
        map_to_odom_transform.child_frame_id = "odom"
        map_to_odom_transform.header.seq = self.transform_sequence_number
        self.transform_sequence_number = self.transform_sequence_number + 1

        map_to_odom_transform.transform.translation.x = map_to_odom_trans[0]
        map_to_odom_transform.transform.translation.y = map_to_odom_trans[1]
        map_to_odom_transform.transform.translation.z = map_to_odom_trans[2]
        map_to_odom_transform.transform.rotation.x = map_to_odom_rot[0]
        map_to_odom_transform.transform.rotation.y = map_to_odom_rot[1]
        map_to_odom_transform.transform.rotation.z = map_to_odom_rot[2]
        map_to_odom_transform.transform.rotation.w = map_to_odom_rot[3]

        # publish
        self.transform_broadcaster.sendTransform(map_to_odom_transform)

        # self.lock.release()

    # get the distance tranformed image
    def distance_transform(self, image):
        image_reformat = np.float32(image)

        # Threshold the image to convert a binary image
        ret, thresh = cv.threshold(image_reformat, 50, 1, cv.THRESH_BINARY_INV)
        # Determine the distance transform.
        dist = cv.distanceTransform(np.uint8(thresh), cv.DIST_L2, 0)

        return dist


if __name__ == "__main__":
    rospy.init_node("particle_filter_localization")

    particle_filter = ParticleFilter()
    particle_filter.plotter.show()

    rospy.spin()