#!/usr/bin/env python3

import rospy
import random
import time
from geometry_msgs.msg import Pose, PoseArray, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from nav_msgs.srv import GetMap
from sensor_msgs.msg import LaserScan
from math import radians, degrees
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from tf import TransformBroadcaster
from tf import TransformListener
from tf.transformations import quaternion_matrix, inverse_matrix, concatenate_matrices, quaternion_from_matrix
import tf2_geometry_msgs
import tf2_ros
import tf_conversions
import numpy as np
import cv2 as cv
import math
from threading import Lock
import copy
import matplotlib.pyplot as plt
from visualizer import DataVisualizer
import matplotlib.animation as animation
import numpy as np

class Particle():
    def __init__(self, x, y, theta, weight):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight


class ParticleFilter():
    def __init__(self):
        self.sampling_method = rospy.get_param("~sampling_method", "uniform")
        self.num_particles = rospy.get_param("~num_particles", 100) # Number of particles
        self.num_motion_updates = rospy.get_param("~num_motion_updates", 1) # Number of motion updates before a sensor update
        self.num_scan_rays = rospy.get_param("~num_scan_rays", 6) # (Approximate) number of scan rays to evaluate
        self.num_sensing_updates = rospy.get_param("~num_sensing_updates", 2) # Number of sensing updates before resampling
        self.motion_distance_noise_stddev = rospy.get_param("~motion_distance_noise_stddev", 0.01) # Standard deviation of distance noise for motion update
        self.motion_rotation_noise_stddev = rospy.get_param("~motion_rotation_noise_stddev", math.pi / 60.) # Standard deviation of rotation noise for motion update
        self.sensing_noise_stddev = rospy.get_param("~sensing_noise_stddev", 0.5) # Standard deviation of sensing noise
        self.magnetometer_noise_stddev = 0.349066 # 20 degrees # Standard deviation of magnetometer noise

        self.particles_sequence_number = 0
        self.prev_odom_msg_ = False
        self.cur_odom_msg = False
        self.estimated_pose_ = Pose()

        self.motion_update_count_ = 0 # Number of motion updates since last sensor update
        self.sensing_update_count_ = 0
        self.estimated_pose_valid_ = False

        self.data_plotter = DataVisualizer()

        self.lock = Lock()

        # ROS publishers and subscribers
        self.particlesPub = rospy.Publisher("/particlecloud", PoseArray, queue_size=10)
        rospy.Subscriber("/odom", Odometry, self.odomCallback)
        # rospy.Subscriber("/scan", LaserScan, self.scanCallback)

        self.particles_seq = 0
        self.particles_pub_timer = rospy.Timer(rospy.Duration(0.1), self.publishParticles)

        self.estimated_pose_pub_ = rospy.Publisher('estimated_pose', PoseStamped, queue_size=1)
        self.estimated_pose_seq_ = 0
        self.estimated_pose_pub_timer_ = rospy.Timer(rospy.Duration(0.1), self.publishEstimatedPose)
        
        self.transform_broadcaster = tf2_ros.TransformBroadcaster()
        self.transform_seq = 0

        self.lock.acquire()

        self.getMap()
        self.initializeParticles()

        self.lock.release()


    def getMap(self):
        # get the static map
        rospy.loginfo("Waiting for static map service.")
        rospy.wait_for_service("static_map")
        try:
            get_map = rospy.ServiceProxy('/static_map', GetMap)
            response = get_map()
            self.map = response.map
            rospy.loginfo("Map received.")
        except rospy.ServiceException as e:
            print("Service call failed:", e)
        
        # map width and height
        self.map_width = self.map.info.width
        self.map_height = self.map.info.height


        # map origin
        self.map_origin = self.map.info.origin
        print(self.map_origin)

        # get an image of the map
        self.map_image = np.reshape(self.map.data, (self.map_height, self.map_width)).astype(np.uint8)
    
        # dilation for inflate the occupancy grid
        kernel = np.ones((3, 3), np.uint8)   # kernel for dilation
        threshold_value = 90
        _, binary_image = cv.threshold(self.map_image.astype(np.uint8), threshold_value, 255, cv.THRESH_BINARY)     # thresholding
        binary_image = np.flip(binary_image, axis=0)    # flip the image since it has a flipped x axis in the RViz map

        self.dilated_map_image = cv.dilate(binary_image, kernel, iterations=1)   # dilate
        
        # define the map boundaries
        self.map_x_min = self.map_origin.position.x + 10 #self.map.info.origin.position.x
        self.map_x_max = self.map_x_min + 5
        self.map_y_min = self.map_origin.position.y + 10 #self.map.info.origin.position.y
        self.map_y_max = self.map_y_min + 5

        # Preprocess the distance transform for fast ray casting
        # self.map_image_distance_transform_ = distance_transform(self.map_image)
    

    def initializeParticles(self):
        # self.particles_array = np.zeros((self.num_particles))
        particles = []

        for p in range(self.num_particles):
            while True:
                particle = self.getParticle()
                if self.isValidParticle(particle):
                    particles.append(particle)
                    break
        self.particles_array = particles #np.array(particles)

    def getParticle(self):
        if self.sampling_method == "uniform":
            x = self.uniformSampling(self.map_x_min, self.map_x_max)
            y = self.uniformSampling(self.map_y_min, self.map_y_max)
            theta = self.uniformSampling(-math.pi, math.pi)
            
            particle = Particle(x, y, theta, 1.0)

        return particle
    
    def isValidParticle(self, particle):
        # get the x and y pixel values for x and y distance vakues
        p_x = self.map_origin.position.x - particle.x
        p_y = self.map_origin.position.y - particle.y
        x_pixel = int(abs(p_x) / self.map.info.resolution)
        y_pixel = self.map_height - int(abs(p_y) / self.map.info.resolution)
        
        if self.dilated_map_image[y_pixel][x_pixel] != 255:
            return True
        return False


    def odomCallback(self, odom_msg):
        # Skip the first call since we are looking for movements
        if not self.prev_odom_msg_:
            self.prev_odom_msg_ = odom_msg
            return

        # # Distance moved since the previous odometry message
        global_delta_x = odom_msg.pose.pose.position.x - self.prev_odom_msg_.pose.pose.position.x
        global_delta_y = odom_msg.pose.pose.position.y - self.prev_odom_msg_.pose.pose.position.y

        distance = math.sqrt(math.pow(global_delta_x, 2.) + math.pow(global_delta_y, 2.))

        # Previous robot orientation
        prev_theta = 2 * math.acos(self.prev_odom_msg_.pose.pose.orientation.w)

        if self.prev_odom_msg_.pose.pose.orientation.z < 0.:
            prev_theta = -prev_theta

        # Figure out if the direction is backward
        if (prev_theta < 0. and global_delta_y > 0.) or (prev_theta > 0. and global_delta_y < 0.):
            distance = -distance

        # Current orientation
        theta = 2 * math.acos(odom_msg.pose.pose.orientation.w)

        if odom_msg.pose.pose.orientation.z < 0.:
            theta = -theta

        # Rotation since the previous odometry message
        rotation = theta - prev_theta



        # delta_x = odom_msg.pose.pose.position.x - self.prev_odom_msg_.pose.pose.position.x
        # delta_y = odom_msg.pose.pose.position.y - self.prev_odom_msg_.pose.pose.position.y

        # distance = (delta_x**2 + delta_y**2)**0.5
        # if 

        # pre_quaternion = [0, 0, self.prev_odom_msg_.pose.pose.orientation.z, self.prev_odom_msg_.pose.pose.orientation.w]
        # cur_quaternion = [0, 0, odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w]
        
        # pre_rotation = euler_from_quaternion(pre_quaternion)[2]
        # cur_rotation = euler_from_quaternion(cur_quaternion)[2]

        # rotation = cur_rotation - pre_rotation
        # if rotation > math.pi:
        #     rotation = 

        # print(distance_, distance, rotation_, rotation)

        # Return if the robot hasn't moved
        if abs(distance) < 0.01 and abs(rotation) < 0.05:
            # print("removing: ", distance, rotation)
            return

        self.lock.acquire()

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
        self.prev_odom_msg_ = odom_msg

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
        estimated_pose_theta = 2. * math.acos(self.estimated_pose_.orientation.w)

        if self.estimated_pose_.orientation.z < 0.:
            estimated_pose_theta = -estimated_pose_theta

        self.estimated_pose_.position.x += math.cos(estimated_pose_theta) * distance
        self.estimated_pose_.position.y += math.sin(estimated_pose_theta) * distance

        estimated_pose_theta = self.wrap_angle(estimated_pose_theta + rotation)

        self.estimated_pose_.orientation.w = math.cos(estimated_pose_theta / 2.)
        self.estimated_pose_.orientation.z = math.sin(estimated_pose_theta / 2.)

        # Increment the motion update counter
        self.motion_update_count_ = self.motion_update_count_ + 1

        self.lock.release()


    def publishParticles(self, event):
        self.lock.acquire()

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

        self.lock.release()

    def publishEstimatedPose(self, event):        
        # if not self.estimated_pose_valid_:
        #     return

        self.lock.acquire()

        # Publish the estimated pose
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.seq = self.estimated_pose_seq_
        self.estimated_pose_seq_ = self.estimated_pose_seq_ + 1

        pose_stamped.pose = self.estimated_pose_

        self.estimated_pose_pub_.publish(pose_stamped)

        transform = TransformStamped()

        transform.header.frame_id = "map"
        transform.header.stamp = rospy.Time.now()
        transform.header.seq = self.transform_seq
        self.transform_seq = self.transform_seq + 1

        transform.child_frame_id = "base_footprint"

        transform.transform.translation.x = self.estimated_pose_.position.x
        transform.transform.translation.y = self.estimated_pose_.position.y

        transform.transform.rotation.w = self.estimated_pose_.orientation.w
        transform.transform.rotation.z = self.estimated_pose_.orientation.z

        # print(transform)
        
        listener = TransformListener()
        while not rospy.is_shutdown():
            try:
                # print("getting odom")        
                listener.waitForTransform("/odom", "/base_footprint", rospy.Time(), rospy.Duration(10.0))
                (odom_to_base_trans,odom_to_base_rot) = listener.lookupTransform('/odom', '/base_footprint',  rospy.Time())
                
                break
            except Exception as e:
                continue
        
        map_to_base_trans = [self.estimated_pose_.position.x, self.estimated_pose_.position.y, self.estimated_pose_.position.z]
        map_to_base_rot = [self.estimated_pose_.orientation.x, self.estimated_pose_.orientation.y, self.estimated_pose_.orientation.z, self.estimated_pose_.orientation.w]

        map_to_base_matrix = quaternion_matrix(map_to_base_rot)
        map_to_base_matrix[:3, 3] = map_to_base_trans

        odom_to_base_matrix = quaternion_matrix(odom_to_base_rot)
        odom_to_base_matrix[:3, 3] = odom_to_base_trans

        odom_to_base_inverse = inverse_matrix(odom_to_base_matrix)

        map_to_odom_matrix = concatenate_matrices(map_to_base_matrix, odom_to_base_inverse)

        map_to_odom_rot = quaternion_from_matrix(map_to_odom_matrix)
        map_to_odom_trans = map_to_odom_matrix[:3, 3]

        map_to_odom_transform = TransformStamped()
        map_to_odom_transform.header.stamp = rospy.Time.now()
        map_to_odom_transform.header.frame_id = "map"
        map_to_odom_transform.child_frame_id = "odom"
        map_to_odom_transform.header.seq = self.transform_seq
        self.transform_seq = self.transform_seq + 1

        map_to_odom_transform.transform.translation.x = map_to_odom_trans[0]
        map_to_odom_transform.transform.translation.y = map_to_odom_trans[1]
        map_to_odom_transform.transform.translation.z = map_to_odom_trans[2]
        map_to_odom_transform.transform.rotation.x = map_to_odom_rot[0]
        map_to_odom_transform.transform.rotation.y = map_to_odom_rot[1]
        map_to_odom_transform.transform.rotation.z = map_to_odom_rot[2]
        map_to_odom_transform.transform.rotation.w = map_to_odom_rot[3]

        self.transform_broadcaster.sendTransform(map_to_odom_transform)

        self.lock.release()


    def normalizeWeights(self):
        sum_of_weights = sum([p.weight for p in self.particles_array])
        # print("sum of weights: ", sum_of_weights)

        if sum_of_weights == 0:
            return 
        
        weights = []
        x = []
        for p in self.particles_array:
            p.weight = p.weight/sum_of_weights
            weights.append(p.weight)
            x.append(self.particles_array.index(p))
        # print("maximum weight: ", max(weights))``


    def uniformSampling(self, a, b):
        return random.uniform(min(a, b), max(a, b))

    def normalSampling(self, mean, std):
        return np.random.normal(mean, std)

    def wrap_angle(self, angle):
        # Function to wrap an angle between 0 and 2*Pi
        while angle < 0.0:
            angle = angle + 2 * math.pi

        while angle > 2 * math.pi:
            angle = angle - 2 * math.pi

        return angle

    # Resampling
    def resample_particles(self):

        # self.lock.acquire()

        # Copy old particles
        old_particles = copy.deepcopy(self.particles_array)
        particles = []

        # Iterator for old_particles
        old_particles_i = 0
        print("resampling started.")

        # Find a new set of particles by randomly stepping through the old set, biased by their probabilities
        while len(particles) < self.num_particles:

            # select the resampling method

            # multinomial
            value = self.uniformSampling(0.0, 1.0)

            # stratified 
            # value = self.uniformSampling(len(particles)/self.num_particles, (len(particles)+1)/self.num_particles)
            
            # systematic
            # if len(particles) == 0:
            #     value = self.uniformSampling(0, 1/self.num_particles)
            # else:
            #     value += len(particles)/self.num_particles
            

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
        
        # resampled particle array
        self.particles_array = particles

        print("resampling is over.")

        # Normalise the new particles
        self.normalizeWeights()

        # Don't use the estimated pose just after resampling
        self.estimated_pose_valid = False

        # Induce a sensing update
        self.motion_update_count = self.num_motion_updates  

        # self.lock.release()     


if __name__ == "__main__":
    rospy.init_node("particle_filter_localization")

    particle_filter = ParticleFilter()

    rospy.spin()