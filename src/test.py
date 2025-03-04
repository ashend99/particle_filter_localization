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

class Particle():
    def __init__(self, x, y, theta, weight):
        self.x = x
        self.y = y
        self.theta = theta
        self.weight = weight

class ParticleFilter():
    def __init__(self):

        # parameters
        # self.num_particles = rospy.get_param("~num_particles", 5000)
        self.sampling_method = rospy.get_param("~sampling_method", "uniform")
        # self.motion_distance_noise_stddev = rospy.get_param("~motion_distance_noise_stddev", 0.01)
        # self.motion_rotation_noise_stddev = rospy.get_param("~motion_rotation_noise_stddev", math.pi/60)
        # self.num_motion_updates = rospy.get_param("~num_motion_updates", 5)
        # self.num_scan_rays = rospy.get_param("~num_scan_rays", 6)
        self.num_particles = rospy.get_param("~num_particles", 5000) # Number of particles
        self.num_motion_updates = rospy.get_param("~num_motion_updates", 5) # Number of motion updates before a sensor update
        self.num_scan_rays = rospy.get_param("~num_scan_rays", 6) # (Approximate) number of scan rays to evaluate
        self.num_sensing_updates = rospy.get_param("~num_sensing_updates", 5) # Number of sensing updates before resampling
        self.motion_distance_noise_stddev = rospy.get_param("~motion_distance_noise_stddev", 0.01) # Standard deviation of distance noise for motion update
        self.motion_rotation_noise_stddev = rospy.get_param("~motion_rotation_noise_stddev", math.pi / 60.) # Standard deviation of rotation noise for motion update
        self.sensing_noise_stddev = rospy.get_param("~sensing_noise_stddev", 0.5) # Standard deviation of sensing noise
        self.magnetometer_noise_stddev = 0.349066 # 20 degrees # Standard deviation of magnetometer noise


        self.particles_sequence_number = 0
        self.pre_odom_msg = False
        self.cur_odom_msg = False
        self.estimated_pose = Pose()

        self.motion_update_count = 0 # Number of motion updates since last sensor update
        self.sensing_update_count = 0 # Number of sensing updates since resampling
        self.estimated_pose_valid = False # Don't use the estimated pose just after initialisation


        self.lock = Lock()

        self.lock.acquire()

        # ROS publishers and subscribers
        self.particlesPub = rospy.Publisher("/particlecloud", PoseArray, queue_size=10)
        rospy.Subscriber("/odom", Odometry, self.odomCallback)
        rospy.Subscriber("/scan", LaserScan, self.scanCallback)

        self.particles_seq = 0
        self.particles_pub_timer = rospy.Timer(rospy.Duration(0.1), self.publishParticles)

        self.estimated_pose_pub = rospy.Publisher('estimated_pose', PoseStamped, queue_size=1)
        self.estimated_pose_seq = 0
        self.estimated_pose_pub_timer = rospy.Timer(rospy.Duration(0.1), self.publishEstimatedPose)

        self.transform_broadcaster = tf2_ros.TransformBroadcaster()
        self.transform_seq = 0


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
        self.map_image_distance_transform_ = distance_transform(self.map_image)

    
    def odomCallback(self, odom_msg):
        if not self.pre_odom_msg:
            self.pre_odom_msg = odom_msg
            return
        
        self.cur_odom_msg = odom_msg

        # Distance change
        delta_x = self.cur_odom_msg.pose.pose.position.x - self.pre_odom_msg.pose.pose.position.x
        delta_y = self.cur_odom_msg.pose.pose.position.y - self.pre_odom_msg.pose.pose.position.y

        distance = (delta_x**2 + delta_y**2)**0.5

        # Previous robot orientation
        pre_theta = 2 * math.acos(self.pre_odom_msg.pose.pose.orientation.w)

        if self.pre_odom_msg.pose.pose.orientation.z < 0.:
            pre_theta = -pre_theta

        # Figure out if the direction is backward
        if (pre_theta < 0 and delta_y > 0) or (pre_theta > 0 and delta_y < 0):
            distance = -distance

        # Current orientation
        theta = 2 * math.acos(self.cur_odom_msg.pose.pose.orientation.w)

        if odom_msg.pose.pose.orientation.z < 0.:
            theta = -theta

        # Rotation since the previous odometry message
        rotation = theta - pre_theta


        # Return if the robot hasn't moved
        if round(distance, 2) == 0 and round(rotation, 2) == 0:
            return
        
        # print(round(distance, 2), round(rotation, 2))
        self.lock.acquire()
        
        for p in self.particles_array:
            distance_noise = self.normalSampling(0, self.motion_distance_noise_stddev)
            rotation_noise = self.normalSampling(0, self.motion_rotation_noise_stddev)
            p.x += (distance + distance_noise) * math.cos(p.theta)
            p.y += (distance + distance_noise) * math.sin(p.theta)
            theta = p.theta + rotation + rotation_noise
            p.theta = self.wrap_angle(theta)
        

        self.pre_odom_msg = self.cur_odom_msg


        particles = []

        for p in self.particles_array:
            if not(p.x < self.map_x_min or p.x > self.map_x_max or p.y < self.map_y_min or p.y > self.map_y_max):
                particles.append(p)
        self.particles_array = particles

        self.normalizeWeights()

        # If the estimated pose is valid move it too
        if self.estimated_pose_valid:
            estimated_pose_theta = 2. * math.acos(self.estimated_pose.orientation.w)

            if self.estimated_pose.orientation.z < 0.:
                estimated_pose_theta = -estimated_pose_theta

            self.estimated_pose.position.x += math.cos(estimated_pose_theta) * distance
            self.estimated_pose.position.y += math.sin(estimated_pose_theta) * distance

            estimated_pose_theta = self.wrap_angle(estimated_pose_theta + rotation)

            self.estimated_pose.orientation.w = math.cos(estimated_pose_theta / 2.)
            self.estimated_pose.orientation.z = math.sin(estimated_pose_theta / 2.)


        self.motion_update_count = self.motion_update_count + 1

        self.lock.release()


    def scanCallback(self, msg):
        # # print("scan")
        if self.motion_update_count < self.num_motion_updates:
            return
        
        print("scan")
        self.lock.acquire()

        step = int(math.floor(float(len(msg.ranges))/self.num_scan_rays))

        start = time.time()

        first_particle = True
        for p in self.particles_array:
            likelihood = 1
            for i in range(0, len(msg.ranges), step):
                scan_range = msg.ranges[i]

                local_angle = msg.angle_increment * i + msg.angle_min

                global_angle = self.wrap_angle(p.theta + local_angle)

                particle_range = self.hit_scan(p.x, p.y, global_angle, 7.0, first_particle) 

                ray_likelihood = 1/math.sqrt(2*math.pi*math.pow(self.sensing_noise_stddev, 2))*math.exp(-math.pow((particle_range-scan_range),2)/(2*math.pow(self.sensing_noise_stddev,2)))
                likelihood *= ray_likelihood

            p.weight *= likelihood
            first_particle = False
        
        end = time.time()

        self.normalizeWeights()

        self.estimate_pose()

        self.sensing_update_count = self.sensing_update_count + 1

        if self.sensing_update_count > self.num_sensing_updates:
            self.resample_particles()
            self.sensing_update_count = 0


        self.motion_update_count = 0
        self.lock.release()
        # pass
        print("scan ok")

    def hit_scan(self, start_x, start_y, theta, max_range, draw=False):
        # Find the nearest obstacle from position start_x, start_y (in metres) in direction theta

        # Start point in occupancy grid coordinates
        start_point = [int(round((start_x - self.map.info.origin.position.x) / self.map.info.resolution)),
                             int(round((start_y - self.map.info.origin.position.y) / self.map.info.resolution))]

        # End point in real coordinates
        end_x = start_x + math.cos(theta) * max_range
        end_y = start_y + math.sin(theta) * max_range

        # End point in occupancy grid coordinates
        end_point = [int(round((end_x - self.map.info.origin.position.x) / self.map.info.resolution)),
                           int(round((end_y - self.map.info.origin.position.y) / self.map.info.resolution))]

        # Find the first "hit" along scan
        # (unoccupied is value 0, occupied is value 100)
        # hit = find_hit(self.map_image_, start_point, end_point, 50)
        hit = find_hit_df(self.map_image_distance_transform_, start_point, end_point)

        # Convert hit back to world coordinates
        hit_x = hit[0] * self.map.info.resolution + self.map.info.origin.position.x
        hit_y = hit[1] * self.map.info.resolution + self.map.info.origin.position.y

        # Add a debug visualisation marker
        # if draw:
        #     point = Point(start_x, start_y, 0.)
        #     self.marker_.points.append(point)
        #     point = Point(hit_x, hit_y, 0.)
        #     self.marker_.points.append(point)

        # Get distance to hit
        return math.sqrt(math.pow(start_x - hit_x, 2) + math.pow(start_y - hit_y, 2))

    
    def normalizeWeights(self):
        sum_of_weights = sum([p.weight for p in self.particles_array])

        for p in self.particles_array:
            p.weight = p.weight/sum_of_weights



    def wrap_angle(self, angle):
        # Function to wrap an angle between 0 and 2*Pi
        while angle < 0.0:
            angle = angle + 2 * math.pi

        while angle > 2 * math.pi:
            angle = angle - 2 * math.pi

        return angle

    def initializeParticles(self):
        # print("initialize")
        particles_array = []

        for p in range(self.num_particles):
            particle = self.getParticles()
            particles_array.append(particle)
        
        # removing particles in the occupied space
        valid_particles = []
        c = 0
        for particle in particles_array:

            # check whether the particle in the occupied space
            if self.isValidParticle(particle):
                valid_particles.append(particle)
                c += 1
        
        self.particles_array = valid_particles
        # print("publish")
        # self.publishParticles()

    def publishParticles(self, event):
        # self.lock.acquire()

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

        # rospy.loginfo(f"{c} Particles are initialized.")
        # self.lock.release()
    
    def publishEstimatedPose(self, event):        
        # if not self.estimated_pose_valid:
        #     return

        # self.lock_.acquire()

        # print("publishing estimated pose")

        # Publish the estimated pose
        pose_stamped = PoseStamped()
        pose_stamped.header.frame_id = "map"
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.seq = self.estimated_pose_seq
        self.estimated_pose_seq = self.estimated_pose_seq + 1

        pose_stamped.pose = self.estimated_pose

        self.estimated_pose_pub.publish(pose_stamped)


        transform = TransformStamped()

        transform.header.frame_id = "map"
        transform.header.stamp = rospy.Time.now()
        transform.header.seq = self.transform_seq
        self.transform_seq = self.transform_seq + 1

        transform.child_frame_id = "base_footprint"

        transform.transform.translation.x = self.estimated_pose.position.x
        transform.transform.translation.y = self.estimated_pose.position.y

        transform.transform.rotation.w = self.estimated_pose.orientation.w
        transform.transform.rotation.z = self.estimated_pose.orientation.z

        # print(transform)
        
        listener = TransformListener()
        while not rospy.is_shutdown():
            try:
                print("getting odom")        
                listener.waitForTransform("/odom", "/base_footprint", rospy.Time(), rospy.Duration(10.0))
                (odom_to_base_trans,odom_to_base_rot) = listener.lookupTransform('/odom', '/base_footprint',  rospy.Time())
                
                break
            except Exception as e:
                continue
        
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



    def getParticles(self):
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

    def estimate_pose(self):
        # Position of the estimated pose
        estimated_pose_x = 0.0
        estimated_pose_y = 0.0
        estimated_pose_theta = 0.0

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

        self.estimated_pose.orientation.w = math.cos(estimated_pose_theta / 2.)
        self.estimated_pose.orientation.z = math.sin(estimated_pose_theta / 2.)

        self.estimated_pose_valid = True

    def resample_particles(self):
        # Resample the particles
        # Weights are expected to be normalised

        print("resampling")

        # Copy old particles
        old_particles =  self.particles_array.copy()  #  copy.deepcopy(self.particles_)
        self.particles_array = []

        # Iterator for old_particles
        old_particles_i = 0

        # Find a new set of particles by randomly stepping through the old set, biased by their probabilities
        while len(self.particles_array) < self.num_particles:
            value = self.uniformSampling(0.0, 1.0)
            sum = 0.0

            # Loop until a particle is found
            particle_found = False
            while not particle_found:

                # If the random value is between the sum and the sum + the weight of the particle
                if value > sum and value < (sum + old_particles[old_particles_i].weight):

                    # Add the particle to the "particles_" vector
                    self.particles_array.append(old_particles[old_particles_i])

                    # Add jitter to the particle
                    self.particles_array[-1].x = self.particles_array[-1].x + self.normalSampling(0, 0.02)
                    self.particles_array[-1].y = self.particles_array[-1].y + self.normalSampling(0, 0.02)
                    self.particles_array[-1].theta = self.wrap_angle(self.particles_[-1].theta + self.normalSampling(0, math.pi / 30.))

                    # The particle may be out of the map, but that will be fixed by the motion update
                    
                    # Break out of the loop
                    particle_found = True

                # Add particle weight to sum and increment the iterator
                sum = sum + old_particles[old_particles_i].weight
                old_particles_i = old_particles_i + 1

                # If the iterator is past the vector, loop back to the beginning
                if old_particles_i >= len(old_particles):
                    old_particles_i = 0

        # Normalise the new particles
        self.normalizeWeights()

        # Don't use the estimated pose just after resampling
        self.estimated_pose_valid = False

        # Induce a sensing update
        self.motion_update_count = self.num_motion_updates      



    def uniformSampling(self, a, b):
        return random.uniform(min(a, b), max(a, b))

    def normalSampling(self, mean, std):
        return np.random.normal(mean, std)
    
def find_hit_df(img, p1, p2):
    # Draws a line from p1 to p2
    # Stops at the first pixel that is a "hit", i.e. above the threshold
    # Returns the pixel coordinates for the first hit
    #
    # similar to find_hit but uses distance transform image to speed things up

    # Extract the vector
    x1 = float(p1[0])
    y1 = float(p1[1])
    x2 = float(p2[0])
    y2 = float(p2[1])

    dx = x2 - x1
    dy = y2 - y1
    l = math.sqrt(dx**2. + dy**2.)
    dx = dx / l
    dy = dy / l

    step = 1.0 # pixels
    min_step = 2.5 # pixels -- too large risks jumping over obstacles

    max_steps = int(l)

    dist = 0

    # print("find hit")
    while dist < max_steps:

        # Get the next pixel
        x = int(round(x1 + dx*dist))
        y = int(round(y1 + dy*dist))

        # Check if it's outside of the image
        if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
            return [x, y] #p2

        current_distance = img[y, x] 

        # Check for "hit"
        if current_distance <= 0:
            return [x, y]

        # Otherwise, adjust the step size according to the distance transform
        step = current_distance-1.
        if step < min_step:
            step = min_step

        # Move along the ray
        dist += step

    # No hits found
    return p2


def distance_transform(image):

    
	image_reformat = np.float32(image)
	#print('reformat', image_reformat)

	# Threshold the image to convert a binary image
	ret, thresh = cv.threshold(image_reformat, 50, 1, cv.THRESH_BINARY_INV)

	#print('thresh',thresh)

	# Determine the distance transform.
	dist = cv.distanceTransform(np.uint8(thresh), cv.DIST_L2, 0)

	#print('dist',dist[10:20,10:20])
	  
	# Make the distance transform normal.
	#dist_output = cv.normalize(dist, None, 0, 1.0, cv.NORM_MINMAX)

	#print('dist',dist)
	  
	# Display the distance transform
	#cv.imshow('Distance Transform', np.uint8(dist))
	#cv.waitKey(0)

	return dist



if __name__ == "__main__":
    rospy.init_node("particle_filter_localization")

    particle_filter = ParticleFilter()

    rospy.spin()
    # while not rospy.is_shutdown():
    #     rospy.sleep(0.1)
    #     # particle_filter.particlesPub.publish(particle_filter.pose_array)
    #     particle_filter.publishParticles()
    #     # print("here")