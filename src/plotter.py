#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import LaserScan
import matplotlib.pyplot as plt
import numpy as np

class Plotter():
    def __init__(self):
        # rospy.Subscriber("/scan", LaserScan, self.scanCallback)

        self.x = []
        self.y = []
        self.fig, self.ax = plt.subplots()
        # self.line, = self.ax.plot([], [])
        self.stemlines = self.ax.stem([0], [0], 'r', markerfmt='ro', use_line_collection=True)  # Red stems

    def update_plot(self):
        # self.line.set_xdata(self.x)
        # self.line.set_ydata(self.y)
        # print(self.stemlines)
        # self.stemlines[0].remove()
        self.stemlines = self.ax.stem(self.x, self.y, 'r', markerfmt='ro', use_line_collection=True)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
    
    # def scanCallback(self, data):
    #     self.y = data.ranges
    #     self.x = [(data.angle_min + i * data.angle_increment) for i in range(len(data.ranges))]
    #     self.update_plot()

    def show(self):
        plt.xlabel('Distance')
        plt.ylabel('Normalized weights')
        plt.title('Weight distribution')
        plt.grid(True)
        plt.show()
    
    def set_data(self, x, y):
        self.x = x
        self.y = y
        self.update_plot()

if __name__ == "__main__":
    rospy.init_node("scan_test")
    lidar = Plotter()

    lidar.show()

    rospy.spin()