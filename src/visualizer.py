# from sensor_msgs.msg import YourMessageType  # Replace with your message type
import matplotlib.pyplot as plt
from matplotlib import animation


class DataVisualizer:
    def __init__(self):
        self.data_x = []
        self.data_y = []
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [])  # Initialize empty line object
        self.ax.set_xlabel("X-axis Label")  # Set labels (replace with actual labels)
        self.ax.set_ylabel("Y-axis Label")

    def update_data(self, x, y):
        # Extract data from message (replace with your message fields)
        self.data_x = x
        self.data_y = y

    def animate(self, frame_num):
        self.line.set_data(self.data_x[:frame_num], self.data_y[:frame_num])  # Update data to plot
        self.ax.set_xlim(0, len(self.data_x) if len(self.data_x) > 0 else 1)  # Set dynamic x-axis limits
        return self.line,

# # Create an instance of the visualizer
# visualizer = DataVisualizer()
