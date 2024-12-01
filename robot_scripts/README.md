# README
To start, the Interbotix vx300 workspace needs to be cloned and installed. Use the following bash command to get started:

```bash
git clone https://github.com/Interbotix/interbotix_ros_core.git
cd interbotix_ros_core
rosdep install --from-paths src --ignore-src -r -y
catkin_make
source devel/setup.bash
```

This folder includes all the source code developed for the robotic arm. The contents are as follows:

- **Pick and Place Coordinates**: Scripts for defining the coordinates for picking and placing items.
- **End Effector Control**: Code for controlling the end effector of the robotic arm.
- **Publisher of Coordinates**: Scripts to publish coordinates for measurement during pick and place operations.
- **Server Script**: A server script to create a point-to-point (P2P) connection with the sensor PC to receive labels from the Machine Learning model.

Make sure to review each script for detailed implementation and usage instructions. 