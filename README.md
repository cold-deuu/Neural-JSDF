# Neural-JSDF for Canadarm
base code : https://github.com/epfl-lasa/Neural-JSDF

This is the implementation of JSDF for Canadarm in Python(Base Code : MATLAB, Franka).
- - -
## Dependencies
* ROS2 - Humble
* Python 3.11
  * Trimesh
  * Numpy
  * Torch
  * Pinocchio
### Install Dependencies
**Trimesh**
```
pip3 install trimesh
pip3 install pyglet==1.5.27
pip3 install rtree
```

**Pinocchio**
```
pip3 install pin
```

**Numpy**
```
pip3 install numpy
```

**Torch**
```
pip3 install torch
```
- - -
## HOW TO RUN
```
ros2 run data_sampling sampling_node
```

- - -
## To do List
- [x] Point Sampling Code
- [x] Get Random Joints Code (Using Numpy)
- [ ] MLP Regression Code
