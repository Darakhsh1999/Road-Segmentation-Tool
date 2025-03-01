# Road Segmentation Tool (RST)

## Tool to create binary labels of FPV road/street videos

Create binary segmentation maps from closed polygons using linear, quadratic and cubic splines to segment the road surface from the background. The main window runs an event loop that parses mouse callbacks and key-presses to perform different actions. The back-end continually loads in video frames from a local video and writes the labeled binary maps to a video file

---

There are 3 modes; Insert, Edit and Visual similar to VIM workflow. The different modes toggle the mouse callback mappings. In insert mode, the user can add points either from the last point, highlighted by a black circle, or insert in the mid point between two existing points. In edit mode, the user can move, change spline type and delete points. Visual mode prints out coordinates and point type to the console. The config module specifies a container class with settings and parameters for the labeling. The road labeling is done for every N video frames and the binary map for the skipped frames are interpolated.

---

## **Example**

![gif](https://i.imgur.com/YTKJCPz.gif)

![image1](https://i.imgur.com/buGHTAG.png)

![image2](https://i.imgur.com/MSgpsXX.png)

![image3](https://i.imgur.com/ALxZfjm.png)

## **Data sample output**

Some examples of the annotated binary mask is shown below

![Imagee1](https://github.com/user-attachments/assets/3de4e8bf-15ab-492d-8a5b-d52e0d2c979e)

![Imagee2](https://github.com/user-attachments/assets/907afe8a-ee7d-4cb1-a819-4d669ce762ab)

