# stereo

Here, we attempt to create the dense disparity map of pairs of stereo images in order to observe the differences between them. To accomplish this, we use the corner and normalized cross correlation (NCC) algorithms we investigated in previous labs and apply them to the provided image sets in order to ﬁnd points of interest. By showing the correspondences between the images, we can obtain an idea of how each one changes. By estimating the Fundamental Matrix, we are able to quantify the difference from the left image to the right image in the pairs. Random sample consensus (RANSAC) is used in order to remove outliers so that we are able to work with more conﬁdently chosen correspondence points. Finally, we use this information to create a dense disparity map in both the horizontal and vertical directions.

Here's an inlier feature mapping (after RANSAC) and the resulting calculated horizontal disparity map:
![](ransac.png?raw=true "Title")
![](horiz_disparity.png?raw=true "Title")
