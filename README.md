## Writeup P4 CarND-Advanced-Lane-Lines 

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_1.png "Undistorted"
[image2]: ./examples/undistort_2.png "Undistorted"
[image3]: ./examples/binary_1.png "Binary Example"
[image4]: ./examples/warped_1.png "Warp Example"
[imageP1]: ./examples/poly_3.png "Polynomial fitting Example"
[imageP2]: ./examples/poly_2.png "Polynomial fitting Example"
[imageP3]: ./examples/poly_1.png "Polynomial fitting Example"
[image5]: ./examples/combined_1.png "Combined image"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

### Navigating this directory
* Project pipelines are in `p4_video_pipeline.py` and 'p4_pipeline.py'.
* Helper functions are in `utils.py`.
* The images for camera calibration are stored in the folder called `camera_cal`.  
* The images in `test_images` are for testing your pipeline on single frames.

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Computing the camera matrix and distortion coefficients

The code for this step is contained in the function 'camera_setup' in 'utils.py'. It is the first function called in the pipeline. 

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world (points I want to map the chessboard corners to in the undistorted image). Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time all chessboard corners in a test image are successfully detected.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration.  Using the camera matrix and the distortion coefficients from the previous function, we use undistort_image_examples to finally undistort a test image and obtained this result: 

![][image1]
![][image2]

### Pipeline (single images)
Sources consulted: 
* https://github.com/thomasantony/CarND-P04-Advanced-Lane-Lines
* https://github.com/pkern90/CarND-advancedLaneLines
* https://github.com/ksakmann/CarND-Advanced-Lane-Lines
* https://github.com/jessicayung/self-driving-car-nd/tree/master/p4-advanced-lane-lines


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

Next we use the 'binary_transform_v2' function to generate a binary image.

First, a Sobel threshold is applied in x to accentuate vertical lines.
Then, we threshold the Saturation (S) channel in the HLS colour model.
Both binaries are then stacked together to generate a single binary image.
The thresholds were determined via trial and error.

Here's an example of my output for this step.

![][image3]

#### 3. Perspective transform

The code for my perspective transform includes a function called `warp`, which appears in lines 165 through 175 in the file `utils.py`.  The `warp` function only takes the undistorted image as an input.  The source and destination points are hard coded in the following manner:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 120, 720      | 200, 720      | 
| 550, 470      | 200, 0        |
| 700, 470      | 1080, 0       |
| 1160, 720     | 1080, 720     |

The persp_trans and persp_trans_inv matrices are obtained using 'cv2.getPerspective`. The image is then warped using 'cv2.warpPerspective' and the persp_trans matrix.

It was verified that the perspective transform was working as expected by verifying that the lines appear parallel in the warped image.

![][image4]

#### 4. Identify lane-line pixels

The image is divided into 'n' horizontal steps of equal height.
For each step, a count of all the pixels at each x-value within the step window is generated using a histogram using `np.sum`
The histogram is smoothened using `scipy.signal.medfilt`.
The peaks in the left and right halves of the histogram are found using `signal.find_peaks_swt`.
The pixels in the particular horizontal strip having x coordinates close to the two peak x coordinates are finally used further.

#### 5. Fit positions of lane-line pixels with a polynomial
A second order polynomial is fit to each lane line using `np.polyfit`.

##### Example plot
Polynomial fitted to birds-eye-view image:

![][imageP1]

Polynomial drawn on image using function `draw_poly`:

![][imageP2]

Lane line area highlighted using function `highlight_lane_line_area`:

![][imageP3]

#### 6. Calculate the radius of curvature of the lane and the position of the vehicle with respect to the center

I did this in lines 71 through 74 in my code in `p4_pipeline.py`

#### 7. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 59 through 68 in my code in `p4_pipeline.py` using the function `highlight_lane_line_area` and `plot_polynomial`.  Here is an example of my result on a test image:

![][image5]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./videos/project_video_out.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

* Detection of lane lines not good with light (not dark) road surface: 
Solution: 
	* Simplify the 'binary_transform' function by removing L channel threshold and also changing the thresholds for the Sobel threshold in x.
	*  use function 'plausible_curavature' to check if radius of curvature < 500 pixels
	* check if lane lines drawn are similar to the previous set of already accepted lane lines drawn.

* No lane line detected (usually right lane line)
Solution: Relax x gradient and S channel thresholds using a `while` loop that relaxes the thresholds by a tiny amount and then repeats the detection process if no lane line is detected. This allows to relax the thresholds when no lane line is detected without adding noise to frames where lane lines were detected directly.