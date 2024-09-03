# Defining Clustering and Tracking Percolation

The modules in this code accept segmented SEM/optical images of white particles in black background to provide particle stats as well as cluster stats.
Additional modules follow connected paths between particles and offer plotting convinience.

Here's a summary of what each section of the Python code does:

## 1.Regionprops_Irregular_Particle_Analyzer: (Not very effective use module 2 for sizing and particle identification ) ##

    - Purpose: Analyzes particles in a segmented image using skimage.
    
    - Inputs:
    
        segmentedimage: The segmented image to analyze.
        
        choice2: User choice to display plots or not.
        
    - Outputs:
    
        * Displays plots of particle centroids, orientations, and bounding boxes if choice2 indicates to show plots.
        
        * Returns a dataframe with particle statistics including area, centroid, orientation, and axis lengths.

## 2.Particle_Finder_and_Sizer_with_Statistics: ##

    - Purpose: Identifies particles using OpenCV contours and computes various statistics.
    
    - Inputs:
    
        * img_gray: The grayscale segmented image.
        
        * imagename: Name of the image for labeling results.
        
    - Outputs:
    
        Returns:
        
        * Contours of identified particles.
        
        * A dataframe with particle statistics such as centroid, area, perimeter, bounding circle, and rectangle details.
        
        * Bounding boxes for each particle.
Here is an example image of segmented image supplied that is used to identify particles (red outlines):

![image](https://github.com/user-attachments/assets/036b0015-5271-4d16-a239-41b27772ac94)

## 3.Cluster_Finder_with_Statistics: (Contact Network Clustering) ##

    - Purpose: Finds and analyzes clusters of particles based on their contact points.
    
    - Inputs:
    
        *contours: Contours of particles. (from previous module)
        
        *img_gray: The grayscale segmented image.
        
        *address: Path to save images.
        
        *imagename: Name of the image for labeling results.
        
        *choice2: Whether to save images.
        
        *choice3: Whether to display images.
        
    - Outputs:
    
        Returns:
        
        *A dataframe with cluster statistics including bounding rectangle dimensions, number of particles in each cluster, and indices of particles in clusters.
        
        *A dataframe of contact points between particles.
        
        *Optionally saves and/or displays images with cluster outlines.

Here is a sequence of representative images showcasing contact identification at particle scale and clustering based on friend-of-friends algorithm:

![image](https://github.com/user-attachments/assets/a983710a-ccd4-4425-bfb8-e3ea1d37489c)

Some more images showcasing function's capabilities:

![image](https://github.com/user-attachments/assets/06a234e3-027e-49b2-88d6-d568e2646cc4)

Typical data obtained plotted:

![image](https://github.com/user-attachments/assets/736870eb-7af5-4065-aa21-ad9857ae11ca)

The effectiveness of current approach (using Friend-of-Friend alogrithm) in obtains stats such as average cluster size is shown as compared to sizes obtained from radial density correlation:

![image](https://github.com/user-attachments/assets/cba0454e-a516-4fc7-a7f2-5e7896b83197)


## 4.Connection_Paths_Finder: ##

    - Purpose: Finds overlapping or connection pathways between particles.
    
    - Inputs:
    
        * img_gray: The grayscale segmented image.
        
        * contours: Contours of particles.
        
        * thresholddistance: Distance threshold for dilation to find overlapping regions.
        
    - Outputs:
    
        Returns an overlap/connection pathway mask where particles are connected based on the threshold distance.

Here are some connection paths identified and dilated to generate masks to capture emissions at contact points/interfaces:
![image](https://github.com/user-attachments/assets/b5bb1b28-7954-4793-a1ca-ce789e5a0dd1)

Use case example:
![image](https://github.com/user-attachments/assets/0f0698d4-f509-4dc8-bd90-776232cf8ba7)


## 5. Particle_Plotter_Image_Saver: ##

    - Purpose: Plots and saves images with identified particles and bounding boxes.
    
    - Inputs:
    
        * img_gray: The grayscale segmented image.
        
        * contours: Contours of particles.
        
        * boundingboxes: Bounding boxes for particles.
        
        * address: Path to save images.
        
        * choice2: Whether to save images.
        
        * choice3: Whether to display images.
        
    - Outputs:
    
        Optionally saves and/or displays an image with particle contours and bounding boxes.
    
Overall, these functions work together to analyze, visualize, and save data about particles in segmented images, including their statistics, clusters, and connection pathways.
