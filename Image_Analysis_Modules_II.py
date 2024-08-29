# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 03:34:32 2022

@author: Siva Kumar Valluri
Set of Particle analyis codes that work on segmented images
"""

"""
Particle Finder and Sizer: Skimage based##############################################################################################################################

Error: Prone to mistaking cluster as as a single unit/particle if separating boundary line is thin
Plausible Fix: Watershed segmented image using opencv itself instead of imageJ/Fiji

Requires:
-segmented image 
-choice indicating 'y' or 'n' to show plots

Returns:
-Particle statistics as a dataframe: centroid location, area, bounding ellipse details

#####################################################################################################################################################################
"""
def Regionprops_Irregular_Particle_Analyzer(segmentedimage,choice2):
    from skimage.measure import label, regionprops, regionprops_table
    import math
    import cv2 
    import matplotlib as plt
    import pandas as pd
    #from skimage.transform import rotate
    
    img_gray = cv2.cvtColor(segmentedimage, cv2.COLOR_BGR2GRAY)
    label_img = label(img_gray)
    regions = regionprops(label_img)
    
    if choice2.lower() in  ["y","yes","yippee ki yay","alright","alrighty"]:
        fig, ax = plt.subplots()
        ax.imshow(img_gray, cmap=plt.cm.gray)
        
        for props in regions:
            y0, x0 = props.centroid
            orientation = props.orientation
            x1 = x0 + math.cos(orientation) * 0.5 * props.minor_axis_length
            y1 = y0 - math.sin(orientation) * 0.5 * props.minor_axis_length
            x2 = x0 - math.sin(orientation) * 0.5 * props.major_axis_length
            y2 = y0 - math.cos(orientation) * 0.5 * props.major_axis_length
        
            ax.plot((x0, x1), (y0, y1), '-r', linewidth=1)
            ax.plot((x0, x2), (y0, y2), '-r', linewidth=1)
            ax.plot(x0, y0, '.g', markersize=5)
        
            minr, minc, maxr, maxc = props.bbox
            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)
            ax.plot(bx, by, '-b', linewidth=1)
        
        plt.show()
        
    table = regionprops_table(label_img, properties=('area','centroid','orientation','major_axis_length','minor_axis_length'))   
    data=pd.DataFrame(table)
    return data



"""
Particle Finder and Sizer: Open CV based##############################################################################################################################
Requires:
-segmented image 
-name for the segmented image

Returns:
-Contours (perimeters of particles identifed by open cv2 module)
-Particle statistics as a dataframe: centroid location, area,perimeter, bounding circle and min area bounding rectangle details
-bounding box details as a tuple (for plotting)
####################################################################################################################################
"""

def Particle_Finder_and_Sizer_with_Statistics(img_gray,imagename):
    import cv2
    import numpy as np
    import pandas as pd
    
    contours, hierarchy = cv2.findContours(img_gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    Particles_Statistics_df=pd.DataFrame()
    Particles_Statistics_df = pd.DataFrame(columns = ['Image Title','centroid_x','centroid_y','Area/pixel','Perimeter/pixel','circle_center_x','circle_center_y','Radius/pixel','rect_center_x','rect_center_y','Rect_width/pixel','Rect_height/pixel'])
    boundingboxes=[]
    
    #Removing very small 'particles'
    contour_new =[]
    cc=list(contours)
    for contournumber in range(0, len(cc),1):
        M = cv2.moments(cc[contournumber])
        if (cc[contournumber].shape[0] > 4 and M['m00'] > 0): # Considering particles that are more than 4 pixels and with moment greater than zero (necessary for centroid)
            contour_new .append(cc[contournumber])

    contours=tuple(contour_new)    
    for contour_number in range(0,len(contours),1):
        cnt = contours[contour_number]
        M = cv2.moments(cnt)
        centroid_x = int(M['m10']/M['m00'])
        centroid_y = int(M['m01']/M['m00'])
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt,True)
            
        #Fitting circle
        (x,y),radius = cv2.minEnclosingCircle(cnt)
        circle_center_x = int(x)
        circle_center_y = int(y)
        center = (int(x),int(y))
        radius = int(radius)
        #cv2.circle(img,center,radius,(0,255,0),2)  #To draw the circle
            
        #Fitting least area rectangle
        #It returns a Box2D structure which contains following details - ( center (x,y), (width, height), angle of rotation ). But to draw this rectangle, we need 4 corners of the rectangle. It is obtained by the function cv.boxPoints()
        rect = cv2.minAreaRect(cnt)
        rect_center_x=rect[0][0]
        rect_center_y=rect[0][1]
        rect_width=rect[1][0]
        rect_height=rect[1][1]
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        #cv2.drawContours(img,[box],0,(0,0,255),2) #To draw the rectangle, also note: Color BGR currently red chosen
        Dataset1 = np.column_stack((imagename,centroid_x,centroid_y,area,perimeter,circle_center_x,circle_center_y,radius,rect_center_x,rect_center_y,rect_width,rect_height))    
        X1 = pd.DataFrame(Dataset1,columns = ['Image Title','centroid_x','centroid_y','Area/pixel','Perimeter/pixel','circle_center_x','circle_center_y','Radius/pixel','rect_center_x','rect_center_y','Rect_width/pixel','Rect_height/pixel'])
        Particles_Statistics_df = pd.concat([Particles_Statistics_df, X1], ignore_index=True)
        boundingboxes.append(box)
    boundingboxes=tuple(boundingboxes)
    
    return contours,Particles_Statistics_df,boundingboxes

"""
Cluster Finder with Stats: Custom contact definition##############################################################################################################################
Requires:
-contours generated by opencv
-segmented image 
-address to save images
-name for the segmented image
-choice2-Do you want to save images generated?
-choice3-Do you want to SEE images generated?

Returns:
-Cluster statistics as a dataframe: min area bounding rectangle, number of particles in cluster, Indices of particles in cluster

####################################################################################################################################
"""


def Cluster_Finder_with_Statistics(contours,img_gray,address,imagename,choice2,choice3):
    import numpy as np
    from scipy.spatial import cKDTree
    import pandas as pd
    from scipy.spatial import ConvexHull
    import cv2
    
    Cluster_Statistics_df=pd.DataFrame()
    Cluster_Statistics_df = pd.DataFrame(columns = ['Image Title','Rect fit-width/pixel','Rect fit-height/pixel','Number of particles in cluster','Indices of particles in cluster'])
    
    #Converting contour tuple into list of particle perimeters
    List_of_particle_perimeters = []
    for contour in contours:
        particle_perimeter=contour[:,0,:]
        List_of_particle_perimeters.append(particle_perimeter)
        
        
    ####################################################################################Finding contacting particle pairs#######################################################################################
    Contact_points_df = pd.DataFrame()
    Contact_points_df = pd.DataFrame(columns = ['Contact-x/pixel', 'Contact-y/pixel', 'Particleindex-1', 'Particleindex-2'])
    
    
    #Plotting just the defined 'contact points' on image
    image_test = img_gray.copy()
    image_test = cv2.cvtColor(image_test, cv2.COLOR_GRAY2BGR)
    List_of_contacting_particle_indices = []    
    for current_particle_index in range(0,len(List_of_particle_perimeters),1):
        other_particle_perimeter_indeces= list(np.arange(0,len(List_of_particle_perimeters),1))
        other_particle_perimeter_indeces.remove(current_particle_index)

        for other_particle_index in other_particle_perimeter_indeces:            
            kd_tree1 = cKDTree(List_of_particle_perimeters[current_particle_index])
            kd_tree2 = cKDTree(List_of_particle_perimeters[other_particle_index])
            number_of_contact_points=kd_tree1.count_neighbors(kd_tree2, r=4) #Second input term is radius of survey
            contact_indexes = kd_tree1.query_ball_tree(kd_tree2, r=4)
            for i in range(len(contact_indexes)):
                for j in contact_indexes[i]:
                    x = int((List_of_particle_perimeters[current_particle_index][i, 0]+List_of_particle_perimeters[other_particle_index][j, 0])/2)
                    y = int((List_of_particle_perimeters[current_particle_index][i, 1]+List_of_particle_perimeters[other_particle_index][j, 1])/2)
                    c = int(current_particle_index)
                    o = int(other_particle_index)

                    Dataset1 = np.column_stack((x,y,c,o))    
                    X1 = pd.DataFrame(Dataset1,columns = ['Contact-x/pixel', 'Contact-y/pixel', 'Particleindex-1', 'Particleindex-2'])
                    Contact_points_df = pd.concat([Contact_points_df, X1], ignore_index=True)
                    #Plotting just the defined 'contact points'
                    image_test = cv2.circle(image_test, (x,y), radius=2, color=(0, 0, 255), thickness=-1)
                    #plt.scatter(x,y,s=20) #code verification 
                    #Connecting the points
                    #plt.plot([List_of_particle_perimeters[current_particle_index][i, 0], List_of_particle_perimeters[other_particle_index][j, 0]],[List_of_particle_perimeters[current_particle_index][i, 1], List_of_particle_perimeters[other_particle_index][j, 1]], "-r")
            if number_of_contact_points>0:
                pair_indices=np.array([current_particle_index,other_particle_index])
                List_of_contacting_particle_indices.append(pair_indices)
    
    result_df = Contact_points_df.drop_duplicates(['Contact-x/pixel', 'Contact-y/pixel'],ignore_index=True)
    
    #########Finding clusters by going through pairs and ensuring 'friend-of-friends' are grouped together as one cluster#######################################################################################       
    Particle_indices_in_Cluster = []
    while len(List_of_contacting_particle_indices)>0:
        first, *rest = List_of_contacting_particle_indices
        first = set(first)
    
        lf = -1
        while len(first)>lf:
            lf = len(first)
    
            rest2 = []
            for r in rest:
                if len(first.intersection(set(r)))>0:
                    first |= set(r)
                else:
                    rest2.append(r)     
            rest = rest2
        Particle_indices_in_Cluster.append(first)
        List_of_contacting_particle_indices = rest
    
    #Converting Cluster indices tuple into list of lists again
    for cluster in range(0,len(Particle_indices_in_Cluster),1): 
        Particle_indices_in_Cluster[cluster]=list(Particle_indices_in_Cluster[cluster])   
    
    
    ###########################################################################
    boundingboxes_of_clusters = []
    Hulls_of_clusters = []
    for cluster in Particle_indices_in_Cluster:
        Circumference_Points_of_Cluster = []
        Circumference_Points_of_Cluster = np.array([0,0])
        for particle_index in range(0,len(cluster),1):            
            Circumference_Points_of_Cluster = np.vstack((Circumference_Points_of_Cluster,np.array(List_of_particle_perimeters[int(cluster[particle_index])])))
        
        Circumference_Points_of_Cluster=np.delete(Circumference_Points_of_Cluster,0,0)
        Hull=ConvexHull(Circumference_Points_of_Cluster)
        hull_points = Circumference_Points_of_Cluster[Hull.vertices]
        Hulls_of_clusters.append(hull_points)
        
        rect = cv2.minAreaRect(hull_points)
        
        #data no.2 and 3
        rect_width=rect[1][0]
        rect_height=rect[1][1]
        
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        boundingboxes_of_clusters.append(box)
        
        #data no.4
        number_of_particle_in_cluster =len(cluster)
        
        Dataset1 = np.column_stack((imagename,rect_width,rect_height,number_of_particle_in_cluster,str(cluster)))    
        X1 = pd.DataFrame(Dataset1,columns = ['Image Title','Rect fit-width/pixel','Rect fit-height/pixel','Number of particles in cluster','Indices of particles in cluster'])
        Cluster_Statistics_df = pd.concat([Cluster_Statistics_df, X1], ignore_index=True)
        
    Hulls_of_clusters=tuple(Hulls_of_clusters)
    boundingboxes_of_clusters=tuple(boundingboxes_of_clusters)
    #Plotting particles (contours) identified and their bounding boxes####################################################################
    if choice3.lower() in  ["y","yes","yippee ki yay","alright","alrighty"]:
        image_copy = img_gray.copy()
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2BGR)
        #cv2.drawContours(image=image_copy, contours=Hulls_of_clusters, contourIdx=-1, color=[0,250,0], thickness=2, lineType=cv2.LINE_AA)
        cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=[0,0,250], thickness=1, lineType=cv2.LINE_AA)
        cv2.drawContours(image=image_copy, contours=boundingboxes_of_clusters, contourIdx=-1, color=[0,250,0], thickness=2, lineType=cv2.LINE_AA)
        cv2.imwrite(str(address)+'\\'+'Clusters identified in image .tif', image_copy)
        if choice2.lower() in  ["y","yes","yippee ki yay","alright","alrighty"]:
            cv2.imshow('Clusters identified', image_copy)
            cv2.waitKey(0)
    
    return Cluster_Statistics_df, result_df


"""
Connection Pathway Finder##############################################################################################################################
Requires:
-segmented image 
-threshold distance - i.e the distance particles will be dilated to find 'overlap' regions which are the connection pathways
 between the particles 

Returns:
-Overlap/connection pathway mask with segmented image dimensions
######################################################################################################################################################
"""
def Connection_Paths_Finder(img_gray,contours,thresholddistance):
    import numpy as np
    import cv2    
    h, w = img_gray.shape[:2]
    overlap = np.zeros((h, w), dtype=np.int32)
    overlap_mask = np.zeros((h, w), dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thresholddistance, thresholddistance))
    
    label_map = np.zeros_like(img_gray)

    # For each list of contour points...
    for particle_number in range(len(contours)):
        # Create a mask image that contains the contour filled in
        cimg = np.zeros_like(img_gray)
        cv2.drawContours(cimg, contours, particle_number, color=255, thickness=-1)

        # Access the image pixels and create a 1D numpy array then add to list
        pts = np.where(cimg == 255)
        label_map[pts[0], pts[1]]=particle_number
    
    # grows the blobs by `distance` and sums to get overlaps
    for xlabel in range(1, len(contours)):
        mask = 255 * np.uint8(label_map == xlabel)
        overlap += cv2.dilate(mask, kernel, iterations=1) // 255
    overlap = np.uint8(overlap > 1)
   
    noverlaps, overlap_components = cv2.connectedComponents(overlap, connectivity=8)
    for label in range(1, noverlaps):
        mask = 255 * np.uint8(overlap_components == label)
        if np.any(cv2.bitwise_and(img_gray, mask)):
            overlap_mask = cv2.bitwise_or(overlap_mask, mask)
         
    #Removing 'particle' regions from connection paths found for the given threshold distance
    regionwithinthreshold = np.ones_like(img_gray)                    
    for row in range(0,int(img_gray.shape[0]),1):
        for column in range(0,int(img_gray.shape[1]),1):
            if img_gray[row,column] != 255 and overlap_mask[row,column] == 255:
                regionwithinthreshold[row,column]=255
            else:
                regionwithinthreshold[row,column]=0    
    
    return regionwithinthreshold

"""
Plotter and Image Saver##############################################################################################################################
Requires:
-segmented image 
-threshold distance - i.e the distance particles will be dilated to find 'overlap' regions which are the connection pathways
 between the particles 

Returns:
-Overlap/connection pathway mask with segmented image dimensions
######################################################################################################################################################
"""
def Particle_Plotter_Image_Saver(img_gray,contours,boundingboxes,address,choice2,choice3):
    import cv2
    if choice3.lower() in  ["y","yes","yippee ki yay","alright","alrighty"]:
        image_copy = img_gray.copy()
        image_copy = cv2.cvtColor(image_copy, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=[0,0,250], thickness=1, lineType=cv2.LINE_AA)
        cv2.drawContours(image=image_copy, contours=boundingboxes, contourIdx=-1, color=[120,100,0], thickness=1, lineType=cv2.LINE_AA)
        cv2.imwrite(str(address)+'\\'+'Particles identified in image .tif', image_copy)
        if choice2.lower() in  ["y","yes","yippee ki yay","alright","alrighty"]:
            cv2.imshow('Particles identified', image_copy)
            cv2.waitKey(0)

"""
##Laser Optical Microscope Particle Analyzer ###################################################################################################################
Requires:
-segmented image(s) 
-choice1-Do you want to find connection paths between particles?
-choice2-Do you want to save images generated?
-choice3-Do you want to SEE images generated?
-choice4-Do you have emission frame grabs to partition light emitted from specific regions
-counternumber- Number of images 
-address- Where to save images generated

Returns:
-Dataframes of particle and connected-component details such as area, bounding box details, etc.
-Images of connected particles, connection paths based on thresholds


Folder structure:
-Single folder contains 4/8 static images and 4/8 instance images (needs fixing)
##################################################################################################################################
"""







      
        
