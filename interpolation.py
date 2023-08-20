import cv2
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

# Setup experiment frames 
circle1 = Image.new(mode="I", size=(100,100))
circle2 = Image.new(mode="I", size=(100,100))
draw_obj1 = ImageDraw.Draw(circle1)
draw_obj2 = ImageDraw.Draw(circle2)
r1 = 20
r2 = 35
x1 = 10
y1 = 20
x2 = 60
y2 = 60
draw_obj1.ellipse([(x1,y1),(x1+r1,y1+r1)], fill=255)
draw_obj2.ellipse([(x2,y2),(x2+r2,y2+r2)], fill=255)
circle1_np = np.array(circle1).astype(np.uint8)
circle2_np = np.array(circle2).astype(np.uint8)
road_frame1 = Image.open("frame1.png")
road_frame2 = Image.open("frame2.png")
road_frame1_np = np.array(road_frame1)[:,:,0].astype(np.uint8)
road_frame2_np = np.array(road_frame2)[:,:,0].astype(np.uint8)

type = "road" # road/circle
# Choose image
if type == "road":
    im1 = road_frame1_np
    im2 = road_frame2_np
else:
    im1 = circle1_np
    im2 = circle2_np


## INTERPOLATION METHOD 1 
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import interpn

def bwperim(bw, n=4):
    """
    perim = bwperim(bw, n=4)
    Find the perimeter of objects in binary images.
    A pixel is part of an object perimeter if its value is one and there
    is at least one zero-valued pixel in its neighborhood.
    By default the neighborhood of a pixel is 4 nearest pixels, but
    if `n` is set to 8 the 8 nearest pixels will be considered.
    Parameters
    ----------
      bw : A black-and-white image
      n : Connectivity. Must be 4 or 8 (default: 8)
    Returns
    -------
      perim : A boolean image

    From Mahotas: http://nullege.com/codes/search/mahotas.bwperim
    """

    if n not in (4,8):
        raise ValueError('mahotas.bwperim: n must be 4 or 8')
    rows,cols = bw.shape

    # Translate image by one pixel in all directions
    north = np.zeros((rows,cols))
    south = np.zeros((rows,cols))
    west = np.zeros((rows,cols))
    east = np.zeros((rows,cols))

    north[:-1,:] = bw[1:,:]
    south[1:,:]  = bw[:-1,:]
    west[:,:-1]  = bw[:,1:]
    east[:,1:]   = bw[:,:-1]
    idx = (north == bw) & \
          (south == bw) & \
          (west  == bw) & \
          (east  == bw)
    if n == 8:
        north_east = np.zeros((rows, cols))
        north_west = np.zeros((rows, cols))
        south_east = np.zeros((rows, cols))
        south_west = np.zeros((rows, cols))
        north_east[:-1, 1:]   = bw[1:, :-1]
        north_west[:-1, :-1] = bw[1:, 1:]
        south_east[1:, 1:]     = bw[:-1, :-1]
        south_west[1:, :-1]   = bw[:-1, 1:]
        idx &= (north_east == bw) & \
               (south_east == bw) & \
               (south_west == bw) & \
               (north_west == bw)
    return ~idx * bw

def signed_bwdist(im):
    ''' Find perim and return masked image (signed/reversed) '''    
    im = -bwdist(bwperim(im))*np.logical_not(im) + bwdist(bwperim(im))*im
    return im

def bwdist(im):
    ''' Find distance map of image '''
    return distance_transform_edt(1-im)

def interpolate_v1(top, bottom, precision):
    '''
    Interpolate between two contours

    Input: top 
            [X,Y] - Image of top contour (mask)
           bottom
            [X,Y] - Image of bottom contour (mask)
           precision
             float  - % between the images to interpolate 
                Ex: num=0.5 - Interpolate the middle image between top and bottom image
    Output: out
            [X,Y] - Interpolated image at num (%) between top and bottom

    '''
    if precision>2:
        print("Error: Precision must be between 0 and 1 (float)")

    top = signed_bwdist(top)
    bottom = signed_bwdist(bottom)

    # row,cols definition
    r, c = top.shape

    # Reverse % indexing
    precision = 1+precision

    # rejoin top, bottom into a single array of shape (2, r, c)
    top_and_bottom = np.stack((top, bottom))

    # create ndgrids 
    points = (np.r_[0, 2], np.arange(r), np.arange(c))
    xi = np.rollaxis(np.mgrid[:r, :c], 0, 3).reshape((r**2, 2))
    xi = np.c_[np.full((r**2),precision), xi]

    # Interpolate for new plane
    out = interpn(points, top_and_bottom, xi)
    out = out.reshape((r, c))

    # Threshold distmap to values above 0
    out = out > 0

    return out



## INTERPOLATION METHOD 2 
from scipy.ndimage import center_of_mass
from scipy.ndimage import shift
def interpolate_v2(images,t):
    #input: 
    # images: list of arrays/frames ordered according to motion
    # t: parameter ranging from 0 to 1 corresponding to first and last frame 
    #returns: interpolated image

    #direction of movement, assumed to be approx. linear 
    a=np.array(center_of_mass(images[0]))
    b=np.array(center_of_mass(images[-1]))

    #find index of two nearest frames 
    arr=np.array([center_of_mass(images[i]) for i in range(len(images))])
    v=a+t*(b-a) #convert t into vector 
    idx1 = (np.linalg.norm((arr - v),axis=1)).argmin()
    arr[idx1]=np.array([0,0]) #this is sloppy, should be changed if relevant values are near [0,0]
    idx2 = (np.linalg.norm((arr - v),axis=1)).argmin()

    if idx1>idx2: # t is closer to the next frame than previous
        b=np.array(center_of_mass(images[idx1])) #center of mass of nearest contour
        a=np.array(center_of_mass(images[idx2])) #center of mass of second nearest contour
        tstar=np.linalg.norm(v-a)/np.linalg.norm(b-a) #define parameter ranging from 0 to 1 for interpolation between two nearest frames
        im1_shift=shift(images[idx2],(b-a)*tstar) #shift frame 1
        im2_shift=shift(images[idx1],-(b-a)*(1-tstar)) #shift frame 2
        return im1_shift+im2_shift #return average

    if idx1<idx2:
        b=np.array(center_of_mass(images[idx2]))
        a=np.array(center_of_mass(images[idx1]))
        tstar=np.linalg.norm(v-a)/np.linalg.norm(b-a)
        im1_shift=shift(images[idx2],-(b-a)*(1-tstar))
        im2_shift=shift(images[idx1],(b-a)*(tstar))
        return im1_shift+im2_shift


## INTERPOLATION METHOD 3 

from scipy.interpolate import RegularGridInterpolator
def interpolate_v3(image1, image2, n=1):

    mask1P = bwperim(image1)
    mask2P = bwperim(image2)

    mask1D = distance_transform_edt(1-mask1P)
    mask2D = distance_transform_edt(1-mask2P)
    
    mask1P_not = np.logical_not(mask1P)
    mask2P_not = np.logical_not(mask2P)

    mask1I = np.logical_and(image1, mask1P_not)
    mask2I = np.logical_and(image2, mask2P_not)

    mask1D[mask1I == 1] = -mask1D[mask1I == 1]
    mask2D[mask2I == 1] = -mask2D[mask2I == 1]
    
    H, W = image1.shape
    stackD = np.zeros((H,W,2))
    stackD[:,:,0] = mask1D
    stackD[:,:,1] = mask2D

    x, y, z = stackD.shape
    xG, yG, zG = np.arange(1,1+x), np.arange(1,1+y), np.array([0,1.0])
    xQ, yQ, zQ = np.meshgrid(np.arange(1,1+x), np.arange(1,1+y), np.linspace(0,1,n+2))
    fn = RegularGridInterpolator((xG,yG,zG), stackD)
    input_data = np.zeros((xQ.size,3)) # (n_points,n_dim)
    input_data[:,0] = xQ.flatten()
    input_data[:,1] = yQ.flatten()
    input_data[:,2] = zQ.flatten()
    VQ = fn(input_data)
    VQ = interpn((xG,yG,zG), stackD, input_data)
    VQ = VQ.reshape((H,W,n+2))

    interpolated_mask = (VQ + np.abs(VQ)) == 0
    interpolated_mask = VQ

    return interpolated_mask


def interpolate_v4(image1, image2, n=1):
    
    mask1 = signed_bwdist(image1)
    mask2 = signed_bwdist(image2)
    n_rows, n_cols = image1.shape

    V = np.zeros((n_rows,n_cols,2))
    V[:,:,0] = mask1
    V[:,:,1] = mask2

    # create ndgrids 
    xG, yG, zG = np.arange(n_rows), np.arange(n_cols), np.array([0,1+n])
    xQ, yQ, zQ = np.meshgrid(np.arange(n_rows), np.arange(n_cols), np.arange(0,n+2))
    fn = RegularGridInterpolator((xG,yG,zG), V)
    input_data = np.zeros((xQ.size,3)) # (n_points,n_dim)
    input_data[:,0] = xQ.flatten()
    input_data[:,1] = yQ.flatten()
    input_data[:,2] = zQ.flatten()
    VQ = fn(input_data)
    VQ = interpn((xG,yG,zG), V, input_data)
    VQ = VQ.reshape((n_rows,n_cols,n+2))
    #VQ = (VQ >= 0)

    return VQ

def interpolate_v5(image1,image2):

    # Center of mass
    COM1 = np.array(center_of_mass(image1))
    COM2 = np.array(center_of_mass(image2))
    COM12 = COM2-COM1 # vector pointing from COM1 to COM2

    norm8 = 2**8-1

    mid_point = (COM1+COM2)/2
    mid_point2 = COM1+0.5*(COM2-COM1) #convert t into vector 

    print(np.unique(image1))


    # Shifted images
    image1_shifted = shift(image1, np.round(0.5*COM12))
    image2_shifted = shift(image2, np.round(-0.5*COM12))

    image1_shifted = norm8*(image1_shifted > 0).astype(np.uint8)
    image2_shifted = norm8*(image2_shifted > 0).astype(np.uint8)

    # Dilate image1
    dilate_kernel = np.ones((3,3), np.uint8)
    image1_dilated = cv2.dilate(image1_shifted, dilate_kernel, iterations=3)
    image2_dilated = cv2.dilate(image2_shifted, dilate_kernel, iterations=3)

    dilated_intersect = norm8*np.logical_and(image1_dilated, image2_dilated).astype(np.uint8)

    return dilated_intersect







### TESTING INTERPOLATION METHODS


## Run method 1
#inter_im = interpolate_v1(im1,im2, 0.5)

#fig, ax = plt.subplots(ncols= 3)
#ax[0].imshow(im1, cmap="gray")
#ax[1].imshow(inter_im, cmap="gray")
#ax[2].imshow(im2, cmap="gray")
#plt.show()


## Run method 2
#inter_im = interpolate_v2([im1,im2], 0.5)
##im_inter = (255*(im_inter - im_inter.min())/(im_inter.max()-im_inter.min())).astype(np.uint8)

#fig, ax = plt.subplots(ncols=3)
#ax[0].imshow(im1, cmap="gray")
#ax[1].imshow(inter_im, cmap="gray")
#ax[2].imshow(im2, cmap="gray")
#plt.show()


### Run method 3
#n = 4
#N = n+2
#inter_im = interpolate_v3(im1, im2,n=n)

#fig, ax = plt.subplots(ncols=N)
#for i in range(N):
    #ax[i].imshow(inter_im[:,:,i], cmap="gray")
#plt.show()

### Run method 4
#n = 4
#N = n+2
#inter_im = interpolate_v4(im1, im2,n=n)

#fig, ax = plt.subplots(ncols=N)
#for i in range(N):
    #ax[i].imshow(inter_im[:,:,i], cmap="gray")
#plt.show()


# Run method 5
n=1
inter_im3 = interpolate_v5(im1, im2)
inter_im13 = interpolate_v5(im1, inter_im3)
inter_im32 = interpolate_v5(inter_im3, im2)
fig, ax = plt.subplots(ncols=5)
ax[0].imshow(im1, cmap="gray")
ax[1].imshow(inter_im13, cmap="gray")
ax[2].imshow(inter_im3, cmap="gray")
ax[3].imshow(inter_im32, cmap="gray")
ax[4].imshow(im2, cmap="gray")
plt.show()