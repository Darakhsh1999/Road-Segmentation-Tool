import cv2
import copy
import numpy as np
from point import Point
from config import Config
from utils import distance_utils, curve_utils


class SegTool():

    def __init__(self, source_path, config):
        
        # Config settings
        self.config: Config = config

        # Video source
        self.source = cv2.VideoCapture(source_path)
        _, self.frame = self.source.read() # (H,W,C), np-uint8
        self.frame_shape = self.frame.shape[:2] # (H,W)

        # Video writer
        encoder = cv2.VideoWriter_fourcc("M","J","P","G")
        flipped_frame_shape = (self.frame_shape[1], self.frame_shape[0]) # (W,H)
        self.video_source = cv2.VideoWriter(f"{self.config.video_name}.avi", encoder, self.config.FPS, flipped_frame_shape) # (W,H)

        # Application state variables
        self.frame0 = self.frame.copy() # raw frame from source
        self.stop = False # Stop program
        self.moving_circle = False # Drag and drop points
        self.path: list[Point] = []
        self.last_path = None
        self.highlight: bool = False # Highlight last point
        self.show_annotation: bool = True # Toggle annotation
        
        # Initialize correct mode
        self.update_mode("Visual")

        if self.config.debug > 0:
            print(f"(H,W) = {self.source.get(4)},{self.source.get(3)}")
            print(f"FPS = {self.source.get(5)}")
            print(f"Frame count = {self.source.get(7)} frames")


    def start_window(self):
        """ Initializes window and run the main event loop """
        
        # Main window configuration 
        cv2.namedWindow("root", self.config.window_mode) 
        cv2.setMouseCallback("root", self.mouse_callback)
        
        # Event loop
        while (True):
            
            # Render image
            cv2.imshow("root", self.frame)

            # Event listener
            result = cv2.waitKey(1)
            if result == -1: continue

            # Parse key
            self.parse_key(result)

            # Check break condition
            if self.stop: break
        
        exit(0)


    def parse_key(self, result):
        """ Key press listener """
        
        if result == ord("f"):
            self.show_annotation = True
            self.forward()
        elif result == ord("v"):
            self.render("Visual")
        elif result == ord("e"):
            self.render("Edit")
        elif result == ord("i"):
            self.render("Insert")
        elif result == ord("t"): # Toggle annotation
            self.show_annotation = not self.show_annotation
            self.render(self.mode)
        elif result == ord("r"): # Reset 
            self.path = []
            self.highlight = False
            self.render(self.mode)
        elif result == ord("z"): # Undo 
            self.path = copy.deepcopy(self.last_path)
            self.highlight = True # incase, we pressed 'r' in this instance
            self.render(self.mode)
        elif result == ord("q"):
            self.source.release()
            self.video_source.release()
            cv2.destroyAllWindows()
            self.stop = True
        else: # no defined input given
            return


    def mouse_callback(self, e, x, y, flags, params):
        """ Mouse action """
        
        if self.mode == "Visual":
            min_dist, min_idx = distance_utils.min_distance(self.path, x, y)
            if (e == cv2.EVENT_LBUTTONDOWN): # MB1 down
                if min_dist < self.config.min_px_dist: 
                    print(f"Point {min_idx} is type {self.path[min_idx].type} at position: {self.path[min_idx].pos}")
                return

        elif self.mode == "Edit":
            
            # Closest clicked point
            min_dist, min_idx = distance_utils.min_distance(self.path, x, y)

            if min_dist < self.config.min_px_dist: 
                if (e == cv2.EVENT_LBUTTONDOWN): # MB1 down (move)
                    self.moving_circle = True
                    self.moving_circle_idx = min_idx
                    return
                elif (e == cv2.EVENT_RBUTTONDOWN): # MB2 down (change type)

                    if (min_idx == 0) or (min_idx == len(self.path)-1): return # Can't change edge points

                    if self.check_spline_change(min_idx): return 

                    new_type = "spline" if self.path[min_idx].type == "linear" else "linear"
                    self.path[min_idx].type = new_type
                    self.render(self.mode)
                    if self.config.debug > 0: print(f"changed point to {new_type}")
                    return
                elif (e == cv2.EVENT_MBUTTONDOWN): # scroll down (delete)
                    self.delete_point(min_idx)
                    self.render(self.mode)
                    if self.config.debug > 0: print(f"deleted point number {min_idx}")
                    return

            if self.moving_circle:
                self.path[self.moving_circle_idx].pos = (x,y)
                self.render(self.mode)

            if (e == cv2.EVENT_LBUTTONUP): # MB1 up
                self.moving_circle = False
                self.render(self.mode)

        elif self.mode == "Insert":

            if (e == cv2.EVENT_LBUTTONDOWN): # MB1 down (add point/ close path)

                # Close path 
                if self.check_closed_path(x, y):
                    self.path[-1].type = "end"
                    self.render(self.mode)
                    if self.config.debug > 0:
                        print(f"Connected a path")
                    return
                
                # New point
                cv2.circle(self.frame, (x,y), **self.config.circle_kwargs)
                if not self.highlight: self.highlight = True

                # Close path line
                if self.path:
                    if (self.path[-1].type == "end"): self.path[-1].type = "linear"
                    cv2.line(self.frame, self.path[-1].pos, (x,y), **self.config.line_kwargs)

                if self.config.debug > 0: print(f"New circle at x: {x}, y: {y}")
                self.path.append(Point(x,y))
                self.render(self.mode)
                return
            
            elif (e == cv2.EVENT_RBUTTONDOWN): # MB2 down (add point between)

                # Consider all points except last point
                min_dist, min_idx = distance_utils.min_distance(self.path[:-1], x, y)

                if min_dist < self.config.min_px_dist: 

                    # find mid point between idx and idx +1
                    x_mid, y_mid = distance_utils.mid_point(self.path[min_idx], self.path[min_idx+1])
                    middle_point = Point(x_mid, y_mid)
                    if self.config.debug > 0: print(f"New circle between index {min_idx} and index {min_idx+1} at x: {x}, y: {y}")
                    self.path.insert(min_idx+1, middle_point)
                    self.render(self.mode)

        else:
            raise RuntimeError(f"Unexpected runtime mode {self.mode}")


    def forward(self):
        """ Propagate forward in the source feed """

        # Interpolate 
        #self.interpolate(self.path,self.last_path)
        self.last_path = copy.deepcopy(self.path)

        # Propogate forward N frames
        for _ in range(self.config.frame_skips):
            success, frame = self.source.read()
            if not success: 
                self.source.release()
                cv2.destroyAllWindows()
                self.stop = True
                return

        # Update frame
        self.frame: np.ndarray = frame
        self.frame0 = frame.copy() # raw frame
        self.render(self.mode)


    def render(self, mode):
        """ render() is called whenever frame changes and needs to re-render """

        # Render from raw frame
        self.frame = self.frame0.copy()

        if self.show_annotation:
            # Render path points and connections
            for point_idx, point in enumerate(self.path):

                # Draw point
                cv2.circle(self.frame, point.pos, **self.config.circle_kwargs) 

                # Draw line
                if point_idx != 0:
                    if point.type in ["linear","end"]:
                        
                        if self.path[point_idx-1].type == "spline": continue

                        # Linear line
                        cv2.line(
                            self.frame,
                            self.path[point_idx-1].pos,
                            self.path[point_idx].pos,
                            **self.config.line_kwargs
                        )
                    elif point.type == "spline":

                        if self.path[point_idx-1].type == "spline": continue # only render first of consecutive spline points

                        next_idx = point_idx+1 if (self.path[point_idx+1].type == "spline") else point_idx
                        spline_pixels = curve_utils.spline_curve(self.path[point_idx-1:next_idx+2]) # (x,y)

                        # Edge detection
                        spline_pixels = np.maximum(spline_pixels, 0)
                        spline_pixels[:,0] = np.minimum(spline_pixels[:,0], self.frame_shape[1]-1)
                        spline_pixels[:,1] = np.minimum(spline_pixels[:,1], self.frame_shape[0]-1)

                        self.frame[spline_pixels[:,1],spline_pixels[:,0],:] = self.config.color

            # Render fill
            if self.path and (self.path[-1].type == "end"):

                # Connect end to start with line
                cv2.line(
                    self.frame,
                    self.path[-1].pos,
                    self.path[0].pos,
                    **self.config.line_kwargs
                )

                # Fill in the polygon
                if not self.moving_circle:

                    color_mask = cv2.inRange(self.frame, self.config.color, self.config.color) # 0/255 mask, np-uint8
                    contours, _ = cv2.findContours(color_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                    contour_image = self.frame0.copy()
                    for cont in contours:
                        cv2.drawContours(contour_image, [cont], contourIdx=-1, color=self.config.color, thickness=cv2.FILLED)
                    self.frame = cv2.addWeighted(self.frame, config.alpha, contour_image, 1-config.alpha, 0)

            # Render circle highlight point
            if self.highlight:
                cv2.circle(self.frame, self.path[-1].pos, **self.config.highlight_kwargs)

        # Render mode text
        self.update_mode(mode=mode)
        
    

    def interpolate(self):

        black_mask = np.zeros(self.frame_shape, dtype=np.uint8)
        
        if (self.path is None) or (self.last_path is None):
            self.write_frame(black_mask.T, n=self.config.frame_skips)
        elif (self.path == self.last_path):
            mask = self.path_to_mask(self.last_path)
            self.write_frame(mask.T, n=self.config.frame_skips)
        else:
            pass # perform interpolation

            first_frame = self.path_to_mask(self.last_path)




            last_frame = self.path_to_mask(self.path)

            # create pair-wise mapping between both paths
    

    def path_to_mask(self, path):
        pass
    
    def write_frame(self, binary_mask, n=1):

        mask = binary_mask.T
        stacked_mask = np.stack((mask,mask,mask), axis=-1) 
        for _ in range(n):
            self.video_source.write(stacked_mask)
        return

    def update_mode(self, mode):
        """ Updates mode and renders it on screen"""
        self.mode = mode
        cv2.putText(self.frame, f"Mode: {self.mode}", **self.config.text_kwargs)

    def check_closed_path(self, x, y):
        if (len(self.path) > 2) and (distance_utils.distance(self.path[0].pos, x, y) < self.config.min_px_dist): 
                return True
        else:
            return False
    
    def check_spline_change(self, min_idx):
        """ Returns True if cannot change type of point """

        if (self.path[min_idx-1].type == "spline") and (self.path[min_idx+1].type == "spline"):
            return True # Surrounded
        if (min_idx > 1) and (self.path[min_idx-1].type == "spline") and (self.path[min_idx-2].type == "spline"):  
            return True # 2 behind
        if (min_idx < len(self.path)-2) and (self.path[min_idx+1].type == "spline") and (self.path[min_idx+2].type == "spline"):  
            return True # 2 infront
        return False

    def delete_point(self, idx):

        # Delete point
        if (idx == len(self.path)-1) and len(self.path) > 2: # deleting last point
            if self.path[idx].type == "end":
                self.path[idx-1].type = "end"
            del self.path[idx]
        else:
            del self.path[idx]

        # Update highlight
        if len(self.path) == 0: self.highlight = False



if __name__ == "__main__":

    config = Config()
    source_path = r".\Videos\ScenicDrive_short.mp4"
    SegTool(source_path, config).start_window()