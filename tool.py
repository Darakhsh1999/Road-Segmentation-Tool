import utils
import cv2
import numpy as np
from point import Point
from config import Config


class SegTool():

    def __init__(self, source_path, config):
        
        self.config: Config = config
        self.source = cv2.VideoCapture(source_path)
        _, self.frame = self.source.read() # (H,W,C), np-uint8
        self.frame0 = self.frame.copy()
        self.update_mode("Visual")
        self.stop = False # Stop program
        self.moving_circle = False # Drag and drop points
        self.path: list[Point] = []
        self.highlight: int | None = None # Highlight point

        self.start_window() # init window


    def start_window(self):
        """ Initializes window and run the main event loop """
        
        # Window 
        cv2.namedWindow("root", self.config.window_mode) 
        cv2.setMouseCallback("root", self.mouse_callback)

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

    def parse_key(self, result):
        """ Key press listener """
        
        if result == ord("f"):
            self.forward()
        elif result == ord("v"):
            self.render("Visual")
        elif result == ord("e"):
            self.render("Edit")
        elif result == ord("i"):
            self.render("Insert")
        elif result == ord("c"):
            self.render("Car")
        elif result == ord("r"): 
            self.path = []
            self.highlight = None
            self.render(self.mode)
        elif result == ord("q"):
            self.source.release()
            cv2.destroyAllWindows()
            self.stop = True
        else: # no defined input given
            return


    def mouse_callback(self, e, x, y, flags, params):
        """ Mouse action """
        
        if self.mode == "Visual":
            min_dist, min_idx = utils.min_distance(self.path, x, y)
            if (e == cv2.EVENT_LBUTTONDOWN): # MB1 down
                if min_dist < self.config.min_px_dist: 
                    print(f"Point {min_idx} is type {self.path[min_idx].type} at position: {self.path[min_idx].pos}")
                return

        elif self.mode == "Edit":
            
            # Closest clicked point
            min_dist, min_idx = utils.min_distance(self.path, x, y)

            if min_dist < self.config.min_px_dist: 
                if (e == cv2.EVENT_LBUTTONDOWN): # MB1 down (move)
                    self.moving_circle = True
                    self.moving_circle_idx = min_idx
                    return
                elif (e == cv2.EVENT_RBUTTONDOWN): # MB2 down (change type)
                    old_point = self.path[min_idx]
                    new_type = "spline" if old_point.type == "linear" else "linear"
                    px, py = old_point.pos
                    self.path[min_idx] = Point(px, py, type=new_type)
                    self.render(self.mode)
                    print(f"changed point type from {old_point.type} to {new_type} ")
                    return
                elif (e == cv2.EVENT_MBUTTONDOWN): # scroll down (delete)
                    self.delete_point(min_idx)
                    self.render(self.mode)
                    print(f"deleted point number {min_idx}")
                    return

            if self.moving_circle:
                self.path[self.moving_circle_idx].pos = (x,y)
                self.render(self.mode)

            if (e == cv2.EVENT_LBUTTONUP): # MB1 up
                self.moving_circle = False

        elif self.mode == "Insert":

            if (e == cv2.EVENT_LBUTTONDOWN): # MB1 down

                # Close path 
                if self.check_closed_path(x, y):
                    self.path[-1].type = "end"
                    self.render(self.mode)
                    print(f"Connected a path")
                    return

                # New point
                cv2.circle(self.frame, (x,y), **self.config.circle_kwargs)
                if self.highlight is None:
                    self.highlight = 0
                else:
                    self.highlight += 1

                if self.path:
                    cv2.line(self.frame, self.path[-1].pos, (x,y), **self.config.line_kwargs)
                print(f"New circle at x: {x}, y: {y}")
                self.path.append(Point(x,y))
                self.render(self.mode)
                return

        elif self.mode == "Car":
            pass
        else:
            raise RuntimeError(f"Unexpected runtime mode {self.mode}")


    def forward(self):
        """ Propagate forward in the source feed """
        for _ in range(self.config.frame_skipe):
            _, frame = self.source.read()
        self.frame = frame
        self.frame0 = frame.copy() # raw frame
        self.render(self.mode)


    def render(self, mode):
        """ Is called whenever frame changes and needs to re-render """

        # Render from raw frame
        self.frame = self.frame0.copy()

        # Render mode text
        self.update_mode(mode=mode)
        
        # Render path points and connections
        for point_idx, point in enumerate(self.path):

            # Draw point
            cv2.circle(self.frame, point.pos, **self.config.circle_kwargs) 

            if point_idx != 0:
                if point.type == "linear":
                    # Draw linear line
                    cv2.line(
                        self.frame,
                        self.path[point_idx-1].pos,
                        self.path[point_idx].pos,
                        **self.config.line_kwargs
                    )
                elif point.type == "spline":
                    # TODO Draw polynomial line
                    pass

        
        # Render highlight point
        if self.highlight is not None:
            cv2.circle(self.frame, self.path[self.highlight].pos, **self.config.highlight_kwargs)

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
            pts = np.array([point.pos for point in self.path])
            poly_frame = self.frame.copy()
            cv2.fillPoly(poly_frame, pts=[pts], color=config.color)
            self.frame = cv2.addWeighted(self.frame, config.alpha, poly_frame, 1-config.alpha, 0)


    def update_mode(self, mode):
        """ Updates mode and renders it on screen"""
        self.mode = mode
        cv2.putText(self.frame, f"Mode: {self.mode}", **self.config.text_kwargs)

    def check_closed_path(self, x, y):
        if (len(self.path) > 2) and (utils.distance(self.path[0].pos, x, y) < self.config.min_px_dist): 
                return True
        else:
            return False

    def delete_point(self, idx):

        if self.highlight == 0:
            self.highlight = None
        else:
            self.highlight -= 1
        if (idx == len(self.path)-1) and len(self.path) > 2: # deleting last point
            if self.path[idx].type == "end":
                self.path[idx-1].type = "end"
            del self.path[idx]
        else:
            del self.path[idx]
        return


if __name__ == "__main__":

    config = Config()
    source_path = r".\Videos\ScenicDrive.mp4"
    SegTool(source_path, config)