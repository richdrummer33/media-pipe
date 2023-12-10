import time
import cv2
import threading
import mediapipe as mp
import numpy as np
from enum import Enum

# globals processNoiseCov=1e-4, measurementNoiseCov=1e-1, errorCovPost=0.1
_processNoiseCov = 1e-4
_measurementNoiseCov = 0.1
_errorCovPost = 0.5
exit_signal = False

#####################################################################
####################### UI Controls  ################################
#####################################################################

import tkinter as tk
from tkinter import ttk

class UI_Kalman(tk.Frame):
    def __init__(self, master=None, label=None):
        super().__init__(master)
        self.master = master
        self.grid()

        global _processNoiseCov, _measurementNoiseCov, _errorCovPost
        self.processNoiseCov = tk.DoubleVar(value=_processNoiseCov)
        self.measurementNoiseCov = tk.DoubleVar(value=_measurementNoiseCov)
        self.errorCovPost = tk.DoubleVar(value=_errorCovPost)

        self.process_noise_label = ttk.Label(self, text=str(_processNoiseCov))
        self.process_noise_label.grid(row=0, column=2)

        self.measurement_noise_label = ttk.Label(self, text=str(_measurementNoiseCov))
        self.measurement_noise_label.grid(row=1, column=2)

        self.error_cov_label = ttk.Label(self, text=str(_errorCovPost))
        self.error_cov_label.grid(row=2, column=2)

        self.create_widgets()

    def create_widgets(self):
        global _processNoiseCov, _measurementNoiseCov, _errorCovPost
        global exit_signal

        if(exit_signal):
            self.quit()

        # Process Noise Cov
        ttk.Label(self, text="Process Noise Covariance").grid(row=0, column=0)
        process_slider = ttk.Scale(self, from_=1e-6, to_=1e-1, variable=self.processNoiseCov, orient="horizontal")
        process_slider.grid(row=0, column=1)
        process_slider.bind(
            "<Motion>",
            lambda event: self.on_slider_change_processNoiseCov(process_slider.get(), self.processNoiseCov, 1e-6, 1e-1))

        # Measurement Noise Cov
        ttk.Label(self, text="Measurement Noise Covariance").grid(row=1, column=0)
        measure_slider = ttk.Scale(self, from_=1e-6, to_=1e-2, variable=self.measurementNoiseCov, orient="horizontal")
        measure_slider.grid(row=1, column=1)
        measure_slider.bind(
            "<Motion>",
            lambda event: self.on_slider_change_measurementNoiseCov(measure_slider.get(), self.measurementNoiseCov, 1e-6, 1e-2))

        # Error Cov Post
        ttk.Label(self, text="Error Cov Post").grid(row=2, column=0)
        error_slider = ttk.Scale(self, from_=0.01, to_=1, variable=self.errorCovPost, orient="horizontal")
        error_slider.grid(row=2, column=1)
        error_slider.bind(
            "<Motion>",
            lambda event: self.on_slider_change_errorCovPost(error_slider.get(), self.errorCovPost, 0.01, 1))

        # loop
        self.mainloop()

    
    def on_slider_change_processNoiseCov(self, val, variable, min_val, max_val):
        global _processNoiseCov
        new_val = float(val)
        new_val = max(min(new_val, max_val), min_val)
        _processNoiseCov = new_val
        self.process_noise_label.config(text=str(_processNoiseCov))

    def on_slider_change_measurementNoiseCov(self, val, variable, min_val, max_val):
        global _measurementNoiseCov
        new_val = float(val)
        new_val = max(min(new_val, max_val), min_val)
        _measurementNoiseCov = new_val
        self.measurement_noise_label.config(text=str(_measurementNoiseCov))

    def on_slider_change_errorCovPost(self, val, variable, min_val, max_val):
        global _errorCovPost
        new_val = float(val)
        new_val = max(min(new_val, max_val), min_val)
        _errorCovPost = new_val
        self.error_cov_label.config(text=str(_errorCovPost))

        
#####################################################################
####################### Filters #####################################
#####################################################################

# Ref:      https://chat.openai.com/c/67b75b6d-53c6-438f-a576-c7011d3e83c2
# See also: https://jayrambhia.wordpress.com/2012/07/26/kalman-filter/

# RB: Kalman filter
class Kalman2D:
    def __init__(self):
        global _processNoiseCov, _measurementNoiseCov, _errorCovPost
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * _processNoiseCov
        self.kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * _measurementNoiseCov
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * _errorCovPost

    def update(self, point):
        # if the values differ from the global variable, update it
        if self.kalman.processNoiseCov[0,0] != _processNoiseCov:
            self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * _processNoiseCov

        if self.kalman.measurementNoiseCov[0,0] != _measurementNoiseCov:
            self.kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * _measurementNoiseCov
            
        if self.kalman.errorCovPost[0,0] != _errorCovPost:
            self.kalman.errorCovPost = np.eye(4, dtype=np.float32) * _errorCovPost
        
        # run the kalman filter
        prediction = self.kalman.predict()
        estimate = self.kalman.correct(np.array(point, np.float32))
        return (int(estimate[0]), int(estimate[1]))

# RB:   Exponential moving average filter
class EMAFilter:
    def __init__(self, alpha):
        self.alpha = alpha  # Smoothing factor (0 < alpha < 1)
        self.prev_value = None

    def update(self, new_value):
        if self.prev_value is None:
            self.prev_value = new_value
        else:
            self.prev_value = [(1 - self.alpha) * prev + self.alpha * new for prev, new in zip(self.prev_value, new_value)] # self.prev_value = (1 - self.alpha) * self.prev_value + self.alpha * new_value
        return self.prev_value  # Always return the updated value

    def get_smoothed_value(self):
        return self.prev_value

# RB:   Low pass filter
class LowpassFilter:
    def __init__(self, alpha):
        self.alpha = alpha  # Smoothing factor (0 < alpha < 1)
        self.prev_output = None

    def update(self, new_input):
        if self.prev_output is None:
            self.prev_output = new_input
        else:
            # Apply the low-pass filter to each element of the input sequence
            self.prev_output = [self.alpha * new + (1 - self.alpha) * prev
                                for new, prev in zip(new_input, self.prev_output)]
        return self.prev_output

    def get_filtered_value(self):
        return self.prev_output

# RB:   Sliding window filter
class SlidingWindowFilter:
    def __init__(self, window_size):
        self.window_size = window_size
        self.data = []

    def update(self, new_input):
        if isinstance(new_input, list):
            self.data.extend(new_input)
        else:
            self.data.append(new_input)

        while len(self.data) > 2 * self.window_size:
            self.data.pop(0)  # Remove excess data points

        x_values = self.data[::2]  # Extract x values
        y_values = self.data[1::2]  # Extract y values

        avg_x = sum(x_values) / len(x_values) if x_values else None
        avg_y = sum(y_values) / len(y_values) if y_values else None

        return [avg_x, avg_y]


#####################################################################
####################### FaceMeshDetector ############################
#####################################################################

class FaceMeshDetector():

    class Filter(Enum):
        NONE = 1
        KALMAN = 2
        EMA = 3 # EXPONENTIAL MOVING AVERAGE
        LOWPASS = 4
        SLIDING_WINDOW = 5

    filter = Filter.KALMAN

    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5):

        # FILTERS
        self.ema_filters = {}
        self.ema_alpha = 0.8

        self.lowpass_filters = {}
        self.lowpass_alpha = 0.2

        self.kalman_filters = {} # RB [Ref: https://chat.openai.com/c/67b75b6d-53c6-438f-a576-c7011d3e83c2]

        self.sliding_window_filters = {}
        self.sliding_window_size = 5
        
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
            static_image_mode=self.staticMode, 
            max_num_faces=self.maxFaces,
            min_detection_confidence=self.minDetectionCon, 
            min_tracking_confidence=self.minTrackCon
            )
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        # cv2.namedWindow("FaceMeshWindow", cv2.WINDOW_NORMAL)
        # cv2.resizeWindow("FaceMeshWindow", 800, 600)
        global exit_signal
        if(exit_signal):
            cv2.destroyAllWindows()
            return img, None
    
        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for face_id, faceLms in enumerate(self.results.multi_face_landmarks):

                # Unfiltered landmarks
                #if draw:
                #    self.mpDraw.draw_landmarks(img, faceLms, self.mpFaceMesh.FACEMESH_CONTOURS,
                #                           self.drawSpec, self.drawSpec)

                face = []
                for id,lm in enumerate(faceLms.landmark):
                    # print(lm)
                    ih, iw, ic = img.shape
                    x,y = int(lm.x*iw), int(lm.y*ih)

                    point_id = (face_id, id)

                    # Kalman filter call
                    if self.filter == self.Filter.KALMAN:
                        if point_id not in self.kalman_filters:
                            self.kalman_filters[point_id] = Kalman2D()
                        x, y = self.kalman_filters[point_id].update([x, y])
                    # Exponential moving average filter call
                    elif self.filter == self.Filter.EMA:
                        if face_id not in self.ema_filters:
                            self.ema_filters[point_id] = EMAFilter(self.ema_alpha)
                        x, y = self.ema_filters[point_id].update([x, y])
                    # Low-pass filter call
                    elif self.filter == self.Filter.LOWPASS:
                        if face_id not in self.lowpass_filters:
                            self.lowpass_filters[point_id] = LowpassFilter(self.lowpass_alpha)
                        x, y = self.lowpass_filters[point_id].update([x, y])
                    # Sliding window filter call
                    elif self.filter == self.Filter.SLIDING_WINDOW:
                        if face_id not in self.sliding_window_filters:
                            self.sliding_window_filters[point_id] = SlidingWindowFilter(self.sliding_window_size)
                        x, y = self.sliding_window_filters[point_id].update([x, y])

                    # Draw filtered landmarks
                    # Corrected
                    if draw and x is not None and y is not None:  # Check for None
                        x, y = int(x), int(y)  # Cast to integer
                        cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

                    # cv2.imshow("FaceMeshWindow", img)  

                    # Adds text to the points
                    # cv2.putText(img, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN,
                    #           0.7, (0, 255, 0), 1)
                    # print(id,x,y)

                    face.append([x,y])
                faces.append(face)
        return img, faces


#####################################################################
####################### Example usage ###############################
#####################################################################

class Mode(Enum):
    IMAGE = 1
    VIDEO = 2
    CAMERA = 3

def image_update(img):
    pTime = 0
    detector = FaceMeshDetector(maxFaces=1)
        
    while True:
        # Clone the image to keep the original intact
        img_copy = img.copy()
        img_copy, faces = detector.findFaceMesh(img_copy)
        #if len(faces) != 0:
        #    print(faces[0])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img_copy, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 3)
        cv2.imshow("Image", img_copy)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def video_update(cap):
    pTime = 0
    detector = FaceMeshDetector(maxFaces=1)
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        #if len(faces)!= 0:
        #    print(faces[0])
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN,
                    3, (0, 255, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)

def ui_start():
        root = tk.Tk()
        root.title("Kalman Filter Params")
        app = UI_Kalman(master=root)
        app.mainloop()

def main():
    switcher = {
        Mode.IMAGE: lambda: image_update(cv2.imread("Images/1.jpg")),
        Mode.VIDEO: lambda: video_update(cv2.VideoCapture("Videos/1.mp4")),
        Mode.CAMERA: lambda: video_update(cv2.VideoCapture(0))
    }
    selected_mode = Mode.CAMERA 
    switcher[selected_mode]()

if __name__ == "__main__":
    main()
        #t1 = threading.Thread(target=main)
        #t2 = threading.Thread(target=ui_start)
        
  #      t1.start()
  #      t2.start()
  #      
  #      t1.join()
  #      t2.join()
#
  #  except KeyboardInterrupt:
  #      print("Ctrl-C received, setting exit signal.")
  #      exit_signal = True
#
  #      # Optionally, wait for threads to finish
  #      t1.join()
  #      t2.join()
#
  #      print("Threads have been terminated.")
