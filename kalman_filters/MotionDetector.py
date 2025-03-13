import cv2
import numpy as np
from scipy.ndimage import label
from skimage.measure import regionprops
from skimage.morphology import dilation, square
from KalmanFilter import KalmanFilter

class MotionDetector:
    def __init__(self, frame_hysteresis, motion_threshold, distance_threshold, skip_frames, max_objects):
        """
        Initialize the MotionDetector with parameters for tracking.
        
        Parameters:
        - frame_hysteresis: Number of frames to consider for object activation/deactivation.
        - motion_threshold: Minimum motion area to consider an object.
        - distance_threshold: Maximum distance to match object candidates with existing objects.
        - skip_frames: Number of frames to skip between updates.
        - max_objects: Maximum number of objects to track simultaneously.
        """
        self.frame_hysteresis = frame_hysteresis
        self.motion_threshold = motion_threshold
        self.distance_threshold = distance_threshold
        self.skip_frames = skip_frames
        self.max_objects = max_objects
        self.objects = []
        self.previous_frames = []
        self.frame_count = 0

    def update_tracking(self, frame):
        """
        Update object tracking based on the current frame.
        
        Parameters:
        - frame: The current frame from the video feed.
        """
        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.previous_frames.append(gray_frame)

        # Maintain only the last 'frame_hysteresis' frames
        if len(self.previous_frames) > self.frame_hysteresis:
            self.previous_frames.pop(0)

        # Skip frames based on 'skip_frames' parameter
        if self.frame_count % self.skip_frames != 0:
            self.frame_count += 1
            return

        # Process frames to detect motion
        motion_frame = self.calculate_motion_frame()
        threshold_frame = self.apply_threshold(motion_frame)
        dilated_frame = self.dilate_frame(threshold_frame)
        labeled_frame, num_objects = label(dilated_frame)
        object_candidates = self.get_object_candidates(labeled_frame, num_objects)

        # Update object trackers with new candidates
        self.update_objects(object_candidates)

        # Draw bounding boxes around tracked objects
        self.draw_bounding_boxes(frame)
        self.frame_count += 1

    def calculate_motion_frame(self):
        """
        Calculate the motion frame by comparing differences between recent frames.
        
        Returns:
        - motion_frame: The frame highlighting areas of motion.
        """
        if len(self.previous_frames) < 3:
            return np.zeros_like(self.previous_frames[-1])
        
        # Compute differences between consecutive frames
        diff1 = cv2.absdiff(self.previous_frames[-1], self.previous_frames[-2])
        diff2 = cv2.absdiff(self.previous_frames[-2], self.previous_frames[-3])
        motion_frame = np.minimum(diff1, diff2)
        return motion_frame

    def apply_threshold(self, frame):
        """
        Apply a threshold to filter out noise from the motion frame.
        
        Parameters:
        - frame: The motion frame to threshold.
        
        Returns:
        - threshold_frame: The frame after applying the threshold.
        """
        threshold_frame = np.where(frame > self.motion_threshold, frame, 0)
        return threshold_frame

    def dilate_frame(self, frame):
        """
        Dilate the thresholded frame to enhance detected motion areas.
        
        Parameters:
        - frame: The thresholded frame.
        
        Returns:
        - dilated_frame: The dilated frame with enhanced motion areas.
        """
        dilated_frame = dilation(frame, square(3))
        return dilated_frame

    def get_object_candidates(self, labeled_frame, num_objects):
        """
        Extract object candidates from the labeled frame.
        
        Parameters:
        - labeled_frame: The frame with labeled regions of detected motion.
        - num_objects: Number of detected objects.
        
        Returns:
        - object_candidates: List of object candidates with their centroids and bounding boxes.
        """
        object_candidates = []
        for region in regionprops(labeled_frame):
            if region.area >= self.motion_threshold:
                object_candidates.append({
                    'centroid': region.centroid,
                    'bbox': region.bbox
                })
        return object_candidates

    def update_objects(self, object_candidates):
        """
        Update tracked objects with new candidates.
        
        Parameters:
        - object_candidates: List of new object candidates.
        """
        new_objects = []
        for candidate in object_candidates:
            candidate_centroid = np.array(candidate['centroid'])
            
            # Check if the candidate matches any existing object
            if len(self.objects) < self.max_objects:
                matched = False
                for obj in self.objects:
                    predicted_state, _ = obj['kalman_filter'].predict()
                    predicted_position = predicted_state[:2]
                    distance = np.linalg.norm(predicted_position - candidate_centroid)
                    
                    if distance < self.distance_threshold:
                        obj['kalman_filter'].update(candidate_centroid)
                        obj['bbox'] = candidate['bbox']
                        obj['missed_frames'] = 0
                        new_objects.append(obj)
                        matched = True
                        break
                
                # Create a new Kalman filter for unmatched candidates
                if not matched:
                    initial_state = np.hstack((candidate_centroid, [0, 0]))
                    initial_covariance = np.eye(4)
                    process_noise = np.eye(4) * 0.1
                    measurement_noise = np.eye(2) * 0.1
                    state_transition = np.array([
                        [1, 0, 1, 0],
                        [0, 1, 0, 1],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                    ])
                    observation_model = np.array([
                        [1, 0, 0, 0],
                        [0, 1, 0, 0]
                    ])
                    kalman_filter = KalmanFilter(
                        initial_state,
                        initial_covariance,
                        process_noise,
                        measurement_noise,
                        state_transition,
                        observation_model
                    )
                    new_objects.append({
                        'kalman_filter': kalman_filter,
                        'bbox': candidate['bbox'],
                        'missed_frames': 0
                    })
        
        # Remove objects that have been missed for too many frames
        self.objects = [obj for obj in new_objects if obj['missed_frames'] < self.frame_hysteresis]

    def draw_bounding_boxes(self, frame):
        """
        Draw bounding boxes around tracked objects on the frame.
        
        Parameters:
        - frame: The frame to draw bounding boxes on.
        """
        for obj in self.objects:
            y1, x1, y2, x2 = obj['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def run(self, video_path):
        """
        Run the motion detection on a video file.
        
        Parameters:
        - video_path: Path to the video file.
        """
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update tracking with the current frame
            self.update_tracking(frame)
            
            # Display the frame with tracking results
            cv2.imshow('Tracking', frame)
            
            # Exit if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Initialize and run the motion detector
    detector = MotionDetector(
        frame_hysteresis=3,
        motion_threshold=25,
        distance_threshold=50,
        skip_frames=1,
        max_objects=10
    )
    detector.run('video.mp4')
