import numpy as np

class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, process_noise, measurement_noise, state_transition, observation_model):
        """
        Initialize the Kalman Filter with the given parameters.
        
        Parameters:
        - initial_state: Initial state vector.
        - initial_covariance: Initial covariance matrix.
        - process_noise: Process noise covariance matrix.
        - measurement_noise: Measurement noise covariance matrix.
        - state_transition: State transition matrix.
        - observation_model: Observation model matrix.
        """
        self.state = initial_state
        self.covariance = initial_covariance
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.state_transition = state_transition
        self.observation_model = observation_model
        self.previous_positions = [initial_state.copy()]
        self.missed_frames = 0

    def predict(self):
        """
        Predict the next state based on the current state and state transition model.
        
        Returns:
        - state: Predicted state vector.
        - covariance: Predicted covariance matrix.
        """
        # Predict the next state using the state transition matrix
        self.state = np.dot(self.state_transition, self.state)
        
        # Predict the covariance of the state
        self.covariance = np.dot(np.dot(self.state_transition, self.covariance), self.state_transition.T) + self.process_noise
        
        return self.state, self.covariance

    def update(self, measurement):
        """
        Update the state with the given measurement.
        
        Parameters:
        - measurement: The observed measurement vector.
        
        Returns:
        - state: Updated state vector.
        """
        # Predict the measurement based on the current state
        predicted_measurement = np.dot(self.observation_model, self.state)
        
        # Calculate the innovation (difference between actual and predicted measurements)
        innovation = measurement - predicted_measurement
        
        # Calculate the innovation covariance
        innovation_covariance = np.dot(np.dot(self.observation_model, self.covariance), self.observation_model.T) + self.measurement_noise
        
        # Compute the Kalman Gain
        kalman_gain = np.dot(np.dot(self.covariance, self.observation_model.T), np.linalg.inv(innovation_covariance))
        
        # Update the state with the Kalman Gain and innovation
        self.state = self.state + np.dot(kalman_gain, innovation)
        
        # Update the covariance matrix
        self.covariance = self.covariance - np.dot(np.dot(kalman_gain, self.observation_model), self.covariance)
        
        # Store the updated state and reset the missed frames counter
        self.previous_positions.append(self.state.copy())
        self.missed_frames = 0
        
        return self.state

    def increment_missed_frames(self):
        """
        Increment the count of missed frames when the filter does not receive a measurement update.
        """
        self.missed_frames += 1

    def get_previous_positions(self):
        """
        Get the list of previously recorded states.
        
        Returns:
        - previous_positions: List of previously recorded state vectors.
        """
        return self.previous_positions
