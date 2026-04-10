"""
Landmark Extractor Module
Extracts hand and pose landmarks from video frames using MediaPipe
Normalizes and formats landmarks for LSTM model input
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple, Optional, List, Dict
import config

class LandmarkExtractor:
    """
    Extracts and normalizes hand landmarks from video frames.
    Uses MediaPipe Hands for real-time hand detection.
    """
    
    def __init__(self):
        """Initialize MediaPipe Hands detector"""
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=config.HAND_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.HAND_TRACKING_CONFIDENCE,
            model_complexity=1  # 0=light, 1=full (M1/M2 can handle both)
        )
        
    def extract_landmarks(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Dict]:
        """
        Extract hand landmarks from a single frame.
        
        Args:
            frame: Input frame from video (BGR format from OpenCV)
            
        Returns:
            Tuple of:
            - landmarks: Normalized feature vector of shape (42,) or None if no hands
            - metadata: Dict with detection info (hands_detected, confidence, frame_shape)
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get frame dimensions for normalization
        h, w, c = frame.shape
        
        # Detect hands
        results = self.hands.process(rgb_frame)
        
        metadata = {
            'hands_detected': results.multi_hand_landmarks is not None,
            'num_hands': len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0,
            'frame_shape': (h, w),
            'confidence': []
        }
        
        # If no hands detected, return None
        if not results.multi_hand_landmarks:
            return None, metadata
        
        # Extract landmarks from detected hands (up to 2 hands)
        landmarks_list = []
        
        for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Extract coordinates for this hand (21 landmarks)
            hand_coords = []
            
            for landmark in hand_landmarks.landmark:
                # Normalize coordinates to 0-1 range (invariant to frame size)
                x = landmark.x
                y = landmark.y
                z = landmark.z  # Depth information
                visibility = landmark.visibility if hasattr(landmark, 'visibility') else 1.0
                
                hand_coords.extend([x, y, z])
            
            landmarks_list.extend(hand_coords)
            
            # Store confidence if available
            if results.multi_handedness:
                confidence = results.multi_handedness[hand_idx].classification[0].score
                metadata['confidence'].append(confidence)
        
        # 21 landmarks * 3 coords (x,y,z) * 2 hands = 126 values max
        LANDMARKS_PER_HAND = 21 * 3  # always use x, y, z
        MAX_LANDMARKS = LANDMARKS_PER_HAND * 2  # support up to 2 hands

        while len(landmarks_list) < MAX_LANDMARKS:
            landmarks_list.extend([0.0, 0.0, 0.0])

        landmarks = np.array(landmarks_list[:MAX_LANDMARKS], dtype=np.float32)
        
        return landmarks, metadata
    
    def extract_landmarks_sequence(self, frame: np.ndarray, 
                                   sequence_buffer: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict]:
        """
        Extract landmarks and add to sequence buffer.
        Maintains a rolling window of landmarks for LSTM input.
        
        Args:
            frame: Input frame
            sequence_buffer: List of previous landmark arrays
            
        Returns:
            Updated sequence buffer and metadata
        """
        landmarks, metadata = self.extract_landmarks(frame)
        
        if landmarks is not None:
            sequence_buffer.append(landmarks)
            
            # Keep buffer size manageable
            if len(sequence_buffer) > config.MAX_SEQUENCE_LENGTH:
                sequence_buffer.pop(0)
        
        return sequence_buffer, metadata
    
    def draw_landmarks(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Draw hand landmarks on frame for visualization.
        
        Args:
            frame: Input frame
            landmarks: Feature vector of landmarks
            
        Returns:
            Frame with drawn landmarks
        """
        h, w = frame.shape[:2]
        
        # Reshape landmarks back to hand format (21 landmarks * 3 coordinates)
        landmarks_reshaped = landmarks[:63].reshape(21, 3)
        
        # Create MediaPipe landmark format
        from mediapipe.framework.formats import landmark_pb2
        
        hand_landmarks = landmark_pb2.NormalizedLandmarkList()
        for coord in landmarks_reshaped:
            lm = landmark_pb2.NormalizedLandmark()
            lm.x = float(coord[0])
            lm.y = float(coord[1])
            lm.z = float(coord[2])
            lm.visibility = 1.0
            hand_landmarks.landmark.append(lm)
        
        # Draw on frame
        annotated_image = frame.copy()
        if hand_landmarks.landmark:
            self.mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS,
                self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1)
            )
        
        return annotated_image
    
    def draw_hand_detection_box(self, frame: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
        """
        Draw bounding box around detected hand.
        
        Args:
            frame: Input frame
            landmarks: Feature vector
            
        Returns:
            Frame with bounding box
        """
        h, w = frame.shape[:2]
        
        # Reshape to get coordinates
        landmarks_reshaped = landmarks[:63].reshape(21, 3)
        
        # Skip if all zeros (no hand)
        if np.all(landmarks_reshaped[:, :2] == 0):
            return frame
        
        x_coords = landmarks_reshaped[:, 0]
        y_coords = landmarks_reshaped[:, 1]
        
        # Get bounding box
        x_min = int(np.min(x_coords[x_coords > 0]) * w) if np.any(x_coords > 0) else 0
        x_max = int(np.max(x_coords[x_coords > 0]) * w) if np.any(x_coords > 0) else w
        y_min = int(np.min(y_coords[y_coords > 0]) * h) if np.any(y_coords > 0) else 0
        y_max = int(np.max(y_coords[y_coords > 0]) * h) if np.any(y_coords > 0) else h
        
        # Add padding
        padding = 10
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)
        
        # Draw rectangle
        annotated = frame.copy()
        cv2.rectangle(annotated, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        
        return annotated
    
    def release(self):
        """Release MediaPipe resources"""
        self.hands.close()


def test_landmark_extractor():
    """Test the landmark extractor with webcam"""
    print("[TEST] Starting landmark extractor test...")
    
    extractor = LandmarkExtractor()
    cap = cv2.VideoCapture(0)
    
    print("[TEST] Press 'q' to quit")
    
    frame_count = 0
    landmark_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        landmarks, metadata = extractor.extract_landmarks(frame)
        frame_count += 1
        
        if landmarks is not None:
            landmark_count += 1
            
            # Draw landmarks
            frame = extractor.draw_landmarks(frame, landmarks)
            frame = extractor.draw_hand_detection_box(frame, landmarks)
            
            # Display info
            info_text = f"Hands: {metadata['num_hands']} | Landmarks shape: {landmarks.shape}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No hands detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.imshow('Landmark Extractor Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    extractor.release()
    
    print(f"[TEST] Processed {frame_count} frames")
    print(f"[TEST] Hand landmarks detected in {landmark_count} frames ({100*landmark_count/frame_count:.1f}%)")


if __name__ == "__main__":
    test_landmark_extractor()
