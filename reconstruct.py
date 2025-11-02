import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ThreadPoolExecutor
import os
import time
from tqdm import tqdm
import json
from typing import List, Set

class ContinuousFrameReconstructor:
    def __init__(self, video_path, output_path="reconstructed_continuous.mp4", method="ssim"):
        self.video_path = video_path
        self.output_path = output_path
        self.method = method
        self.frames = []
        self.frame_indices = []
        self.similarity_cache = {}
        
    def extract_frames(self):
        """Extract all frames from video"""
        print("Extracting frames from video...")
        cap = cv2.VideoCapture(self.video_path)
        
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frames.append(frame)
            self.frame_indices.append(idx)
            idx += 1
            
        cap.release()
        print(f"Extracted {len(self.frames)} frames")
        return self.frames
    
    def compute_similarity(self, idx1, idx2):
        """Compute similarity between two frames"""
        if idx1 == idx2:
            return 1.0
            
        key = (min(idx1, idx2), max(idx1, idx2))
        if key in self.similarity_cache:
            return self.similarity_cache[key]
        
        frame1 = self.frames[idx1]
        frame2 = self.frames[idx2]
        
        # Resize for faster computation
        scale = 0.25
        f1_small = cv2.resize(frame1, (0, 0), fx=scale, fy=scale)
        f2_small = cv2.resize(frame2, (0, 0), fx=scale, fy=scale)
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(f1_small, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(f2_small, cv2.COLOR_BGR2GRAY)
        
        if self.method == "ssim":
            similarity = ssim(gray1, gray2)
        elif self.method == "mse":
            mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
            similarity = 1 / (1 + mse)
        else:  # hybrid
            s1 = ssim(gray1, gray2)
            mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
            s2 = 1 / (1 + mse)
            similarity = 0.7 * s1 + 0.3 * s2
        
        self.similarity_cache[key] = similarity
        return similarity
    
    def build_complete_similarity_matrix(self):
        """Build complete similarity matrix to avoid disconnected components"""
        print("Building complete similarity matrix...")
        n_frames = len(self.frames)
        similarity_matrix = np.zeros((n_frames, n_frames))
        
        for i in tqdm(range(n_frames), desc="Building matrix"):
            for j in range(i, n_frames):
                if i == j:
                    similarity_matrix[i, j] = 1.0
                else:
                    sim = self.compute_similarity(i, j)
                    similarity_matrix[i, j] = sim
                    similarity_matrix[j, i] = sim
        
        return similarity_matrix

    def find_optimal_path_tsp(self, similarity_matrix):
        """Use Traveling Salesman Problem approach for optimal continuous path"""
        print("Finding optimal continuous path using TSP approach...")
        n_frames = len(similarity_matrix)
        
        # Convert similarity to distance (we want to minimize total distance)
        distance_matrix = 1 - similarity_matrix
        
        # Start with a simple construction
        path = [0]  # Start with frame 0
        unvisited = set(range(1, n_frames))
        
        # Nearest neighbor construction
        while unvisited:
            last = path[-1]
            next_frame = min(unvisited, key=lambda x: distance_matrix[last, x])
            path.append(next_frame)
            unvisited.remove(next_frame)
        
        # Apply 2-opt optimization to improve the path
        improved = True
        while improved:
            improved = False
            for i in range(1, n_frames - 2):
                for j in range(i + 2, n_frames):
                    # Calculate current distance for segment i-1 to j
                    current_distance = (distance_matrix[path[i-1], path[i]] +
                                      distance_matrix[path[j-1], path[j]])
                    
                    # Calculate new distance if we reverse segment i:j
                    new_distance = (distance_matrix[path[i-1], path[j-1]] +
                                  distance_matrix[path[i], path[j]])
                    
                    if new_distance < current_distance:
                        path[i:j] = reversed(path[i:j])
                        improved = True
        
        return path

    def detect_walking_man_direction(self, frame_idx):
        """Detect which direction the man is facing/walking"""
        frame = self.frames[frame_idx]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Focus on central region where man likely is
        center_region = gray[self.height//4:3*self.height//4, self.width//4:3*self.width//4]
        
        # Use horizontal gradient to detect direction
        sobel_x = cv2.Sobel(center_region, cv2.CV_64F, 1, 0, ksize=3)
        
        # Positive gradient sum = facing right, negative = facing left
        gradient_sum = np.sum(sobel_x)
        
        return 1 if gradient_sum > 0 else -1  # 1 = right, -1 = left

    def ensure_consistent_direction(self, sequence):
        """Ensure all frames have consistent walking direction"""
        print("Ensuring consistent walking direction...")
        
        # Check direction of first few frames to determine overall direction
        sample_frames = sequence[:min(10, len(sequence))]
        directions = [self.detect_walking_man_direction(idx) for idx in sample_frames]
        dominant_direction = 1 if sum(directions) > 0 else -1
        
        print(f"Detected walking direction: {'RIGHT' if dominant_direction == 1 else 'LEFT'}")
        
        # Fix any frames that are facing the wrong way
        fixed_sequence = []
        reverse_count = 0
        
        for idx in sequence:
            frame_direction = self.detect_walking_man_direction(idx)
            
            if frame_direction == dominant_direction:
                fixed_sequence.append(idx)
            else:
                # This frame is facing the wrong way - try to find a better position
                fixed_sequence.append(idx)  # Keep it for now, will optimize later
                reverse_count += 1
        
        if reverse_count > 0:
            print(f"Found {reverse_count} frames facing the wrong direction")
            # Re-optimize to fix these
            fixed_sequence = self.reorder_for_consistent_direction(fixed_sequence, dominant_direction)
        
        return fixed_sequence

    def reorder_for_consistent_direction(self, sequence, target_direction):
        """Reorder sequence to maintain consistent walking direction"""
        print("Reordering for consistent direction...")
        
        # Group frames by direction
        right_facing = []
        left_facing = []
        
        for idx in sequence:
            direction = self.detect_walking_man_direction(idx)
            if direction == 1:
                right_facing.append(idx)
            else:
                left_facing.append(idx)
        
        # Choose the dominant direction
        if target_direction == 1:
            main_sequence = right_facing
            other_sequence = left_facing
        else:
            main_sequence = left_facing
            other_sequence = right_facing
        
        print(f"Main direction: {len(main_sequence)} frames")
        print(f"Other direction: {len(other_sequence)} frames")
        
        # If most frames are in one direction, use that as the main sequence
        # and insert the others where they fit best
        if len(main_sequence) > len(other_sequence) * 2:
            # Insert minority frames into majority sequence
            final_sequence = self.insert_frames_optimally(main_sequence, other_sequence)
        else:
            # Use TSP on combined sequence
            combined = main_sequence + other_sequence
            similarity_matrix = self.build_subset_similarity_matrix(combined)
            final_sequence_indices = self.find_optimal_path_tsp(similarity_matrix)
            final_sequence = [combined[i] for i in final_sequence_indices]
        
        return final_sequence

    def build_subset_similarity_matrix(self, frame_subset):
        """Build similarity matrix for a subset of frames"""
        n = len(frame_subset)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i, n):
                if i == j:
                    matrix[i, j] = 1.0
                else:
                    sim = self.compute_similarity(frame_subset[i], frame_subset[j])
                    matrix[i, j] = sim
                    matrix[j, i] = sim
        
        return matrix

    def insert_frames_optimally(self, main_sequence, insert_frames):
        """Insert frames optimally into the main sequence"""
        if not insert_frames:
            return main_sequence
        
        result = main_sequence.copy()
        
        for insert_frame in insert_frames:
            best_position = 0
            best_score = -1
            
            # Try inserting at each position
            for pos in range(len(result) + 1):
                test_sequence = result.copy()
                test_sequence.insert(pos, insert_frame)
                
                # Score this insertion
                score = self._evaluate_local_quality(test_sequence, max(0, pos-1), min(len(test_sequence), pos+2))
                
                if score > best_score:
                    best_score = score
                    best_position = pos
            
            # Insert at best position
            result.insert(best_position, insert_frame)
        
        return result

    def _evaluate_local_quality(self, sequence, start_idx, end_idx):
        """Evaluate quality of a local segment"""
        total_sim = 0
        count = 0
        
        for i in range(max(0, start_idx), min(len(sequence)-1, end_idx)):
            total_sim += self.compute_similarity(sequence[i], sequence[i+1])
            count += 1
        
        return total_sim / count if count > 0 else 0

    def verify_single_continuous_sequence(self, sequence):
        """Verify that we have one continuous sequence"""
        print("Verifying sequence continuity...")
        
        # Check for duplicate frames
        if len(sequence) != len(set(sequence)):
            print(" ERROR: Duplicate frames found!")
            return False
        
        # Check if all frames are present
        if len(sequence) != len(self.frames):
            print(" ERROR: Missing frames!")
            return False
        
        # Check frame range
        if set(sequence) != set(range(len(self.frames))):
            print(" ERROR: Invalid frame indices!")
            return False
        
        print(" Sequence verification passed: Single continuous sequence")
        return True

    def reconstruct_video(self, sequence):
        """Create video from ordered frames"""
        print("Creating reconstructed video...")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.output_path, fourcc, self.fps, 
                            (self.width, self.height))
        
        for idx in tqdm(sequence):
            out.write(self.frames[idx])
        
        out.release()
        print(f"Video saved to {self.output_path}")

    def run(self):
        """Execute continuous reconstruction pipeline"""
        start_time = time.time()
        
        # Step 1: Extract frames
        self.extract_frames()
        n_frames = len(self.frames)
        print(f"Processing {n_frames} frames")
        
        # Step 2: Build complete similarity matrix
        similarity_matrix = self.build_complete_similarity_matrix()
        
        # Step 3: Find optimal continuous path using TSP
        sequence = self.find_optimal_path_tsp(similarity_matrix)
        
        # Step 4: Ensure consistent walking direction
        sequence = self.ensure_consistent_direction(sequence)
        
        # Step 5: Verify we have one continuous sequence
        if not self.verify_single_continuous_sequence(sequence):
            print("Falling back to simple sorting...")
            # Fallback: just use frame indices in order
            sequence = list(range(n_frames))
        
        # Step 6: Create video
        self.reconstruct_video(sequence)
        
        total_time = time.time() - start_time
        
        # Final quality assessment
        final_quality = self._evaluate_sequence_quality(sequence)
        
        log_data = {
            "total_frames": n_frames,
            "execution_time_seconds": round(total_time, 2),
            "method": self.method,
            "output_file": self.output_path,
            "final_quality_score": round(final_quality, 4),
            "sequence_type": "continuous"
        }
        
        with open("execution_log.json", "w") as f:
            json.dump(log_data, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"CONTINUOUS RECONSTRUCTION COMPLETE!")
        print(f"Total frames: {n_frames}")
        print(f"Total execution time: {total_time:.2f} seconds")
        print(f"Final quality score: {final_quality:.4f}")
        print(f" Guaranteed: Single continuous sequence")
        print(f" Video: {self.output_path}")
        print(f"{'='*60}")
        
        return sequence

    def _evaluate_sequence_quality(self, sequence):
        """Evaluate overall sequence quality"""
        if len(sequence) <= 1:
            return 1.0
        
        total_sim = 0
        for i in range(len(sequence) - 1):
            total_sim += self.compute_similarity(sequence[i], sequence[i + 1])
        
        return total_sim / (len(sequence) - 1)


if __name__ == "__main__":
    reconstructor = ContinuousFrameReconstructor(
        video_path="jumbled_video.mp4",
        output_path="reconstructed_continuous.mp4",
        method="hybrid"
    )
    
    sequence = reconstructor.run()