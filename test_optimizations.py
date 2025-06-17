#!/usr/bin/env python3
"""
Quick test script to demonstrate optimization performance improvements.
"""

import numpy as np
import time
import cv2
from scipy import ndimage as ndi
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# Import our optimized functions
from examples.instance_segmentation_example import (
    create_instance_mask, 
    ensure_connected_instance_labels,
    _vectorized_size_filter,
    colorize_instance_mask
)

def create_test_images():
    """Create test images with different characteristics for benchmarking."""
    
    # Sparse image (1% foreground)
    sparse_img = np.zeros((512, 512), dtype=np.uint8)
    for i in range(5):
        center = (np.random.randint(50, 462), np.random.randint(50, 462))
        cv2.circle(sparse_img, center, np.random.randint(10, 20), 1, -1)
    
    # Medium density image (15% foreground)
    medium_img = np.zeros((512, 512), dtype=np.uint8)
    for i in range(25):
        center = (np.random.randint(30, 482), np.random.randint(30, 482))
        cv2.circle(medium_img, center, np.random.randint(8, 15), 1, -1)
    
    # Dense image (40% foreground)
    dense_img = np.zeros((512, 512), dtype=np.uint8)
    for i in range(100):
        center = (np.random.randint(20, 492), np.random.randint(20, 492))
        cv2.circle(dense_img, center, np.random.randint(5, 12), 1, -1)
    
    return {
        'sparse': sparse_img,
        'medium': medium_img, 
        'dense': dense_img
    }

def benchmark_function(func, *args, **kwargs):
    """Benchmark a function call and return timing."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    return result, end_time - start_time

def test_create_instance_mask():
    """Test the optimized create_instance_mask function."""
    print("🔬 Testing create_instance_mask() optimization...")
    
    test_images = create_test_images()
    
    for img_type, binary_mask in test_images.items():
        print(f"\n  {img_type.capitalize()} image ({np.sum(binary_mask)/binary_mask.size*100:.1f}% foreground):")
        
        # Test with our optimized version
        result, elapsed = benchmark_function(create_instance_mask, binary_mask)
        num_objects = len(np.unique(result)) - 1
        print(f"    ✅ Optimized: {elapsed:.4f}s → {num_objects} objects")

def test_connectivity_check():
    """Test the ultra-fast connectivity checking."""
    print("\n🔗 Testing ensure_connected_instance_labels() optimization...")
    
    # Create a test mask with many labels (simulates real segmentation output)
    test_mask = np.zeros((256, 256), dtype=np.int32)
    label = 1
    for i in range(0, 256, 20):
        for j in range(0, 256, 20):
            cv2.circle(test_mask, (j+10, i+10), 8, label, -1)
            label += 1
    
    num_labels = len(np.unique(test_mask)) - 1
    density = np.sum(test_mask > 0) / test_mask.size
    
    print(f"  Test mask: {num_labels} labels, density {density:.4f}")
    
    result, elapsed = benchmark_function(ensure_connected_instance_labels, test_mask)
    print(f"    ✅ Ultra-fast connectivity: {elapsed:.6f}s (likely skipped due to density heuristic)")

def test_size_filtering():
    """Test the vectorized size filtering."""
    print("\n📏 Testing _vectorized_size_filter() optimization...")
    
    # Create a mask with objects of various sizes
    test_mask = np.zeros((256, 256), dtype=np.int32)
    label = 1
    
    # Small objects (should be removed)
    for i in range(5):
        cv2.circle(test_mask, (50 + i*20, 50), 2, label, -1)
        label += 1
    
    # Medium objects (should be kept)
    for i in range(5):
        cv2.circle(test_mask, (50 + i*20, 100), 8, label, -1)
        label += 1
    
    # Large objects (should be removed if max_size set)
    for i in range(3):
        cv2.circle(test_mask, (80 + i*40, 150), 25, label, -1)
        label += 1
    
    original_count = len(np.unique(test_mask)) - 1
    print(f"  Original: {original_count} objects")
    
    # Test size filtering
    result, elapsed = benchmark_function(_vectorized_size_filter, test_mask, 20, 500)
    filtered_count = len(np.unique(result)) - 1
    print(f"    ✅ Vectorized filtering: {elapsed:.6f}s → {filtered_count} objects (removed {original_count - filtered_count})")

def test_colorization():
    """Test the optimized colorization."""
    print("\n🎨 Testing colorize_instance_mask() optimization...")
    
    # Test with different numbers of labels
    test_cases = [
        ("Few labels", 5),
        ("Many labels", 100)
    ]
    
    for case_name, num_labels in test_cases:
        # Create test mask
        test_mask = np.zeros((256, 256), dtype=np.int32)
        for i in range(num_labels):
            x = (i % 10) * 25 + 10
            y = (i // 10) * 25 + 10
            cv2.circle(test_mask, (x, y), 8, i + 1, -1)
        
        result, elapsed = benchmark_function(colorize_instance_mask, test_mask)
        print(f"  {case_name} ({num_labels}): {elapsed:.6f}s")

def main():
    """Run all optimization tests."""
    print("🚀 SEGMENTATION OPTIMIZATION BENCHMARK")
    print("=" * 50)
    
    # Set random seed for reproducible results
    np.random.seed(42)
    
    # Run tests
    test_create_instance_mask()
    test_connectivity_check()
    test_size_filtering()
    test_colorization()
    
    print("\n" + "=" * 50)
    print("✅ All optimization tests completed!")
    print("\n💡 Key optimizations:")
    print("   • Adaptive algorithm selection based on image characteristics")
    print("   • Density-based early exits for connectivity checking")
    print("   • Vectorized operations for size filtering")
    print("   • Efficient color palettes and I/O operations")
    print("\n🎯 For maximum speed, use: --fast-mode")

if __name__ == "__main__":
    main() 