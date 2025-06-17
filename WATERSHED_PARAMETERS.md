# Configurable Watershed Parameters for Instance Segmentation

## 🌊 **Overview**

The instance segmentation pipeline now supports **configurable watershed parameters** that allow you to fine-tune the object separation behavior for different types of microscopy images.

## 🎛️ **Available Parameters**

### **1. Peak Detection Sensitivity**
```bash
--watershed-peak-sensitivity 0.3
```
- **Range**: `0.1` to `0.5`
- **Default**: `0.3`
- **Effect**: Controls how sensitive the peak detection is
  - **Lower values** (0.1-0.2): More sensitive, detects more peaks → **more objects separated**
  - **Higher values** (0.4-0.5): Less sensitive, fewer peaks → **fewer objects separated**
- **Use case**: Adjust based on object density and separation needs

### **2. Sparse Image Threshold**
```bash
--watershed-sparse-threshold 0.01
```
- **Range**: `0.005` to `0.02`
- **Default**: `0.01` (1% foreground)
- **Effect**: Density threshold below which simple connected components are used instead of watershed
  - **Lower values**: More images use watershed → **better separation but slower**
  - **Higher values**: More images skip watershed → **faster but less separation**
- **Use case**: Optimize for speed vs. quality trade-off

### **3. Dense Image Threshold**
```bash
--watershed-dense-threshold 0.3
```
- **Range**: `0.2` to `0.5`
- **Default**: `0.3` (30% foreground)
- **Effect**: Density threshold above which optimized dense watershed is used
  - **Lower values**: More images use dense algorithm → **better for crowded images**
  - **Higher values**: Fewer images use dense algorithm → **may miss separations in dense areas**
- **Use case**: Adjust based on typical image density

### **4. Fallback Peak Threshold**
```bash
--watershed-fallback-threshold 0.7
```
- **Range**: `0.5` to `0.9`
- **Default**: `0.7`
- **Effect**: Peak detection threshold when scikit-image is not available
  - **Lower values**: More peaks detected → **more aggressive separation**
  - **Higher values**: Fewer peaks detected → **more conservative separation**
- **Use case**: Fallback behavior tuning

## 🧪 **Recommended Settings by Image Type**

### **High-Density Cell Images** (many touching cells)
```bash
python examples/instance_segmentation_example.py \
  --watershed-peak-sensitivity 0.2 \
  --watershed-dense-threshold 0.25 \
  --watershed-sparse-threshold 0.005
```
- More aggressive peak detection
- Lower density thresholds for better separation

### **Sparse Cell Images** (few, well-separated cells)
```bash
python examples/instance_segmentation_example.py \
  --watershed-peak-sensitivity 0.4 \
  --watershed-sparse-threshold 0.02 \
  --watershed-dense-threshold 0.4
```
- Less aggressive peak detection
- Higher thresholds for speed optimization

### **Mixed Density Images** (default settings)
```bash
python examples/instance_segmentation_example.py \
  --watershed-peak-sensitivity 0.3 \
  --watershed-sparse-threshold 0.01 \
  --watershed-dense-threshold 0.3
```
- Balanced settings for general use

## 🔬 **Parameter Tuning Guide**

### **Step 1: Assess Your Images**
1. **High density**: Many touching/overlapping objects → Use aggressive settings
2. **Low density**: Well-separated objects → Use conservative settings  
3. **Mixed**: Varies per image → Use default settings

### **Step 2: Test Peak Sensitivity**
Start with different `--watershed-peak-sensitivity` values:
```bash
# More separation (more objects)
--watershed-peak-sensitivity 0.2

# Less separation (fewer objects)  
--watershed-peak-sensitivity 0.4
```

### **Step 3: Optimize for Speed vs Quality**
Adjust density thresholds based on your needs:
```bash
# Prioritize quality (slower)
--watershed-sparse-threshold 0.005 --watershed-dense-threshold 0.2

# Prioritize speed (faster)
--watershed-sparse-threshold 0.02 --watershed-dense-threshold 0.4
```

## 📊 **Performance Impact**

| Parameter | Lower Value | Higher Value |
|-----------|-------------|--------------|
| `peak-sensitivity` | More objects, slower | Fewer objects, faster |
| `sparse-threshold` | More watershed usage | More simple components |
| `dense-threshold` | More dense algorithm | More standard algorithm |
| `fallback-threshold` | More aggressive fallback | More conservative fallback |

## 🎯 **Example Usage**

### **Basic Usage with Custom Parameters**
```bash
python examples/instance_segmentation_example.py \
  --data-dir data/raw \
  --results-dir results/custom_watershed \
  --watershed-peak-sensitivity 0.25 \
  --watershed-dense-threshold 0.35 \
  --verbose
```

### **High-Quality Separation (Slower)**
```bash
python examples/instance_segmentation_example.py \
  --data-dir data/raw \
  --results-dir results/high_quality \
  --watershed-peak-sensitivity 0.2 \
  --watershed-sparse-threshold 0.005 \
  --watershed-dense-threshold 0.2 \
  --num-processes 4
```

### **Fast Processing (Lower Quality)**
```bash
python examples/instance_segmentation_example.py \
  --data-dir data/raw \
  --results-dir results/fast_mode \
  --watershed-peak-sensitivity 0.4 \
  --watershed-sparse-threshold 0.02 \
  --fast-mode
```

## 🔧 **Advanced Configuration**

For even more control, you can modify the hardcoded parameters in the code:

```python
# In create_instance_mask function
default_watershed_config = {
    'multi_resolution_threshold': 512,  # Image size for multi-resolution
    'min_watershed_distance': 3,        # Minimum distance for watershed viability  
    'max_objects_threshold': 50,        # Max objects before algorithm switch
    'gpu_size_threshold': 512*512       # GPU usage threshold
}
```

## 📈 **Monitoring Results**

Use `--verbose` to see which algorithms are being used:
```
🌊 Watershed config: peak_sensitivity=0.3, sparse_threshold=0.01, dense_threshold=0.3
⏱️  create_instance_mask took 0.040s
```

The timing output will help you understand the performance impact of your parameter choices. 