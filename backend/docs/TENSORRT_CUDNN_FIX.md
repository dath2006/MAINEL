# TensorRT cuDNN Missing DLL Fix

## Problem

When running `build_tensorrt_engines.py`, you see:

```
Error loading "onnxruntime_providers_tensorrt.dll" which depends on "cudnn64_9.dll" which is missing.
Falling back to ['CPUExecutionProvider'] and retrying.
```

**Result**: Running on CPU (386ms per frame) instead of GPU (8ms per frame).

## Root Cause

ONNX Runtime TensorRT Execution Provider requires **cuDNN 9** DLLs to be accessible in PATH, but they're not installed or not in PATH.

## Solution

### Step 1: Install cuDNN via pip (Already Done ✅)

```powershell
cd D:\MAINEL\backend
.\venv\Scripts\activate
pip install nvidia-cudnn-cu12
```

### Step 2: Find cuDNN DLL Location

```powershell
python -c "import nvidia.cudnn; import os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'bin'))"
```

Example output:

```
D:\MAINEL\backend\venv\Lib\site-packages\nvidia\cudnn\bin
```

### Step 3: Add cuDNN to System PATH

**Option A: Temporary (Current Session Only)**

```powershell
# Replace with your actual path from Step 2
$env:PATH = "D:\MAINEL\backend\venv\Lib\site-packages\nvidia\cudnn\bin;$env:PATH"

# Verify
where.exe cudnn64_9.dll
```

**Option B: Permanent (Recommended)**

1. Copy the path from Step 2
2. Open System Environment Variables:
   - Press `Win + R`
   - Type `sysdm.cpl` and press Enter
   - Click "Advanced" tab → "Environment Variables"
3. Under "System variables", find "Path" and click "Edit"
4. Click "New" and paste the cuDNN bin path
5. Click OK on all dialogs
6. **Restart PowerShell** for changes to take effect

**Option C: Copy DLLs to CUDA Directory**

```powershell
# Find cuDNN DLLs
$cudnnPath = python -c "import nvidia.cudnn; import os; print(os.path.join(os.path.dirname(nvidia.cudnn.__file__), 'bin'))"

# Copy to CUDA bin (requires admin)
Copy-Item "$cudnnPath\cudnn*.dll" "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin\"
```

### Step 4: Verify cuDNN is Found

```powershell
# Should show path to DLL
where.exe cudnn64_9.dll

# Should show path to other cuDNN components
where.exe cudnn_cnn_infer64_9.dll
where.exe cudnn_ops_infer64_9.dll
```

### Step 5: Test TensorRT Again

```powershell
cd D:\MAINEL\backend
.\venv\Scripts\activate

# Rebuild engines with GPU
python scripts/build_tensorrt_engines.py
```

**Expected output (GPU working):**

```
[PeopleNet] Attempting to use TensorrtExecutionProvider with FP16
Session created in 143.5 seconds  # First build is slow (kernel profiling)
First inference: 143.35s
Second inference: 8.23ms  # <-- Should be ~8ms, not 380ms
Average inference latency: 8.19ms
Expected FPS: 122.1  # <-- Should be ~120 FPS, not 2.6 FPS
```

## Verification Checklist

- [ ] `nvidia-cudnn-cu12` installed
- [ ] cuDNN bin path in PATH (verify with `where.exe cudnn64_9.dll`)
- [ ] TensorRT build script runs without "Falling back to CPUExecutionProvider"
- [ ] Inference latency is 8-12ms (not 380ms)
- [ ] Expected FPS is 100+ (not 2.6)

## Alternative: Use CUDA EP Instead of TensorRT

If cuDNN issues persist, you can use CUDA Execution Provider (still GPU-accelerated, just without TensorRT optimizations):

**Temporarily disable TensorRT:**

In `preprocessor/peoplenet_detector.py` and `app/core/features/nvidia_reid_extractor.py`, change:

```python
# Comment out TensorRT provider section
# if has_trt and trt_provider:
#     ...

# Use CUDA directly
if 'CUDAExecutionProvider' in available:
    providers.append('CUDAExecutionProvider')
```

**Expected performance with CUDA EP:**

- Still GPU-accelerated
- 1.5-2x faster than CPU
- Not as fast as TensorRT FP16 (which is 2-3x faster than CUDA)

## Current Status

Based on your terminal output:

- ✅ TensorRT provider is detected
- ✅ CUDA 12.1 installed
- ❌ **cuDNN 9 DLLs not accessible in PATH** → Running on CPU
- ❌ Batch processing not integrated yet

**Next Steps:**

1. Add cuDNN to PATH (see Step 3 above)
2. Restart terminal
3. Run `python scripts/build_tensorrt_engines.py` again
4. Should see 8ms inference (not 380ms)
