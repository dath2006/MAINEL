"""
TensorRT Engine Builder

Pre-builds optimized TensorRT FP16 engines from ONNX models for PeopleNet and ReID.
This script should be run during Docker image creation to avoid first-run delays.

Usage:
    python scripts/build_tensorrt_engines.py

Outputs:
    - ./trt_cache/peoplenet/*.engine (PeopleNet TensorRT engine)
    - ./trt_cache/reidnet/*.engine (ReID TensorRT engine)
    - ./trt_cache/timing_cache/*.cache (Shared timing cache)

Expected speedup after prebuilding:
    - First inference: <2 seconds (vs. 2-5 minutes without cache)
    - Subsequent runs: Instant engine load
"""

import os
import sys
from pathlib import Path
import time

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import onnxruntime as ort
import numpy as np
from loguru import logger


class TensorRTEngineBuilder:
    """Build and cache TensorRT engines for ONNX models."""
    
    def __init__(self, cache_base_path: str = "./trt_cache"):
        """
        Initialize TensorRT engine builder.
        
        Args:
            cache_base_path: Base directory for TensorRT cache
        """
        self.cache_base = Path(cache_base_path)
        self.cache_base.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.peoplenet_cache = self.cache_base / "peoplenet"
        self.reidnet_cache = self.cache_base / "reidnet"
        self.timing_cache = self.cache_base / "timing_cache"
        
        for path in [self.peoplenet_cache, self.reidnet_cache, self.timing_cache]:
            path.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"TensorRT cache directory: {self.cache_base.absolute()}")
    
    def build_engine(
        self,
        model_path: str,
        cache_path: Path,
        input_shape: tuple,
        model_name: str,
        fp16: bool = True,
        int8: bool = False,
    ) -> bool:
        """
        Build TensorRT engine for ONNX model using ONNX Runtime TensorRT EP.
        
        Args:
            model_path: Path to ONNX model
            cache_path: Directory to store engine cache
            input_shape: Expected input shape (batch, channels, height, width)
            model_name: Name for logging
            fp16: Enable FP16 precision
            int8: Enable INT8 quantization (requires calibration)
        
        Returns:
            True if successful, False otherwise
        """
        if not Path(model_path).exists():
            logger.error(f"Model not found: {model_path}")
            return False
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Building TensorRT engine for: {model_name}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Cache path: {cache_path}")
        logger.info(f"Input shape: {input_shape}")
        logger.info(f"FP16: {fp16}, INT8: {int8}")
        logger.info(f"{'='*60}\n")
        
        # Configure TensorRT Execution Provider
        providers = []
        
        available = ort.get_available_providers()
        # Note: Provider name can be 'TensorrtExecutionProvider' or 'TensorRTExecutionProvider'
        has_trt = any('tensorrt' in p.lower() for p in available)
        
        if not has_trt:
            logger.error("TensorRT Execution Provider not available!")
            logger.info(f"Available providers: {available}")
            return False
        
        # Find the actual provider name (case-sensitive)
        trt_provider = next((p for p in available if 'tensorrt' in p.lower()), None)
        
        trt_options = {
            'trt_fp16_enable': fp16,
            'trt_int8_enable': int8,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': str(cache_path),
            'trt_max_workspace_size': 4 * 1024 * 1024 * 1024,  # 4GB
            'trt_max_partition_iterations': 1000,
            'trt_min_subgraph_size': 1,
            # Use timing cache to speed up subsequent builds
            'trt_timing_cache_enable': True,
            'trt_timing_cache_path': str(self.timing_cache),
        }
        
        providers.append((trt_provider, trt_options))
        providers.append('CPUExecutionProvider')
        
        try:
            logger.info("Creating ONNX Runtime session with TensorRT EP...")
            start_time = time.time()
            
            # Session options
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            
            session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers
            )
            
            build_time = time.time() - start_time
            logger.info(f"Session created in {build_time:.2f} seconds")
            
            # Get input/output info
            input_name = session.get_inputs()[0].name
            output_names = [o.name for o in session.get_outputs()]
            
            logger.info(f"Input: {input_name}, shape: {session.get_inputs()[0].shape}")
            logger.info(f"Outputs: {output_names}")
            
            # Run warmup inferences to build and cache engine
            logger.info("\nRunning warmup inferences to build TensorRT engine...")
            logger.info("This may take 2-5 minutes on first build (kernel profiling)...")
            
            dummy_input = np.random.randn(*input_shape).astype(np.float32)
            
            # First inference (builds engine)
            logger.info("Warmup 1/3 (building engine)...")
            inference_start = time.time()
            _ = session.run(output_names, {input_name: dummy_input})
            first_inference = time.time() - inference_start
            logger.info(f"  First inference: {first_inference:.2f}s")
            
            # Second inference (cached)
            logger.info("Warmup 2/3 (cached engine)...")
            inference_start = time.time()
            _ = session.run(output_names, {input_name: dummy_input})
            second_inference = time.time() - inference_start
            logger.info(f"  Second inference: {second_inference*1000:.2f}ms")
            
            # Third inference (measure stable latency)
            logger.info("Warmup 3/3 (measuring latency)...")
            inference_start = time.time()
            outputs = session.run(output_names, {input_name: dummy_input})
            third_inference = time.time() - inference_start
            logger.info(f"  Third inference: {third_inference*1000:.2f}ms")
            
            # Calculate average inference time
            avg_inference = (second_inference + third_inference) / 2 * 1000  # Convert to ms
            
            logger.info(f"\n✓ Engine built successfully!")
            logger.info(f"  Output shapes: {[o.shape for o in outputs]}")
            logger.info(f"  Average inference latency: {avg_inference:.2f}ms")
            logger.info(f"  Expected FPS: {1000/avg_inference:.1f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to build TensorRT engine: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def build_peoplenet_engine(self, model_path: str = "model_weights/resnet34_peoplenet.onnx"):
        """Build TensorRT engine for PeopleNet detector."""
        return self.build_engine(
            model_path=model_path,
            cache_path=self.peoplenet_cache,
            input_shape=(1, 3, 544, 960),  # PeopleNet input: 960x544
            model_name="PeopleNet (ResNet34)",
            fp16=True,
            int8=False,  # INT8 for Phase 2
        )
    
    def build_reid_engine(self, model_path: str = "model_weights/resnet50_market1501_aicity156.onnx"):
        """Build TensorRT engine for ReID model."""
        return self.build_engine(
            model_path=model_path,
            cache_path=self.reidnet_cache,
            input_shape=(1, 3, 256, 128),  # ReID input: 256x128
            model_name="ReID (ResNet50)",
            fp16=True,
            int8=False,  # INT8 for Phase 2
        )
    
    def verify_cache(self):
        """Verify that TensorRT engine caches were created."""
        logger.info(f"\n{'='*60}")
        logger.info("Verifying TensorRT cache...")
        logger.info(f"{'='*60}\n")
        
        peoplenet_files = list(self.peoplenet_cache.glob("*"))
        reidnet_files = list(self.reidnet_cache.glob("*"))
        timing_files = list(self.timing_cache.glob("*"))
        
        logger.info(f"PeopleNet cache: {len(peoplenet_files)} files")
        for f in peoplenet_files:
            logger.info(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")
        
        logger.info(f"\nReID cache: {len(reidnet_files)} files")
        for f in reidnet_files:
            logger.info(f"  - {f.name} ({f.stat().st_size / 1024 / 1024:.2f} MB)")
        
        logger.info(f"\nTiming cache: {len(timing_files)} files")
        for f in timing_files:
            logger.info(f"  - {f.name} ({f.stat().st_size / 1024:.2f} KB)")


def main():
    """Build all TensorRT engines."""
    logger.info("TensorRT Engine Builder")
    logger.info("=" * 60)
    logger.info("This script pre-builds optimized TensorRT FP16 engines")
    logger.info("from ONNX models to avoid first-run delays.\n")
    
    # Check TensorRT availability
    available_providers = ort.get_available_providers()
    logger.info(f"Available ONNX Runtime providers: {available_providers}\n")
    
    # Check for TensorRT provider (case-insensitive)
    has_tensorrt = any('tensorrt' in p.lower() for p in available_providers)
    
    if not has_tensorrt:
        logger.error("❌ TensorRT Execution Provider not available!")
        logger.error("Please ensure:")
        logger.error("  1. NVIDIA GPU with CUDA support is available")
        logger.error("  2. TensorRT is installed (comes with CUDA toolkit or pip install tensorrt)")
        logger.error("  3. onnxruntime-gpu is installed (not onnxruntime)")
        sys.exit(1)
    
    logger.info("✓ TensorRT Execution Provider available\n")
    
    # Initialize builder
    builder = TensorRTEngineBuilder()
    
    # Build PeopleNet engine
    success_peoplenet = builder.build_peoplenet_engine()
    
    # Build ReID engine
    success_reid = builder.build_reid_engine()
    
    # Verify caches
    builder.verify_cache()
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Build Summary")
    logger.info(f"{'='*60}")
    logger.info(f"PeopleNet: {'✓ SUCCESS' if success_peoplenet else '❌ FAILED'}")
    logger.info(f"ReID:      {'✓ SUCCESS' if success_reid else '❌ FAILED'}")
    logger.info(f"{'='*60}\n")
    
    if success_peoplenet and success_reid:
        logger.info("🚀 All TensorRT engines built successfully!")
        logger.info("Expected performance improvements:")
        logger.info("  - Detection: 2-3x faster (single frame), 8-12x (batch=8)")
        logger.info("  - ReID: 2.5-5x faster (single crop), 6-16x (batch=32)")
        logger.info("\nNext steps:")
        logger.info("  1. Start backend server: uvicorn app.main:app --reload")
        logger.info("  2. Monitor FPS improvements in logs")
        logger.info("  3. Phase 2: Enable INT8 quantization for additional 1.5x speedup")
        return 0
    else:
        logger.error("❌ Some engines failed to build. Check errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
