"""
Test script to check ReID similarity using NVIDIA TAO ReIdentificationNet ONNX model.
Model: resnet50_market1501_aicity156.onnx (from NVIDIA NGC catalog)

Model Specifications (from NVIDIA):
- Architecture: ResNet50
- Input: B x 3 x 256 x 128 (NCHW, RGB, ImageNet normalized)
- Output: 256-dimensional embedding vector
- Trained on: Market-1501 + AI City Challenge 2023 datasets
- mAP: 93.7%, Rank-1 Accuracy: 94.8%

USAGE:
------
# Basic usage (compare default test images):
    python test_nvidia_reid.py

# Compare custom images:
    python test_nvidia_reid.py --img1 person_a.jpg --img2 person_b.jpg

# Use CPU instead of CUDA:
    python test_nvidia_reid.py --device cpu

Requirements: pip install onnxruntime-gpu opencv-python numpy
"""
import cv2
import numpy as np
import argparse
import os

# Try ONNX Runtime with GPU support
try:
    import onnxruntime as ort
except ImportError:
    print("Please install onnxruntime: pip install onnxruntime-gpu")
    exit(1)


# ImageNet normalization values
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Preprocess image for NVIDIA TAO ReIdentificationNet.
    
    Steps:
    1. Resize to 256x128 (height x width) - model's fixed input size
    2. Convert BGR to RGB
    3. Normalize to [0, 1]
    4. Apply ImageNet normalization (mean/std)
    5. Convert to NCHW format (1, 3, 256, 128)
    """
    # Resize to model input size: 256 height x 128 width
    img_resized = cv2.resize(img, (128, 256))  # cv2 uses (width, height)
    
    # Convert BGR (OpenCV default) to RGB
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    # Normalize to [0, 1]
    img_float = img_rgb.astype(np.float32) / 255.0
    
    # Apply ImageNet normalization
    img_normalized = (img_float - IMAGENET_MEAN) / IMAGENET_STD
    
    # Convert to NCHW format: (H, W, C) -> (C, H, W) -> (1, C, H, W)
    img_nchw = np.transpose(img_normalized, (2, 0, 1))
    img_batch = np.expand_dims(img_nchw, axis=0).astype(np.float32)
    
    return img_batch


def load_model(model_path: str, device: str = "cuda"):
    """
    Load ONNX model with appropriate execution provider.
    """
    providers = []
    
    if device == "cuda":
        # Try CUDA first, fall back to CPU
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            print("Using CUDA (GPU) for inference")
        else:
            providers = ['CPUExecutionProvider']
            print("CUDA not available, falling back to CPU")
    else:
        providers = ['CPUExecutionProvider']
        print("Using CPU for inference")
    
    session = ort.InferenceSession(model_path, providers=providers)
    
    # Get input/output names
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    print(f"Model loaded: {model_path}")
    print(f"  Input name: {input_name}, shape: {session.get_inputs()[0].shape}")
    print(f"  Output name: {output_name}, shape: {session.get_outputs()[0].shape}")
    
    return session, input_name, output_name


def extract_embedding(session, input_name: str, output_name: str, image_path: str) -> np.ndarray:
    """
    Extract ReID embedding from an image.
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    print(f"  Loaded: {image_path} | Shape: {img.shape}")
    
    # Preprocess
    img_preprocessed = preprocess_image(img)
    
    # Run inference
    outputs = session.run([output_name], {input_name: img_preprocessed})
    embedding = outputs[0].flatten()
    
    # L2 normalize the embedding
    embedding = embedding / np.linalg.norm(embedding)
    
    return embedding


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate cosine similarity between two L2-normalized vectors."""
    return float(np.dot(vec1, vec2))


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Calculate Euclidean distance between two vectors."""
    return float(np.linalg.norm(vec1 - vec2))


def main():
    parser = argparse.ArgumentParser(description="NVIDIA TAO ReID Similarity Test")
    parser.add_argument("--img1", default="test2.png", help="Path to first image")
    parser.add_argument("--img2", default="test6.png", help="Path to second image")
    parser.add_argument("--model", default="resnet50_market1501_aicity156.onnx", help="Path to ONNX model")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device to use")
    args = parser.parse_args()
    
    print("\n" + "="*65)
    print("    NVIDIA TAO ReIdentificationNet - Similarity Test")
    print("="*65)
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"\nERROR: Model file not found: {args.model}")
        print("Download from: https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tao/models/reidentificationnet")
        return
    
    # Load model
    print("\n[1] Loading ONNX model...")
    session, input_name, output_name = load_model(args.model, args.device)
    
    # Extract embeddings
    print("\n[2] Extracting embeddings...")
    emb1 = extract_embedding(session, input_name, output_name, args.img1)
    emb2 = extract_embedding(session, input_name, output_name, args.img2)
    
    print(f"\n  Embedding dimension: {emb1.shape[0]}")
    print(f"  Embedding norm (should be ~1.0): {np.linalg.norm(emb1):.4f}, {np.linalg.norm(emb2):.4f}")
    
    # Calculate similarity
    print("\n[3] Computing similarity...")
    cos_sim = cosine_similarity(emb1, emb2)
    euc_dist = euclidean_distance(emb1, emb2)
    
    print("\n" + "-"*65)
    print("                        RESULTS")
    print("-"*65)
    print(f"\n  Cosine Similarity:   {cos_sim:.4f}")
    print(f"  Euclidean Distance:  {euc_dist:.4f}")
    
    # Interpretation (NVIDIA model is more robust, adjust thresholds)
    print("\n" + "-"*65)
    print("                     INTERPRETATION")
    print("-"*65)
    
    if cos_sim > 0.80:
        result = "HIGH MATCH - Same person"
    elif cos_sim > 0.65:
        result = "MEDIUM MATCH - Possibly same person"
    elif cos_sim > 0.45:
        result = "LOW MATCH - Uncertain"
    else:
        result = "NO MATCH - Different people"
    
    print(f"\n  {result}")
    print(f"\n  Thresholds (NVIDIA TAO model):")
    print(f"    > 0.80 = High match (same person)")
    print(f"    > 0.65 = Medium match (possibly same)")
    print(f"    > 0.45 = Low match (uncertain)")
    print(f"    < 0.45 = No match (different people)")
    
    print("\n" + "="*65 + "\n")
    
    return cos_sim, euc_dist


if __name__ == "__main__":
    main()
