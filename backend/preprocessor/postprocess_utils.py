"""
Utility functions for DetectNet_v2 post-processing.
Based on NVIDIA forum reference by m.fiore:
https://forums.developer.nvidia.com/t/run-peoplenet-with-tensorrt/128000/22
"""

import numpy as np


# Model constants
MODEL_H = 544
MODEL_W = 960
STRIDE = 16
BOX_NORM = 35.0
GRID_H = MODEL_H // STRIDE  # 34
GRID_W = MODEL_W // STRIDE  # 60
GRID_SIZE = GRID_H * GRID_W

# Precompute grid centers (normalized by box_norm)
GRID_CENTERS_W = [(i * STRIDE + 0.5) / BOX_NORM for i in range(GRID_W)]
GRID_CENTERS_H = [(i * STRIDE + 0.5) / BOX_NORM for i in range(GRID_H)]


def iou_vectorized(boxes: np.ndarray) -> np.ndarray:
    """
    Compute pairwise IOU matrix for a set of boxes.
    
    Args:
        boxes: Array of shape (N, 4) with boxes in [x1, y1, x2, y2] format
        
    Returns:
        iou_matrix: Array of shape (N, N) with pairwise IOU values
    """
    n = len(boxes)
    if n == 0:
        return np.array([])
    
    # Extract coordinates
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    # Compute areas
    areas = (x2 - x1) * (y2 - y1)
    
    # Create meshgrid for pairwise comparisons
    x1_i, x1_j = np.meshgrid(x1, x1)
    y1_i, y1_j = np.meshgrid(y1, y1)
    x2_i, x2_j = np.meshgrid(x2, x2)
    y2_i, y2_j = np.meshgrid(y2, y2)
    areas_i, areas_j = np.meshgrid(areas, areas)
    
    # Compute intersection
    inter_x1 = np.maximum(x1_i, x1_j)
    inter_y1 = np.maximum(y1_i, y1_j)
    inter_x2 = np.minimum(x2_i, x2_j)
    inter_y2 = np.minimum(y2_i, y2_j)
    
    inter_w = np.maximum(0, inter_x2 - inter_x1)
    inter_h = np.maximum(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    # Compute union
    union_area = areas_i + areas_j - inter_area
    
    # Compute IOU
    iou_matrix = np.where(union_area > 0, inter_area / union_area, 0)
    
    return iou_matrix


def apply_box_norm(o1: float, o2: float, o3: float, o4: float, w: int, h: int) -> tuple:
    """
    Apply the GridNet box normalization.
    
    This is the core denormalization formula from NVIDIA's DetectNet_v2.
    
    Args:
        o1: x1 raw output from model
        o2: y1 raw output from model  
        o3: x2 raw output from model
        o4: y2 raw output from model
        w: column index on the grid (0 to GRID_W-1)
        h: row index on the grid (0 to GRID_H-1)
        
    Returns:
        Tuple of (x1, y1, x2, y2) in absolute pixel coordinates
    """
    # Formula from m.fiore's post on NVIDIA forum:
    # o1 = (o1 - grid_centers_w[x]) * -box_norm
    # o2 = (o2 - grid_centers_h[y]) * -box_norm
    # o3 = (o3 + grid_centers_w[x]) * box_norm
    # o4 = (o4 + grid_centers_h[y]) * box_norm
    x1 = (o1 - GRID_CENTERS_W[w]) * -BOX_NORM
    y1 = (o2 - GRID_CENTERS_H[h]) * -BOX_NORM
    x2 = (o3 + GRID_CENTERS_W[w]) * BOX_NORM
    y2 = (o4 + GRID_CENTERS_H[h]) * BOX_NORM
    return x1, y1, x2, y2


def postprocess_detectnet(
    output_bbox: np.ndarray,
    output_cov: np.ndarray,
    num_classes: int = 3,
    min_confidence: float = 0.4,
    analysis_classes: list = None,
    orig_width: int = None,
    orig_height: int = None
) -> list:
    """
    Postprocess DetectNet_v2 outputs to get bounding boxes.
    
    Based on m.fiore's reference implementation from NVIDIA forum.
    
    Args:
        output_bbox: Bounding box tensor (1, num_classes*4, grid_h, grid_w) or flattened
        output_cov: Coverage tensor (1, num_classes, grid_h, grid_w) or flattened
        num_classes: Number of detection classes
        min_confidence: Minimum confidence threshold
        analysis_classes: List of class indices to process (default: all)
        orig_width: Original image width for scaling
        orig_height: Original image height for scaling
        
    Returns:
        List of dictionaries with keys: 'x1', 'y1', 'x2', 'y2', 'confidence', 'class_id'
    """
    if analysis_classes is None:
        analysis_classes = list(range(num_classes))
    
    # Flatten tensors if needed
    if output_bbox.ndim == 4:
        # Shape: (1, C*4, H, W) -> flatten to (C*4*H*W,)
        boxes = output_bbox[0].transpose(1, 2, 0).flatten()  # Reorder to (H, W, C*4) then flatten
    else:
        boxes = output_bbox.flatten()
    
    if output_cov.ndim == 4:
        # Shape: (1, C, H, W) -> flatten to (C*H*W,)
        cov = output_cov[0].transpose(1, 2, 0).flatten()  # Reorder to (H, W, C) then flatten
    else:
        cov = output_cov.flatten()
    
    # Scale factors for original image size
    scale_x = (orig_width / MODEL_W) if orig_width else 1.0
    scale_y = (orig_height / MODEL_H) if orig_height else 1.0
    
    detections = []
    
    for c in analysis_classes:
        # Index offsets for this class's bbox channels
        # The bbox tensor is organized as: [x1_class0, y1_class0, x2_class0, y2_class0, x1_class1, ...]
        # Each channel has grid_size elements
        x1_idx = c * 4 * GRID_SIZE
        y1_idx = x1_idx + GRID_SIZE
        x2_idx = y1_idx + GRID_SIZE
        y2_idx = x2_idx + GRID_SIZE
        
        for h in range(GRID_H):
            for w in range(GRID_W):
                i = w + h * GRID_W
                
                # Check confidence for this grid cell and class
                cov_idx = c * GRID_SIZE + i
                if cov[cov_idx] >= min_confidence:
                    # Get raw bbox values
                    o1 = boxes[x1_idx + i]
                    o2 = boxes[y1_idx + i]
                    o3 = boxes[x2_idx + i]
                    o4 = boxes[y2_idx + i]
                    
                    # Apply box normalization to get absolute coordinates
                    x1, y1, x2, y2 = apply_box_norm(o1, o2, o3, o4, w, h)
                    
                    # Scale to original image size
                    x1 = x1 * scale_x
                    y1 = y1 * scale_y
                    x2 = x2 * scale_x
                    y2 = y2 * scale_y
                    
                    # Validate box
                    if x2 > x1 and y2 > y1:
                        detections.append({
                            'x1': x1,
                            'y1': y1,
                            'x2': x2,
                            'y2': y2,
                            'confidence': float(cov[cov_idx]),
                            'class_id': c
                        })
    
    return detections


def postprocess_detectnet_vectorized(
    output_bbox: np.ndarray,
    output_cov: np.ndarray,
    num_classes: int = 3,
    min_confidence: float = 0.4,
    analysis_classes: list = None,
    orig_width: int = None,
    orig_height: int = None
) -> list:
    """
    Vectorized version of postprocess_detectnet for better performance.
    
    Args:
        output_bbox: Bounding box tensor (1, num_classes*4, grid_h, grid_w)
        output_cov: Coverage tensor (1, num_classes, grid_h, grid_w)
        num_classes: Number of detection classes
        min_confidence: Minimum confidence threshold
        analysis_classes: List of class indices to process (default: all)
        orig_width: Original image width for scaling
        orig_height: Original image height for scaling
        
    Returns:
        List of dictionaries with detection results
    """
    if analysis_classes is None:
        analysis_classes = list(range(num_classes))
    
    # Scale factors
    scale_x = (orig_width / MODEL_W) if orig_width else 1.0
    scale_y = (orig_height / MODEL_H) if orig_height else 1.0
    
    # Create grid coordinate arrays
    grid_w_arr = np.array(GRID_CENTERS_W)  # Shape: (GRID_W,)
    grid_h_arr = np.array(GRID_CENTERS_H)  # Shape: (GRID_H,)
    
    # Create meshgrid of grid centers
    gc_w, gc_h = np.meshgrid(grid_w_arr, grid_h_arr)  # Both shape: (GRID_H, GRID_W)
    
    detections = []
    
    for c in analysis_classes:
        # Get coverage for this class
        cov = output_cov[0, c, :, :]  # Shape: (GRID_H, GRID_W)
        
        # Debug: Show max coverage for each class
        max_cov = float(cov.max())
        class_names = ['person', 'bag', 'face']
        # if c < len(class_names):
        #     print(f"[PostProcess] Class {c} ({class_names[c]}): max_cov={max_cov:.4f}, threshold={min_confidence}")
        
        # Find grid cells above threshold
        mask = cov >= min_confidence
        if not np.any(mask):
            continue
        
        # Get bbox channels for this class
        o1 = output_bbox[0, c*4 + 0, :, :]  # x1
        o2 = output_bbox[0, c*4 + 1, :, :]  # y1
        o3 = output_bbox[0, c*4 + 2, :, :]  # x2
        o4 = output_bbox[0, c*4 + 3, :, :]  # y2
        
        # Apply box normalization (vectorized)
        x1 = (o1 - gc_w) * -BOX_NORM
        y1 = (o2 - gc_h) * -BOX_NORM
        x2 = (o3 + gc_w) * BOX_NORM
        y2 = (o4 + gc_h) * BOX_NORM
        
        # Scale to original image
        x1 = x1 * scale_x
        y1 = y1 * scale_y
        x2 = x2 * scale_x
        y2 = y2 * scale_y
        
        # Extract valid detections
        valid_x1 = x1[mask]
        valid_y1 = y1[mask]
        valid_x2 = x2[mask]
        valid_y2 = y2[mask]
        valid_cov = cov[mask]
        
        for i in range(len(valid_x1)):
            if valid_x2[i] > valid_x1[i] and valid_y2[i] > valid_y1[i]:
                detections.append({
                    'x1': float(valid_x1[i]),
                    'y1': float(valid_y1[i]),
                    'x2': float(valid_x2[i]),
                    'y2': float(valid_y2[i]),
                    'confidence': float(valid_cov[i]),
                    'class_id': c
                })
    
    return detections


def nms(detections: list, iou_threshold: float = 0.5) -> list:
    """
    Apply Non-Maximum Suppression to detection results.
    
    Args:
        detections: List of detection dictionaries
        iou_threshold: IOU threshold for suppression
        
    Returns:
        Filtered detections after NMS
    """
    if not detections:
        return []
    
    # Convert to numpy arrays
    boxes = np.array([[d['x1'], d['y1'], d['x2'], d['y2']] for d in detections])
    scores = np.array([d['confidence'] for d in detections])
    class_ids = np.array([d['class_id'] for d in detections])
    
    # Apply NMS per class
    keep_indices = []
    
    for c in np.unique(class_ids):
        class_mask = class_ids == c
        class_indices = np.where(class_mask)[0]
        class_boxes = boxes[class_mask]
        class_scores = scores[class_mask]
        
        # Sort by score descending
        order = np.argsort(-class_scores)
        
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(class_indices[i])
            
            if len(order) == 1:
                break
            
            # Compute IOU with remaining boxes
            xx1 = np.maximum(class_boxes[i, 0], class_boxes[order[1:], 0])
            yy1 = np.maximum(class_boxes[i, 1], class_boxes[order[1:], 1])
            xx2 = np.minimum(class_boxes[i, 2], class_boxes[order[1:], 2])
            yy2 = np.minimum(class_boxes[i, 3], class_boxes[order[1:], 3])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            inter = w * h
            area_i = (class_boxes[i, 2] - class_boxes[i, 0]) * (class_boxes[i, 3] - class_boxes[i, 1])
            area_others = (class_boxes[order[1:], 2] - class_boxes[order[1:], 0]) * \
                          (class_boxes[order[1:], 3] - class_boxes[order[1:], 1])
            
            iou = inter / (area_i + area_others - inter + 1e-6)
            
            # Keep boxes with IOU below threshold
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        keep_indices.extend(keep)
    
    # Sort by confidence for consistent output
    keep_indices = sorted(keep_indices, key=lambda i: detections[i]['confidence'], reverse=True)
    
    return [detections[i] for i in keep_indices]
