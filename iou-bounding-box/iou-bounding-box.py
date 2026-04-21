def iou(box_a, box_b):
    """
    Compute Intersection over Union of two bounding boxes.
    Boxes are in [x1, y1, x2, y2] format.
    """
    # 1. Determine the coordinates of the intersection rectangle
    x_inter1 = max(box_a[0], box_b[0])
    y_inter1 = max(box_a[1], box_b[1])
    x_inter2 = min(box_a[2], box_b[2])
    y_inter2 = min(box_a[3], box_b[3])
    
    # 2. Compute the area of intersection
    # Use max(0, ...) to handle cases where boxes do not overlap
    width_inter = max(0, x_inter2 - x_inter1)
    height_inter = max(0, y_inter2 - y_inter1)
    intersection_area = width_inter * height_inter
    
    # 3. Compute the area of both bounding boxes
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    
    # 4. Compute the area of union
    # Union = Area A + Area B - Intersection
    union_area = area_a + area_b - intersection_area
    
    # 5. Handle zero division (if both boxes have 0 area)
    if union_area == 0:
        return 0.0
        
    return float(intersection_area / union_area)