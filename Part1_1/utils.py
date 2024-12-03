import matplotlib.pyplot as plt

def plot_image_with_points(ax, img_array, points, title):
    ax.imshow(img_array)
    ax.scatter(points[:, 0], points[:, 1], c='red', marker='o')
    for i, (x, y) in enumerate(points):
        ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(0,10), ha='center')
    ax.set_title(title)

def plot_img_with_yolo(ax, img_array, yolo_data, yolo_classes):
    
    bounding_boxes = yolo_data['xyxy']
    class_ids = yolo_data['class'].flatten().astype(int).astype(str)
    object_ids = yolo_data['id'].flatten().astype(int).astype(str)

    # Plot the image with bounding boxes
    ax.imshow(img_array)
    
    # Plot each bounding box
    for bbox, class_id, object_id in zip(bounding_boxes, class_ids, object_ids):
        x_min, y_min, x_max, y_max = bbox
        width, height = x_max - x_min, y_max - y_min
        rect = plt.Rectangle((x_min, y_min), width, height, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(x_min, y_min, f'{object_id}\nclass: {yolo_classes[class_id]}', color='red', fontsize=12, weight='bold')
    
    ax.set_title('Image with YOLO Bounding Boxes')
    return ax