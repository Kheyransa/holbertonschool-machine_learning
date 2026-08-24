def show_boxes(self, image, boxes, box_classes, box_scores, file_name):
    """Displays an image with all boundary boxes, class names, and box scores."""
    import cv2
    import os

    # Make a copy so the original image is not modified
    image_copy = image.copy()

    # Draw every bounding box
    for box, box_class, box_score in zip(boxes, box_classes, box_scores):
        x1, y1, x2, y2 = box.astype(int)

        # Draw bounding box in blue
        cv2.rectangle(
            image_copy,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        # Get class name
        class_name = self.class_names[box_class]

        # Create label
        label = "{} {:.2f}".format(class_name, box_score)

        # Text position: 5 pixels above top-left corner
        text_x = x1
        text_y = y1 - 5

        cv2.putText(
            image_copy,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA
        )

    # Display image
    cv2.imshow(file_name, image_copy)

    # Wait for a key
    key = cv2.waitKey(0) & 0xFF

    # Save if 's' is pressed
    if key == ord('s'):
        if not os.path.exists('detections'):
            os.makedirs('detections')

        cv2.imwrite(
            os.path.join('detections', file_name),
            image_copy
        )

    # Close the window
    cv2.destroyAllWindows()
