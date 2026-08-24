def predict(self, folder_path):
    """Predicts the contents of all images in a folder."""
    import os

    # Load all images and their paths
    images, image_paths = self.load_images(folder_path)

    predictions = []

    for image, image_path in zip(images, image_paths):
        # Preprocess image
        pimage, image_shape = self.process_image(image)

        # Run YOLO model
        prediction = self.model.predict(pimage)

        # Filter predictions
        boxes, box_classes, box_scores = self.filter_boxes(
            prediction,
            self.anchors,
            self.class_t,
            self.nms_t
        )

        # Convert boxes to original image dimensions
        boxes = self.process_boxes(
            boxes,
            image_shape
        )

        # Save prediction
        predictions.append(
            (boxes, box_classes, box_scores)
        )

        # Get only the filename, without the path
        filename = os.path.basename(image_path)

        # Display image with boxes
        self.show_boxes(
            image,
            boxes,
            box_classes,
            box_scores,
            filename
        )

    return predictions, image_paths
