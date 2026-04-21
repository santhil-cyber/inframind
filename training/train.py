from ultralytics import YOLO
import os

def main():
    """
    Training pipeline for the Inframind YOLOv8 Crack Detection model.
    """
    print("Initializing YOLOv8 training pipeline...")

    # Load a pre-trained model (recommended for custom data to enable transfer learning)
    # yolov8n.pt (Nano) gives the fastest training times and is great for edge deployment.
    # yolov8x.pt (Extra Large) gives the highest accuracy but requires a powerful GPU.
    model = YOLO('yolov8n.pt') 

    # Ensure dataset path is absolute or correct relative to the execution
    dataset_yaml = os.path.abspath('crack_dataset.yaml')
    
    if not os.path.exists(dataset_yaml):
        print(f"Error: Could not find {dataset_yaml}. Please ensure it exists.")
        return

    print("Starting training process. This may take several hours depending on your hardware.")
    
    # Train the model with advanced augmentations for structural defects
    # Adjust 'device' to 'cpu' if you don't have a GPU, or a specific GPU index like '0'
    results = model.train(
        data=dataset_yaml,
        epochs=100,             # Number of epochs
        imgsz=640,              # Image input size
        batch=16,               # Batch size
        name='crack_detection_v1',
        device='auto',          # Auto-selects GPU if available, otherwise CPU
        patience=20,            # Early stopping patience
        save=True,              # Save checkpoints
        cache=True,             # Cache images to RAM for faster training
        # Data Augmentations specific for concrete/structure variance:
        augment=True,
        degrees=15.0,           # Rotate images
        flipud=0.5,             # Flip Up-Down (Cracks can be any direction)
        fliplr=0.5,             # Flip Left-Right
        mosaic=1.0,             # Mosaic augmentation (helps detect smaller cracks)
        hsv_h=0.015,            # Minor color augmentation
        hsv_s=0.7,
        hsv_v=0.4
    )
    
    print("Training complete!")

    # Evaluate the model on the validation dataset
    print("Evaluating model performance on validation set...")
    metrics = model.val()
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")

    print("Pipeline finished successfully. The best model is saved in the 'runs/detect/crack_detection_v1/weights/' directory.")

if __name__ == '__main__':
    main()
