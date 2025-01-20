from ultralytics import YOLO
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def main():
    model = YOLO("yolov8n.yaml")  # The object detection framework
    model.train(
        optimizer="Adam",  # Stochastic gradient descent optimizer
        data='D:\PR_7\Sintez1\data_wl.yaml',  # Path to the training data
        epochs=40,  # Number of training epochs
        batch=15,  # Batch size
        imgsz=640,
        device='cpu',  # Device to train on
        workers=2,  # Number of data loading workers
        patience=0,  # Number of epochs to wait for improvement (early stopping)
        resume=1,  # Resume training from epoch
        lr0=0.01,
        pretrained=False  # Use pretrained model weights
    )


if __name__ == '__main__':
    main()
