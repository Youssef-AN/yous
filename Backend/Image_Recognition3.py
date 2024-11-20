import torch  # For loading and using the CNN model
import cv2  # For processing video frames and drawing bounding boxes
import numpy as np  # For numerical operations
from ultralytics import YOLO  # For YOLOv8 model for object detection
from fastapi import FastAPI, WebSocket  # For the backend API and WebSocket connection
import base64  # For encoding and decoding image data
from BeanClassifierCNN import BeanClassifierCNN  # Imports your CNN model

from fastapi.middleware.cors import CORSMiddleware

# FastAPI application initialization
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://66.81.175.61:8080"],  # Update this with your exact URL and port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set the quantization engine (adjust if you know which one was used)
torch.backends.quantized.engine = 'qnnpack'  # or 'qnnpack'
# Load the pre-trained CNN model for classification
#cnn_model = BeanClassifierCNN(num_classes=3)  # Define CNN model class with specified classes
#cnn_model.load_state_dict(torch.load('bean_classifier2_cnn.pth'))  # Load quantized model weights from file
cnn_model=torch.load('bean_classifier2_cnn_quantized.pth')
cnn_model.eval()  # Set the model to evaluation mode for inference (no dropout, etc.)

# Load the YOLO model for object detection
yolo_model = YOLO('yolov8n.pt')  # Load YOLO model (YOLOv8 pre-trained or custom model)

# Define the class labels used for classification
class_labels = {0: 'semiripe', 1: 'ripe', 2: 'overripe'}

# Function to perform YOLO detection and CNN classification
def classify_bean(frame):
    """
    Detects beans using YOLO, then classifies each bean using the CNN model.
    Returns bounding boxes, labels, and confidence scores for each detection.
    """
    results = yolo_model(frame)  # Run YOLO model on the frame to detect objects
    predictions = []  # List to store predictions for each detected bean

    # Process detected objects (bounding boxes)
    if results and results[0].boxes is not None:  # Check if there are detected boxes
        for box in results[0].boxes:  # Loop over detected boxes
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Get bounding box coordinates
            bean_image = frame[y1:y2, x1:x2]  # Crop the detected bean area from the frame

            # Skip invalid crops (edge cases where box is out of frame bounds)
            if bean_image.size == 0:
                continue

            # Preprocess the bean image for CNN classification
            bean_image = cv2.resize(bean_image, (64, 64))  # Resize to match CNN input size
            bean_image = np.transpose(bean_image, (2, 0, 1))  # Change to (C, H, W) format
            bean_image = torch.tensor(bean_image, dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize and add batch dimension

            # Perform CNN classification without gradient calculation
            with torch.no_grad():
                output = cnn_model(bean_image)  # Get CNN output
                _, predicted = torch.max(output, 1)  # Get the predicted class ID
                class_id = predicted.item()  # Convert prediction to an integer

            # Get the label and confidence score for the detected bean
            label = class_labels.get(class_id, "Unknown")  # Map class ID to label
            confidence = torch.softmax(output, dim=1).max().item()  # Calculate confidence

            # Append prediction info (label, confidence, and bounding box) to predictions list
            predictions.append({
                "label": label,
                "confidence": confidence,
                "bbox": [x1, y1, x2, y2]
            })

    return predictions  # Return all predictions for the frame

from fastapi import BackgroundTasks
import asyncio

# Define an async version of the classify_bean function
async def classify_bean_async(frame):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, classify_bean, frame)

# WebSocket endpoint to handle real-time video streaming
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint to handle live video frames.
    Receives frames from the client, processes them, and sends back annotated frames.
    """
    await websocket.accept()  # Accept WebSocket connection from client

    try:
        while True:
            # Receive Base64-encoded frame data from client
            data = await websocket.receive_text()  # Get image data as Base64 string
            frame_data = base64.b64decode(data)  # Decode Base64 to raw image data
            nparr = np.frombuffer(frame_data, np.uint8)  # Convert to numpy array
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Decode to OpenCV image

            # Perform object detection and classification on the frame
            predictions = classify_bean(frame)  # Classify detected objects in the frame

            # Annotate the frame with bounding boxes and labels
            # for pred in predictions:
            #     x1, y1, x2, y2 = pred["bbox"]  # Unpack bounding box coordinates
            #     label = pred["label"]  # Get predicted label
            #     confidence = pred["confidence"]  # Get confidence score

            #     # Draw bounding box around detected object
            #     color = (0, 255, 0)  # Green color for bounding box
            #     cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # Draw rectangle on the frame

            #     # Display label and confidence above the bounding box
            #     cv2.putText(
            #         frame, f"{label}: {confidence:.2f}", (x1, y1 - 10),
            #         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            #     )

            # Encode the annotated frame as Base64 for transmission
            _, buffer = cv2.imencode('.jpg', frame)  # Encode frame as JPEG
            encoded_frame = base64.b64encode(buffer).decode('utf-8')  # Encode to Base64 string
            message={
                'frame':encoded_frame,
                'predictions':predictions
            }

            # Send annotated frame back to client
            await websocket.send_text(json.dumps(message))  # Send frame data to client

    except Exception as e:
        print(f"WebSocket connection closed: {e}")  # Log if WebSocket closes or an error occurs

    finally:
        await websocket.close()  # Ensure WebSocket is closed on error or disconnect

#Forwarding                    https://8f4f-165-85-220-33.ngrok-free.app -> http://localhost:8000                                            
#Forwarding                    https://e5de-165-85-220-33.ngrok-free.app -> http://localhost:8080 