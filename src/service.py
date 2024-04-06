from fastapi import FastAPI, UploadFile, File, HTTPException, status, Response
from fastapi.responses import FileResponse
import shutil
import os
import cv2
from ultralytics import YOLO
import pydantic
import json
from src.datacontract.service_config import ServiceConfig
import torch
from src.datacontract.service_output import VideoOutput, Сlass1
import torch
import torchvision.transforms as transforms
from PIL import Image


app = FastAPI()

service_config_path = "./src/configs/service_config.json"
with open(service_config_path, "r") as service_config:
    service_config_json = json.load(service_config)

service_config_adapter = pydantic.TypeAdapter(ServiceConfig)
service_config_python = service_config_adapter.validate_python(service_config_json)


# Initialize YOLO model (replace with your model path)
model = YOLO(service_config_python.path_to_detector)

# Define the directory to save uploaded videos
UPLOAD_DIRECTORY = "./uploaded_videos/"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


def load_classifier(path_to_pth_weights, device):
    model = torch.load(path_to_pth_weights, map_location=device)
    model.eval()
    model.to(device)
    return model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier = load_classifier(service_config_python.path_to_classifier, device)

class_names = ["Aircraft", "Ship"]


@app.post("/video/")
async def process_video(video: UploadFile = File(...)):
    # Save the uploaded video to the upload directory
    video_path = os.path.join(UPLOAD_DIRECTORY, video.filename)

    try:
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        # Process the video asynchronously
        output_video_path = await process_video_file(video_path)

        # Create VideoOutput object with only the video path
        video_output = VideoOutput(video_path=output_video_path)

        return video_output

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e)
        )


async def process_video_file(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Error opening video file"
        )

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    output_video_path = os.path.join(
        UPLOAD_DIRECTORY, "output_" + os.path.basename(video_path)
    )
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB for YOLO
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform object detection
        results = model(frame_rgb)

        # Check if objects are detected
        if len(results) > 0:
            # Loop through each detected object
            for box in results[0].boxes.xyxy:
                xtl, ytl, xbr, ybr = map(int, box.tolist())
                cv2.rectangle(frame, (xtl, ytl), (xbr, ybr), (0, 255, 0), 2)

                # Crop the object from the frame
                crop_object = frame[ytl:ybr, xtl:xbr]

                # Convert the crop to PIL Image
                crop_pil = Image.fromarray(cv2.cvtColor(crop_object, cv2.COLOR_BGR2RGB))

                # Preprocess the crop
                preprocess = transforms.Compose(
                    [
                        transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
                input_tensor = preprocess(crop_pil).unsqueeze(0).to(device)

                # Perform classification
                with torch.no_grad():
                    output = classifier(input_tensor)
                _, predicted_idx = torch.max(output, 1)

                # Get the predicted class label
                predicted_label = class_names[predicted_idx.item()]

                # Draw the predicted label on the frame
                cv2.putText(
                    frame,
                    predicted_label,
                    (xtl, ytl - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                video = Сlass1(class1_arg=predicted_label)

        # Write the frame into the output video
        out.write(frame)

    # Release the video resources
    cap.release()
    out.release()

    return output_video_path, video
