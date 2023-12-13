import cv2
import torch
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms

def valToChar(v):
    return chr(v + 64)

def get_prediction(frame, threshold=0.5):
    # Transform the frame to tensor
    #frame_tensor = transform(frame).unsqueeze(0)
    frame_tensor = data_transform(frame).unsqueeze(0).to(device)

    with torch.no_grad():
        prediction = model(frame_tensor)

    # Filter out predictions with low scores
    #pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in prediction[0]['boxes']]
    #pred_scores = prediction[0]['scores']
    #boxes = [box for box, score in zip(pred_boxes, pred_scores) if score > threshold]

    # find the best box
    pred_boxes = prediction[0]['boxes'].cpu().numpy()
    pred_scores = prediction[0]['scores'].cpu().numpy()
    best_score = 0
    best_idx = -1
    for i, score in enumerate(pred_scores):
        if score > best_score:
            best_score = score
            best_idx = i
    if best_idx == -1:
        return [], -12
    pred_box = [pred_boxes[best_idx]]

    class_name = prediction[0]['labels'][best_idx].cpu().numpy()

    return pred_box, class_name

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

data_transform = transforms.Compose([
    transforms.ToTensor(),
])

# Load the model
model = fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = 27
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load('models/trained_model.pth')) # change this to model of your choice
model.to(device)
model.eval()

# Initialize webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to PIL image and then to tensor
    frame_pil = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_pil)

    # Get predictions
    boxes, class_name = get_prediction(frame_pil)

    # Draw the boxes
    #print(boxes)
    #print(class_name)
    cord_0 = 0
    cord_1 = 0
    for box in boxes:
     #   print(box)
        #print(box[0])
        cord_0 = int(box[0])
        cord_1 = int(box[1])
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]),int(box[3])), color=(0, 255, 0), thickness=2)

    # Add class name and confidence
   # cv2.putText(frame, class_name, (box[0][0], box[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.putText(frame, valToChar(class_name), (cord_0, cord_1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Webcam - Faster R-CNN', frame)

    # Break loop with 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
