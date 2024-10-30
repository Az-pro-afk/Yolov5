import cv2
import numpy as np
import socket
import threading
import json
import onnxruntime
from sort import Sort  # Assuming you have a SORT module available
import time  # Import time to control the sending frequency

# Constants for YOLO model
INPUT_WIDTH     = 640
INPUT_HEIGHT    = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD   = 0.45
CONFIDENCE_THRESHOLD = 0.5

# Text parameters
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colors
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)

# Initialize SORT tracker
tracker = Sort()

# Define a global variable to store the frameData
current_frame_data = None

def draw_label(im, label, x, y):
    """Draw text onto image at location."""
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    cv2.rectangle(im, (x,y), (x + dim[0], y + dim[1] + baseline), BLACK, cv2.FILLED)
    cv2.putText(im, label, (x, y + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)

def create_mask(image, top_left=(100, 100), bottom_right=(550, 240)):
# def create_mask(image, top_left=(230, 74), bottom_right=(640, 280)):
    """Create a mask that excludes all regions except the specified rectangular area."""
    height, width = image.shape[:2]
    mask = np.zeros((height, width, 3), dtype=np.uint8)  # Start with a black mask
    
    # Extract coordinates
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    # Ensure coordinates are within image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)
    
    # Apply white mask to the specified rectangular region
    mask[y1:y2, x1:x2] = (255, 255, 255)  # Only the specified region is white
    
    return mask


def show_pixel_value(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:  # Khi di chuyển chuột
        image = param
        pixel_value = image[y, x]

        if len(pixel_value) == 3:
            b, g, r = pixel_value
            print(f"Pixel tại ({x}, {y}): B={b}, G={g}, R={r}")
        else:
            print(f"Pixel tại ({x}, {y}): Grayscale={pixel_value[0]}")

def pre_process(input_image, net):
    """Pre-process the image for YOLO model input."""
    # Create and apply mask
    mask = create_mask(input_image)
    masked_image = cv2.bitwise_and(input_image, mask)
    
 
    # # Show the mask
    # cv2.imshow("Mask", mask)
    # cv2.setMouseCallback("Mask", show_pixel_value, mask)  # Liên kết sự kiện chuột với cửa sổ hiển thị
    # cv2.waitKey(1)  # Display the window briefly to update the mask

    # Convert image to blob
    blob = cv2.dnn.blobFromImage(masked_image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), [0, 0, 0], swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    return outputs

def post_process(input_image, outputs):
    """Process the YOLO model output."""
    class_ids = []
    confidences = []
    boxes = []
    rows = outputs[0].shape[1]
    image_height, image_width = input_image.shape[:2]
    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]
            class_id = np.argmax(classes_scores)
            if classes_scores[class_id] > SCORE_THRESHOLD:
                confidences.append(float(confidence))
                class_ids.append(class_id)
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                left = int((cx - w/2) * x_factor)
                top = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)  
                boxes.append([left, top, width, height])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    
    if isinstance(indices, tuple) and len(indices) > 0:
        indices = indices[0]  # Unpack the tuple

    detections = []
    for i in indices:
        box = boxes[i]
        left, top, width, height = box
        right = left + width  # Calculate the right x-coordinate
        bottom = top + height  # Calculate the bottompython y-coordinate
        label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
        draw_label(input_image, label, left, top)

        # Ensure coordinates are within the image bounds before drawing
        if 0 <= left < image_width and 0 <= top < image_height:
            draw_label(input_image, label, left, top)
            cv2.rectangle(input_image, (left, top), (right, bottom), YELLOW, 2)

        detections.append([left, top, right, bottom, class_ids[i]])  # Store detection with right coordinate
        
    detections.sort(key=lambda x: x[2], reverse=True)

    global tracker
    if len(detections) > 0:
        trackers = tracker.update(np.array(detections))
    else:
        trackers = np.array([])  # Empty array when no detections
    
    global current_frame_data
    current_frame_data = []
    frameData = []
    if len(trackers) > 0:
        iterate = 0
        for tracker_info in trackers:
            tracker_id = int(tracker_info[4])
            bbox = tracker_info[:4]
            class_id = detections[iterate][4]
            frameData.append([bbox, tracker_id, class_id])
            iterate += 1
    else:
        frameData.append([[], -1, -1])
    
    current_frame_data = frameData
    print("frameData: ", current_frame_data)
    return input_image

def start_server_12001(host='127.0.0.1', port=12001):
    """Server to receive images and process with YOLO."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    server_socket.settimeout(45)  # Set a timeout of 5 seconds for accepting connections
    print(f"Server is running at {host}:{port}")
    
    client_socket, client_address = server_socket.accept()
    print(f"Connected to client {client_address}")
    
    data = b''
    while True:
        try:
            chunk = client_socket.recv(16*1024*1024)
            if not chunk:
                print("No more data from client.")
                break
            data = chunk
            nparr = np.frombuffer(data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is not None:  
                detections = pre_process(image, net)
                processed_image = post_process(image.copy(), detections)
                cv2.imshow("Processed Image at Model", processed_image)
                cv2.waitKey(1)
                data = b''
            else:
                print("Failed to decode the image.")
        except KeyboardInterrupt:
            print("Server interrupted by user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

    client_socket.close()
    server_socket.close()
    print("Connection closed at 12001")

def start_server_12002(host='127.0.0.1', port=12002):
    """Server to send current_frame_data."""
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(1)
    server_socket.settimeout(45)  # Set a timeout of 10 seconds for accepting connections
    print(f"Server is running at {host}:{port}")
    
    client_socket, client_address = server_socket.accept()
    print(f"Connected to client {client_address}")
    
    try:
        while True:
            if current_frame_data is not None:
                # Convert numpy arrays to lists and numpy int64 to int before sending
                frame_data_serializable = []
                for item in current_frame_data:
                    bbox = item[0].tolist() if isinstance(item[0], np.ndarray) else item[0]
                    tracker_id = int(item[1])  # Convert numpy int64 to int
                    class_id = int(item[2])  # Convert numpy int64 to int
                    frame_data_serializable.append([bbox, tracker_id, class_id])

                # Send the current_frame_data as JSON
                data_to_send = json.dumps(frame_data_serializable)
                print("sending success")
                client_socket.sendall(data_to_send.encode('utf-8'))
            time.sleep(0.15)  # Control the sending frequency
    except KeyboardInterrupt:
        print("Server interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        client_socket.close()
        server_socket.close()
        print("Connection closed at 12002")

if __name__ == "__main__":
    # Load YOLOv5 model
    classesFile = "coco.names"
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')
    modelWeights = "D:/model/Model_Yolov5s_10_09_2024/runs_5/train/yolov5s_results/weights/last.onnx"
    net = cv2.dnn.readNet(modelWeights)
    
    # Start threads
    thread1 = threading.Thread(target=start_server_12001)
    thread2 = threading.Thread(target=start_server_12002)

    thread1.start()
    thread2.start()

    thread1.join()
    thread2.join()
