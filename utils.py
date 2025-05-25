import cv2
import threading
import numpy as np
import os
import time
import mysql.connector
from mysql.connector import Error
from ultralytics import YOLO
from deepface import DeepFace
from numpy.linalg import norm
from datetime import datetime
import pickle

# Configure database
DB = {
    'host': 'localhost',
    'database': 'aimsglob_vms',
    'user': 'root',
    'password': ''
}

# Set up YOLO
model = YOLO('yolov8n.pt')

# Directories to scan with separate cache files
known_faces_dirs = {
    "contractor": {
        "path": "C:/xampp/htdocs/vms/uploads/contractor_selfie",
        "cache_file": "contractor_faces_cache.pkl",
        "db": {}
    },
    "visitor": {
        "path": "C:/xampp/htdocs/vms/uploads/visitor_selfie",
        "cache_file": "visitor_faces_cache.pkl",
        "db": {}
    }
}

# Combined database of all face embeddings
combined_faces_db = {}

progress = {"current": 0, "total": 0, "status": "idle"}

# Default variables for camera stream
camera = None
camera_lock = threading.Lock()
frame_lock = threading.Lock()
current_frame = None
face_data = []
streaming = False
processing_active = False

active_role_filter = None

# Last known state of directories
last_known_directory_state = {}

# DB connection function
def get_db_connection():
    try:
        conn = mysql.connector.connect(**DB)
        return conn
    except Error as e:
        print(f"Database connection error: {e}")
        return None
    
# Function to calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    return np.dot(embedding1, embedding2)/(norm(embedding1)*norm(embedding2))

# Function to load known faces from directories with separate caches
def load_known_faces():
    global progress, combined_faces_db
    progress["current"] = 0
    progress["status"] = "loading"
    combined_faces_db = {}
    
    all_files = []
    
    # Count total files across all directories for progress tracking
    for category, dir_info in known_faces_dirs.items():
        dir_path = dir_info["path"]
        if not os.path.exists(dir_path):
            print(f"Directory does not exist: {dir_path}")
            continue
            
        for filename in os.listdir(dir_path):
            if filename.endswith(('.jpg', '.jpeg', '.png', '.webp')):
                all_files.append((category, dir_path, filename))
    
    progress["total"] = len(all_files)
    
    # Process each directory with its own cache
    for category, dir_info in known_faces_dirs.items():
        dir_path = dir_info["path"]
        cache_file = dir_info["cache_file"]
        
        # Load cache if exists
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                dir_info["db"] = pickle.load(f)
            print(f"Loaded {category} embeddings from cache: {len(dir_info['db'])} faces")
        else:
            dir_info["db"] = {}
            
    # Process any new files
    new_faces_added = {category: False for category in known_faces_dirs}
    
    for category, dir_path, filename in all_files:
        user_id = filename.split('.')[0]
        dir_info = known_faces_dirs[category]
        
        # Skip if already in cache
        if user_id in dir_info["db"]:
            progress["current"] += 1
            continue
        
        image_path = os.path.join(dir_path, filename)
        try:
            embedding = DeepFace.represent(image_path, model_name="Facenet", enforce_detection=False)
            if embedding:
                dir_info["db"][user_id] = embedding[0]["embedding"]
                print(f"Loaded embedding for {category}/{filename}")
                new_faces_added[category] = True
        except Exception as e:
            print(f"Error processing {filename} from {dir_path}: {e}")
        finally:
            progress["current"] += 1
    
    # Save updated caches
    for category, dir_info in known_faces_dirs.items():
        if new_faces_added[category]:
            with open(dir_info["cache_file"], "wb") as f:
                pickle.dump(dir_info["db"], f)
            print(f"Updated {category} face embeddings cache with {len(dir_info['db'])} faces")
    
    # Combine all databases for search efficiency
    for category, dir_info in known_faces_dirs.items():
        # Add category prefix to user_id to track source
        for user_id, embedding in dir_info["db"].items():
            combined_key = f"{category}_{user_id}"
            combined_faces_db[combined_key] = {
                "embedding": embedding,
                "user_id": user_id,
                "category": category
            }
    
    print(f"Combined face database contains {len(combined_faces_db)} faces")
    progress["status"] = "done"

# Camera thread function
def camera_thread():
    global current_frame, camera, streaming

    while streaming:
        if camera is None:
            try:
                with camera_lock:
                    camera = cv2.VideoCapture(0)
                    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    camera.set(cv2.CAP_PROP_FPS, 30)
                    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            except Exception as e:
                print(f"Camera Initialization Error: {e}")
                time.sleep(1)
                continue

        success, frame = camera.read()
        if not success:
            with camera_lock:
                camera.release()
                camera = None
            time.sleep(1)
            continue

        with frame_lock:
            current_frame = frame.copy()

        time.sleep(0.03)

# Function for face recognition
def face_recognition_thread():
    global current_frame, face_data, processing_active

    while processing_active:
        with frame_lock:
            if current_frame is None:
                time.sleep(0.1)
                continue
            frame_to_process = current_frame.copy()

        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame_to_process, cv2.COLOR_BGR2RGB)

        # Detect face using YOLO
        results = model(frame_rgb)
        new_face_data = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Add padding to improve face detection
                padding = 20
                x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
                x2, y2 = min(frame_rgb.shape[1], x2 + padding), min(frame_rgb.shape[0], y2 + padding)

                # Extract face ROI
                face_crop = frame_rgb[y1:y2, x1:x2]

                # Validate face size before processing
                h, w, _ = face_crop.shape
                if h < 50 or w < 50:
                    print("Face too small to process")
                    continue

                # Save temporary image for DeepFace
                temp_face_path = "temp_face.jpg"
                cv2.imwrite(temp_face_path, face_crop)

                try:
                    # Verify if DeepFace detects a face
                    detected_face = DeepFace.extract_faces(temp_face_path, detector_backend="opencv")
                    if detected_face is None:
                        print("No face detected!")
                        continue

                    # Extract embedding
                    detected_embedding = DeepFace.represent(temp_face_path, model_name="Facenet", enforce_detection=False)
                    if not detected_embedding:
                        continue
                    detected_embedding = detected_embedding[0]["embedding"]
                    print("Face embedding: ", detected_embedding)

                    # Compare with known faces using cosine similarity
                    best_match = {"name": "Tidak Diketahui", "similarity": 0, "user_id": None, "category": None}  # Default: unknown

                    for combined_key, face_info in combined_faces_db.items():
                        # Skip non-matching roles if filter is active
                        if active_role_filter and face_info["category"] != active_role_filter:
                            continue

                        known_embedding = face_info["embedding"]
                        similarity = cosine_similarity(detected_embedding, known_embedding)
                        if similarity > 0.65 and similarity > best_match["similarity"]:
                            best_match = {
                                "name": face_info["user_id"], 
                                "similarity": similarity, 
                                "user_id": face_info["user_id"],
                                "category": face_info["category"]
                            }

                    print(f"Detected face match: {best_match['name']} ({best_match['category'] if best_match['category'] else 'Unknown'}) (Similarity: {best_match['similarity']:.2f})")
                    new_face_data.append({
                        "name": best_match["name"], 
                        "similarity": float(best_match["similarity"]),
                        "user_id": best_match["user_id"],
                        "category": best_match["category"],
                        "box": [x1, y1, x2, y2]
                    })

                except Exception as e:
                    print(f"Error processing face: {e}")
                    new_face_data.append({
                        "name": "Tidak Diketahui", 
                        "similarity": 0, 
                        "user_id": None,
                        "category": None,
                        "box": [x1, y1, x2, y2]
                    })

        with frame_lock:
            face_data = new_face_data

        time.sleep(0.05)

def generate_frames():
    global current_frame, face_data
    
    while streaming:
        with frame_lock:
            if current_frame is None:
                time.sleep(0.1)
                continue
                
            display_frame = current_frame.copy()
            
            # Draw face bounding boxes and labels
            for face in face_data:
                x1, y1, x2, y2 = face["box"]
                name = face["name"]
                similarity = face["similarity"]
                category = face["category"]
                
                # Set color based on category and match confidence
                if name == "Tidak Diketahui":
                    color = (0, 0, 255)  # Red for unknown
                elif category == "contractor":
                    color = (255, 165, 0)  # Orange for contractors
                elif category == "visitor":
                    color = (0, 255, 0)  # Green for visitors
                else:
                    color = (255, 0, 255)  # Magenta for any other category
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                label = f"{name} ({category if category else 'Unknown'}, {similarity:.2f})"
                cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame_bytes = buffer.tobytes()
        
        # Yield frame in multipart response
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.03)  # ~30 FPS

def get_meetings():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get current datetime
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    cursor.execute("SELECT id, title FROM meetingdetails WHERE datetime_startlink <= %s AND datetime_endlink >= %s", 
                  (current_time, current_time))
    
    meetings = cursor.fetchall()
    cursor.close()
    conn.close()
    return meetings

def get_locations():
    conn = get_db_connection()
    cursor = conn.cursor()

    # Get all locations
    cursor.execute("SELECT id, name FROM location")

    locations = cursor.fetchall()
    cursor.close()
    conn.close()
    return locations

# Function to check if there are new images uploaded
def initialize_directory_state():
    global last_known_directory_state
    last_known_directory_state = {}

    for category, dir_info in known_faces_dirs.items():
        dir_path = dir_info["path"]
        if os.path.exists(dir_path):
            files = [f for f in os.listdir(dir_path) if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            last_known_directory_state[dir_path] = set(files)
        else:
            last_known_directory_state[dir_path] = set()

    print("Initialized directory state:", {k: len(v) for k, v in last_known_directory_state.items()})

def check_new_images():
    global last_known_directory_state
    new_images = False
    new_images_count = {category: 0 for category in known_faces_dirs}
    total_new = 0

    for category, dir_info in known_faces_dirs.items():
        dir_path = dir_info["path"]
        if not os.path.exists(dir_path):
            continue

        current_files = set([f for f in os.listdir(dir_path) if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))])

        if dir_path not in last_known_directory_state:
            last_known_directory_state[dir_path] = current_files
            continue

        new_files = current_files - last_known_directory_state[dir_path]
        if new_files:
            new_images = True
            new_images_count[category] = len(new_files)
            total_new += len(new_files)
            print(f"Found {len(new_files)} new {category} images in {dir_path}")

        last_known_directory_state[dir_path] = current_files

    return new_images, new_images_count, total_new

def directory_monitor_thread():
    initialize_directory_state()

    while True:
        new_images, count_by_category, total_count = check_new_images()

        if new_images:
            global new_images_notification
            new_images_notification = {
                "active": True,
                "count": total_count,
                "count_by_category": count_by_category,
                "timestamp": time.time()
            }
            
            # Reload face embeddings if new images found
            load_known_faces()

        time.sleep(30)

new_images_notification = {
    "active": False,
    "count": 0,
    "count_by_category": {},
    "timestamp": 0
}

def start_directory_monitor():
    monitor_thread = threading.Thread(target=directory_monitor_thread)
    monitor_thread.daemon = True
    monitor_thread.start()
    print("Directory monitoring started")

def reset_notification():
    global new_images_notification
    new_images_notification = {
        "active": False,
        "count": 0,
        "count_by_category": {},
        "timestamp": 0
    }

def get_face_stats():
    stats = {}
    for category, dir_info in known_faces_dirs.items():
        stats[category] = len(dir_info["db"])
    stats["total"] = len(combined_faces_db)
    
    return stats