from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from flask_cors import CORS
import threading
from datetime import datetime
import utils

app = Flask(__name__)
CORS(app)

@app.route('/')
def start_loading():
    threading.Thread(target=utils.load_known_faces).start()
    return render_template('loading.html')

@app.route('/progress')
def get_progress():
    return jsonify(utils.progress)

@app.route('/home')
def select_meeting():
    meetings = utils.get_meetings()
    locations = utils.get_locations()
    return render_template('select_meeting.html', meetings=meetings, locations=locations)

@app.route('/index')
def index():
    # Get the GET data
    meeting_id = request.args.get('meeting_id', '')
    location_id = request.args.get('location_id', '')
    role = request.args.get('role', '')
    venue = request.args.get('venue', '')
    meeting_details = None
    location_details = None

    conn = utils.get_db_connection()
    cursor = conn.cursor()

    # Get meeting details
    cursor.execute("SELECT id, title FROM meetingdetails WHERE id=%s", (meeting_id,))
    meeting_details = cursor.fetchone()
    print(meeting_details)

    # Get location details
    cursor.execute("SELECT id, name FROM location WHERE id=%s", (location_id,))
    location_details = cursor.fetchone()
    print(location_details)

    cursor.close()
    conn.close()
    
    return render_template('index.html', 
                           meeting_details=meeting_details, 
                           location_details=location_details,
                           meeting_id=meeting_id, 
                           location_id=location_id,
                           role=role,
                           venue=venue)

# Route: Fetch venues by meeting_id (AJAX)
@app.route('/get_venues/<int:meeting_id>/<int:location_id>')
def get_venues(meeting_id, location_id):
    conn = utils.get_db_connection()
    cursor = conn.cursor(dictionary=True)

    query = "SELECT venue FROM meeting_locations WHERE meeting_id = %s AND location_id = %s"
    cursor.execute(query, (meeting_id, location_id))
    venues = cursor.fetchall()

    cursor.close()
    conn.close()

    return jsonify(venues)

# Route to start stream
@app.route('/api/start_stream', methods=['POST'])
def start_stream():
    data = request.get_json()
    role = data.get("role")  # 0 = Visitor, 1 = Contractor

    if role == "0":
        utils.active_role_filter = "visitor"
    elif role == "1":
        utils.active_role_filter = "contractor"
    else:
        utils.active_role_filter = None

    if not utils.streaming:
        utils.streaming = True
        utils.processing_active = True

        # Start camera thread
        stream_thread = threading.Thread(target=utils.camera_thread)
        stream_thread.daemon = True
        stream_thread.start()

        # Start face recognition
        recognition_thread = threading.Thread(target=utils.face_recognition_thread)
        recognition_thread.daemon = True
        recognition_thread.start()

        return jsonify({"status": "success"})
    
    return jsonify({"status": "already running"})

# Route for stop stream
@app.route('/api/stop_stream', methods=['POST'])
def stop_stream():
    utils.streaming = False
    utils.processing_active = False

    with utils.camera_lock:
        if utils.camera is not None:
            utils.camera.release()
            utils.camera = None

    return jsonify({"status": "success"})

# Route for video feed
@app.route('/api/video_feed')
def video_feed():
    return Response(utils.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route for check in
@app.route('/api/check_in', methods=['POST'])
def check_in():
    data = request.get_json()
    user_id = data.get('user_id')
    meeting_id = data.get('meeting_id')
    location_id = data.get('location_id')
    venue = data.get('venue')
    status = 1
    role = data.get('role')
    
    # Get category from user_id if it has a prefix
    if user_id and '_' in user_id:
        parts = user_id.split('_', 1)
        if len(parts) == 2:
            category, user_id = parts
            # Convert string role to int based on category
            if category == 'visitor':
                role = 0
            elif category == 'contractor':
                role = 1
    
    print("Role:", role)
    print("Location id:", location_id)
    print("User ID:", user_id)

    if not user_id:
        return jsonify({"status": "error", "message": "No user ID"}), 400
    
    # Check for db connection
    conn = utils.get_db_connection()
    if not conn:
        return jsonify({"status": "error", "message": "Database connection failed"}), 500
    
    try:
        cursor = conn.cursor(dictionary=True)

        # Get user name from the user ID (which is the filename)
        user_name = user_id

        print(user_name)

        # Retrieve the location name
        query = "SELECT name FROM location WHERE id = %s"
        cursor.execute(query, (location_id,))
        location_name = cursor.fetchone()
        
        # Query to retrieve details based on roles
        if role == 0:
            query = "SELECT id, name FROM visitor WHERE id = %s"
        elif role == 1:
            query = "SELECT id, name FROM contractor WHERE id = %s"

        # Check if user exists
        cursor.execute(query, (user_id,))
        user = cursor.fetchone()
        
        # Check if user already checked in based on role
        if role == 0:
            query = "SELECT id FROM visitorattendance WHERE visitor_id = %s AND DATE(Clockin_time) = %s AND Clockout_time IS NULL"
        elif role == 1:
            query = "SELECT id FROM contractorattendance WHERE contractor_id = %s AND DATE(clockin_time) = %s AND clockout_time IS NULL"

        current_date = datetime.now().strftime('%Y-%m-%d')
        cursor.execute(
            query, 
            (user_id, current_date)
        )

        existing_checkin = cursor.fetchone()
        
        if existing_checkin:
            return jsonify({
                "status": "error", 
                "message": "User already checked in today. Please check out first."
            }), 400
        
        # Insert new check-in record
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Check in based on role
        if role == 0:
            cursor.execute(
                "INSERT INTO visitorattendance (visitor_id, Clockin_time, MeetingDetails_id, location_id, status, Clockin_location) VALUES (%s, %s, %s, %s, %s, %s)",
                (user_id, now, meeting_id, location_id, status, venue)
            )
        elif role == 1:
            cursor.execute(
                "INSERT INTO contractorattendance (contractor_id, Clockin_time, location_id, status, clockin_location) VALUES (%s, %s, %s, %s, %s)",
                (user_id, now, location_id, status, location_name['name'] if location_name else venue)
            )
        
        conn.commit()

        user_display_name = user['name'] if user else user_id
        return jsonify({
            "status": "success",
            "message": f"Check-in successful for {user_display_name}",
            "timestamp": now
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if 'cursor' in locals():
            cursor.close()
        conn.close()

@app.route('/api/check_out', methods=['POST'])
def check_out():
    data = request.get_json()
    user_id = data.get('user_id')
    role = data.get('role')
    venue = data.get('venue')
    status = 2
    location_id = data.get('location_id')

    
    # Get category from user_id if it has a prefix
    if user_id and '_' in user_id:
        parts = user_id.split('_', 1)
        if len(parts) == 2:
            category, user_id = parts
            # Convert string role to int based on category
            if category == 'visitor':
                role = 0
            elif category == 'contractor':
                role = 1
    
    print("Role for checkout:", role)
    print("User ID for checkout:", user_id)
    
    if not user_id:
        return jsonify({"status": "error", "message": "No user ID provided"}), 400
    
    conn = utils.get_db_connection()
    if not conn:
        return jsonify({"status": "error", "message": "Database connection failed"}), 500
    
    try:
        cursor = conn.cursor(dictionary=True)

        # Retrieve the location name
        query = "SELECT name FROM location WHERE id = %s"
        cursor.execute(query, (location_id,))
        location = cursor.fetchone()

        location_name = location['name'] if location else None
        
        # Query to retrieve details based on roles
        if role == 0:
            query = "SELECT id, name FROM visitor WHERE id = %s"
        elif role == 1:
            query = "SELECT id, name FROM contractor WHERE id = %s"

        # Check if user exists
        cursor.execute(query, (user_id,))
        user = cursor.fetchone()
        
        if not user:
            return jsonify({"status": "error", "message": "User not found"}), 404
        
        if role == 0:
            query = "SELECT id FROM visitorattendance WHERE visitor_id = %s AND DATE(Clockin_time) = %s AND Clockout_time IS NULL ORDER BY Clockin_time DESC LIMIT 1"
        elif role == 1:
            query = "SELECT id FROM contractorattendance WHERE contractor_id = %s AND DATE(clockin_time) = %s AND clockout_time IS NULL ORDER BY clockin_time DESC LIMIT 1"

        # Find last check-in without check-out
        current_date = datetime.now().strftime('%Y-%m-%d')
        cursor.execute(
            query, 
            (user_id, current_date)
        )
        checkin_record = cursor.fetchone()
        
        if not checkin_record:
            return jsonify({
                "status": "error", 
                "message": "No active check-in found for today. Please check in first."
            }), 400
        
        # Update with check-out time
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        if role == 0:
            cursor.execute(
                "UPDATE visitorattendance SET Clockout_time = %s, status = %s, Clockout_location = %s WHERE id = %s",
                (now, status, venue, checkin_record['id'])
            )
        elif role == 1:
            cursor.execute(
                "UPDATE contractorattendance SET Clockout_time = %s, status = %s, clockout_location = %s WHERE id = %s",
                (now, status, location_name, checkin_record['id'])
            )
        else:
            print("Invalid role:", role)
            return jsonify({'status': 'error', 'message': 'Invalid role'}), 400

        conn.commit()

        return jsonify({
            "status": "success",
            "message": f"Check-out successful for {user['name']}",
            "timestamp": now
        })
    
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if 'cursor' in locals():
            cursor.close()
        conn.close()

@app.route('/api/get_recognized_faces', methods=['GET'])
def get_recognized_faces():
    role_value = request.args.get('role')
    
    if role_value is None or role_value == 'NaN':
        role = None
    else:
        role = int(role_value)
    
    with utils.frame_lock:
        current_faces = utils.face_data.copy()
    
    # Filter faces by category if role is specified
    if role is not None:
        category_filter = 'visitor' if role == 0 else 'contractor'
        filtered_faces = [face for face in current_faces 
                         if face["name"] != "Tidak Diketahui" and 
                         (face["category"] == category_filter)]
    else:
        filtered_faces = [face for face in current_faces if face["name"] != "Tidak Diketahui"]
    
    # Extract user IDs (without category prefix)
    user_dict = {}
    face_list = []
    
    for face in filtered_faces:
        user_id = face["user_id"]
        category = face["category"]
        
        if not user_id or not category:
            continue
            
        # For database query, we need the raw user_id without category prefix
        try:
            conn = utils.get_db_connection()
            cursor = conn.cursor(dictionary=True)
            
            table = "visitor" if category == "visitor" else "contractor"
            query = f"SELECT id, name FROM {table} WHERE id = %s"
            cursor.execute(query, (user_id,))
            
            user = cursor.fetchone()
            if user:
                display_name = user["name"]
            else:
                display_name = user_id
                
            cursor.close()
            conn.close()
            
            face_list.append({
                "name": display_name,
                "similarity": face["similarity"],
                "user_id": f"{category}_{user_id}",  # Use combined ID with category prefix
                "category": category
            })
            
        except Exception as e:
            print(f"Database error: {e}")
            # Fallback to just using the raw data
            face_list.append({
                "name": user_id,
                "similarity": face["similarity"],
                "user_id": f"{category}_{user_id}",
                "category": category
            })
    
    # Debug
    print("Recognized faces:", face_list)
    
    return jsonify({"status": "success", "faces": face_list})

@app.route('/api/check_new_images', methods=['GET'])
def check_new_images():
    notification = utils.new_images_notification
    if notification["active"]:
        # Get count by category
        category_details = ""
        for category, count in notification["count_by_category"].items():
            if count > 0:
                category_details += f"{count} {category}, "
        
        if category_details:
            category_details = category_details[:-2]  # Remove trailing comma and space
            
        return jsonify({
            "new_images": True,
            "count": notification["count"],
            "count_by_category": notification["count_by_category"],
            "message": f"Found new face images ({category_details}). Please reload the known faces!"
        })
    else:
        return jsonify({"new_images": False})
    
@app.route('/api/acknowledge_notification', methods=['POST'])
def acknowledge_notification():
    utils.reset_notification()
    return jsonify({"status": "success"})

@app.route('/api/reload_faces', methods=['POST'])
def reload_faces():
    threading.Thread(target=utils.load_known_faces).start()
    utils.reset_notification()
    return jsonify({"status":"loading"})

@app.route('/api/face_stats', methods=['GET'])
def face_stats():
    return jsonify(utils.get_face_stats())

if __name__ == '__main__':
    utils.start_directory_monitor()
    app.run(debug=True, host='0.0.0.0', port=5000)