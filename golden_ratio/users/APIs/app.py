import os
from flask import Flask, flash, request, redirect, url_for, render_template, send_from_directory, send_file, session
from reportlab.lib import colors
from werkzeug.utils import secure_filename
from flask_login import login_user, login_required, logout_user, current_user
from flask_migrate import Migrate
from extensions import db, login_manager
from models import User
import cv2
import numpy as np
import mediapipe as mp
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from landmarks import l1
from m2r import filter_landmark3 as filter_landmark3,filter_landmark4 as filter_landmark4
from refer import l2, rft_arr, rft1_arr
from phiweb import phi_matrix_method
from sym_asym import final_res



app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'  # Change this to a secure key in production
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # Ensure this is correct for your setup
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Optional: to suppress warnings
db.init_app(app)
login_manager.init_app(app)
login_manager.login_view = 'login'
migrate = Migrate(app, db)
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
if not os.path.exists(app.config['OUTPUT_FOLDER']):
    os.makedirs(app.config['OUTPUT_FOLDER'])

Name = ['Eye Spacing', 'Right eye width', 'Left eye width',
        'Right start of brow to arc', 'Left start of brow to arc',
        'Right oral corner to face side', 'Left oral corner to face side',
        'Oral center to chin', 'Upper lip width', 'Nose width', 'Forehead width',
        'Chin width']

Name2 = ['Right eye corner to face edge', 'Left eye corner to face edge',
         'Right eyebrow width', 'Left eyebrow width', 'Nose length',
         'Oral width', 'Nose tip to oral center', 'Lower lip width',
         'Right eye corner to cheekbone', 'Left eye corner to cheekbone',
         'Middle forehead to right face edge', 'Middle forehead to left face edge']



def idraw_lines_with_text(image, landmarks, landmark_pairs):
    sum1 = 0
    red = []
    reference_real_world_size = 3.5
    distances = []  # List to store distances (ft)
    for pair, n, name in zip(landmark_pairs, rft_arr, Name):
        start_idx = pair['start']
        end_idx = pair['end']
        reference = pair['refval'] #n is aso used for reference both pair['refval'] and n for same list  rft_arr
        start_pt = landmarks[start_idx]
        end_pt = landmarks[end_idx]
        distance = np.linalg.norm(np.array(start_pt) - np.array(end_pt)) / reference_real_world_size
        ft = float(distance)  # Store the distance as float
        distances.append(ft)  # Append the distance to the list
        v = ft / reference
        s1=v/1.618
        if v * 100 > 100:
            s1=1.618/v
            v = reference / ft
            print(f"{name}\npatient:{ft:.2f}\treference:{n}\nGR_percentage:{s1 * 100:.2f}")
        else:
            print(f"{name}\npatient:{ft:.2f}\treference:{n}\nGR_percentage:{s1 * 100:.2f}")
        sum1 += v * 100
        red.append(v * 100)
        cv2.line(image, start_pt, end_pt, (0, 0, 255, 0), 2)
        midpoint = ((start_pt[0] + end_pt[0]) // 2, (start_pt[1] + end_pt[1]) // 2)
        text_position = (midpoint[0], midpoint[1] - 10)
        cv2.putText(image, f"{v * 100}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    return image, sum1, red, distances  # Return the list of distances


def idraw_lines_with_text1(image, landmarks, landmark_pairs):
    sum2 = 0
    green = []
    reference_real_world_size = 3.5
    distances = []  # List to store distances (ft1)
    for pair, r, name2 in zip(landmark_pairs, rft1_arr, Name2):
        start_idx = pair['start']
        end_idx = pair['end']
        reference = pair['refval'] #r is also used for 'reference' both pair['refval'] and n comes from same list  rft1_arr
        start_pt = landmarks[start_idx]
        end_pt = landmarks[end_idx]
        distance = np.linalg.norm(np.array(start_pt) - np.array(end_pt)) / reference_real_world_size
        ft1 = float(distance)  # Store the distance as float
        distances.append(ft1)  # Append the distance to the list
        c = reference / 1.618
        s = ft1 / c
        s1 = s / 1.618
        if s1 * 100 > 100:
            s1 = 1.618 / s
            print(f"{name2}\npatient:{ft1:.2f}\treference:{r}\nGR_percentage:{s1 * 100:.2f}")
        else:
            print(f"{name2}\npatient:{ft1:.2f}\treference:{r}\nGR_percentage:{s1 * 100:.2f}")
        sum2 += s1 * 100
        green.append(s1 * 100)
        cv2.line(image, tuple(start_pt), tuple(end_pt), (0, 255, 0), 2)
        midpoint = ((start_pt[0] + end_pt[0]) // 2, (start_pt[1] + end_pt[1]) // 2)
        text_position = (midpoint[0], midpoint[1] - 10)
        cv2.putText(image, f"{s1 * 100}", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return image, sum2, green, distances  # Return the list of distances


def apply_filter(image, landmarks, landmark_indices):
    for idx in landmark_indices:
        landmark_pt = tuple(map(int, landmarks[idx]))
        cv2.circle(image, landmark_pt, 3, (0, 255, 0), -1)
    return image

def calculate_golden_ratio(input_path, output_path):
    ratios = []
    image = cv2.imread(input_path)
    if image is None:
        raise FileNotFoundError(f"Failed to load image from: {input_path}")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            height, width, _ = image.shape
            landmarks = [(int(landmark.x * width), int(landmark.y * height)) for landmark in face_landmarks.landmark]
            landmark_indices = [lm['start'] for lm in l1] + [lm['end'] for lm in l1]
            image = apply_filter(image, landmarks, landmark_indices)
            landmark_indices1 = [lm['start'] for lm in l2] + [lm['end'] for lm in l2]
            image1 = apply_filter(image, landmarks, landmark_indices1)
            image, rsum, redlist, ft_distances = idraw_lines_with_text(image, landmarks, l1)
            image1, gsum, greenlist, ft1_distances = idraw_lines_with_text1(image1, landmarks, l2)
            asum = rsum + gsum
            avg = asum / (len(l1) + len(l2))
            print(f"percentage: {avg:.2f}%")

            # Prepare ratio data
            ft_combined = ft_distances + ft1_distances  # Combine the distances
            gr_percentage = redlist + greenlist
            for idx,(name, patient_value, reference_value,g) in enumerate(zip(Name + Name2, ft_combined, rft_arr + rft1_arr,gr_percentage)):

                ratios.append({
                    'Name': name,
                    'patient_value': f"{patient_value:.3f}",
                    'reference_value': f"{reference_value:.3f}",
                    'gr_percentage': f"{g:.3f}"
                })


            # for landmark_pt in landmarks:
            #     cv2.circle(image, landmark_pt, 1, (255, 0, 0), -1)

    cv2.imwrite(output_path, image)
    return output_path, ratios, avg

inpt_arr=[]
inpt1_arr=[]
#value1

def rdraw_lines_with_text(image, landmarks, landmark_pairs):
    reference_real_world_size = 3.5

    for pair in landmark_pairs:
        start_idx = pair['start']
        end_idx = pair['end']
        start_pt = landmarks[start_idx]
        end_pt = landmarks[end_idx]
        distance = np.linalg.norm(np.array(start_pt) - np.array(end_pt))/reference_real_world_size
        inpt_arr.append(float(f"{distance:.3f}"))

        cv2.line(image, start_pt, end_pt, (0, 0, 255, 0), 2)

        # Calculate the midpoint
        midpoint = ((start_pt[0] + end_pt[0]) // 2, (start_pt[1] + end_pt[1]) // 2)

        # Adjust the position of the text to be above the line
        text_position = (midpoint[0], midpoint[1] - 10)

        #Annotate with the label
        #cv2.putText(image, label, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return image

# value 1.618

def rdraw_lines_with_text1(image, landmarks, landmark_pairs):
    reference_real_world_size = 3.5

    for pair in landmark_pairs:
        start_idx = pair['start']
        end_idx = pair['end']
        start_pt = landmarks[start_idx]
        end_pt = landmarks[end_idx]
        distance = np.linalg.norm(np.array(start_pt) - np.array(end_pt))/reference_real_world_size
        inpt1_arr.append(float(f"{distance:.3f}"))

        #Draw the line
        cv2.line(image, start_pt, end_pt, (0, 255, 0), 2)

        # Calculate the midpoint
        midpoint = ((start_pt[0] + end_pt[0]) // 2, (start_pt[1] + end_pt[1]) // 2)

        # Adjust the position of the text to be above the line
        text_position = (midpoint[0], midpoint[1] - 10)

        #Annotate with the label
       # cv2.putText(image, str(w), text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    return image

def image_per(image_path_var,output_path):

    ratios1=[]
    inpt_arr.clear()
    inpt1_arr.clear()
    image = cv2.imread(image_path_var)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    if image is None:
       raise FileNotFoundError(f"Failed to load image from: {image_path_var}")


    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Process the image with MediaPipe face mesh
    results = face_mesh.process(image_rgb)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
          height, width, _ = image.shape
          l= []
          landmarks = [(int(landmark.x * width),int(landmark.y * height)) for landmark in face_landmarks.landmark]


          landmark_indices = [lm['start'] for lm in filter_landmark3] + [lm['end'] for lm in filter_landmark3]
          image = apply_filter(image, landmarks, landmark_indices)
          landmark_indices1 = [lm['start'] for lm in filter_landmark4] + [lm['end'] for lm in filter_landmark4]
          image = apply_filter(image, landmarks, landmark_indices1)

          image = rdraw_lines_with_text(image, landmarks, filter_landmark3)
          image= rdraw_lines_with_text1(image, landmarks, filter_landmark4)


          # for landmark_pt in landmarks:
          #
          #     cv2.circle(image,landmark_pt, 1, (255, 0, 0), -1)  # Draw a small dot for each landmark


    for i, j, r1, r2 in zip(inpt_arr, inpt1_arr, Name, Name2):
        p = i * 1.618
        q = p / j
        if q * 100 > 100:
            q = j / p
            q * 100
            ratios1.append({
                'Description1': f"{r1}",
                'Description2': f"{r2}",
                'dist1': f"{i:.3f}",
                'dist2': f"{j:.3f}",
                'Percentage': f"{q * 100:.3f}"
            })
        else:
            q * 100
            ratios1.append({
                'Description1': f"{r1}",
                'Description2': f"{r2}",
                'dist1': f"{i:.3f}",
                'dist2': f"{j:.3f}",
                'Percentage': f"{q * 100:.3f}"
            })

    avg_percentage = sum([float(ratio['Percentage']) for ratio in ratios1]) / len(ratios1)
    cv2.imwrite(output_path, image)
    return output_path, ratios1, avg_percentage



@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('home'))
    if 'visited' in session:
        return redirect(url_for('login'))
    else:
        session['visited'] = True
        return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('home'))
        flash('Login failed. Check your username and/or password.')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if User.query.filter_by(username=username).first():
            flash('Username already exists. Please choose a different username.')
            return redirect(url_for('signup'))

        user = User(username=username)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash('Sign up successful! You can now log in.')
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/home')
@login_required
def home():
    return render_template('index.html', user=current_user)

@app.route('/logout')
@login_required
def logout():
    logout_user()
    session.pop('visited', None)
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
def upload_file():
    print("Upload file function called")
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(file_path)
        session['filename'] = filename
        flash('Image successfully uploaded and displayed below')
        print(f"Redirecting to select_method with filename: {filename}")  # Debug log
        return redirect(url_for('select_method'))
    else:
        flash('Allowed image types are png, jpg, jpeg, gif')
        return redirect(request.url)

@app.route('/select-method')
def select_method():
    filename = session.get('filename')
    print(f"Filename received in select_method: {filename}")  # Debug log
    if filename is None:
        flash('Filename parameter is missing.')
        return redirect(url_for('index'))
    session.pop('results', None)
    session.pop('ratios', None)
    session.pop('avg_percentage', None)
    session.pop('input_image', None)
    session.pop('method', None)
    return render_template('select_method.html', filename=filename)

@app.route('/static/uploads/<filename>')
def output_file(filename):
    return send_from_directory('static/uploads', filename)


@app.route('/display/<filename>')
def display_image(filename):
    print(filename)
    return render_template('results.html', filename=filename)


@app.route('/process', methods=['GET'])
def process_image():
    filename = session.get('filename')
    method = request.args.get('method')

    if not filename or not method:
        flash('Invalid request. Please select an image and a method.')
        return redirect(url_for('index'))
    session.pop('results', None)
    session.pop('image_path', None)

    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_filename = f"processed_{filename}"
    output_filename1 = f"processed1_{filename}"
    print("file name----->",filename)
    output_filename2 = f"processed2_{filename}"
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    print("op---->",output_filename)
    output_path1 = os.path.join(app.config['OUTPUT_FOLDER'], output_filename1)
    print("op1---->",output_filename1)
    output_path2 = os.path.join(app.config['OUTPUT_FOLDER'], output_filename2)
    print("op2---->",output_filename2)

    if method == 'golden_ratio':
        processed_image, ratios, avg_percentage = calculate_golden_ratio(input_path, output_path)
    elif method == 'phi_matrix':
        processed_image, ratios, avg_percentage = phi_matrix_method(input_path, output_path)
    elif method == 'input_as_reference':
        processed_image, ratios, avg_percentage = image_per(input_path, output_path)
    elif method == 'symmetric_asymmetric':
        # Ensure method-specific handling0
        processed_image, processed_image1, processed_image2, ratios = final_res(input_path, output_path, output_path1,
                                                                                output_path2)
        session['results'] = {
            'filename': output_filename,
            'processed_image': output_filename,
            'processed_image1': output_filename1,
            'processed_image2': output_filename2,
            'ratios': ratios,
            'method': method,
            'input_image': input_path
        }
        print(f"Symmetric Asymmetric ratios: {ratios}")
        return render_template('results.html',
                               output_path=output_filename,
                               output_path1=output_filename1,
                               output_path2=output_filename2,
                               ratios=ratios,
                               method=method)
    else:
        flash('Invalid method selected')
        return redirect(url_for('index'))
    session['results'] = {
        'filename': output_filename,
        'processed_image': processed_image,
        'ratios': ratios,
        'avg_percentage': avg_percentage,
        'method': method,
        'input_image': input_path
    }
    print(f"Other methods ratios: {ratios}")
    return render_template('results.html',
                           filename=output_filename,
                           processed_image=processed_image,
                           ratios=ratios,
                           avg_percentage=avg_percentage,
                           method=method,
                           input_image=input_path)


@app.route('/compare-report', methods=['GET'])
def compare_report():
    filename = session.get('filename')

    if not filename:
        flash('Invalid request. Please select an image.')
        return redirect(url_for('index'))

    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    output_filename_golden_ratio = f"golden_ratio_{filename}"
    output_filename_phi_matrix = f"phi_matrix_{filename}"
    output_filename_input_reference = f"input_reference_{filename}"

    output_path_golden_ratio = os.path.join(app.config['OUTPUT_FOLDER'], output_filename_golden_ratio)
    output_path_phi_matrix = os.path.join(app.config['OUTPUT_FOLDER'], output_filename_phi_matrix)
    output_path_input_reference = os.path.join(app.config['OUTPUT_FOLDER'], output_filename_input_reference)

    # Call the processing functions for each method

    phi_matrix_image, phi_matrix_ratios, phi_matrix_avg_percentage = phi_matrix_method(
        input_path, output_path_phi_matrix)
    input_as_reference_image, input_as_reference_ratios, input_as_reference_avg_percentage = image_per(
        input_path, output_path_input_reference)
    golden_ratio_image, golden_ratio_ratios, golden_ratio_avg_percentage = calculate_golden_ratio(
        input_path, output_path_golden_ratio)

    # Ensure the paths are correctly formatted for use in templates
    golden_ratio_image_url = url_for('static', filename=f'output/{output_filename_golden_ratio}')
    phi_matrix_image_url = url_for('static', filename=f'output/{output_filename_phi_matrix}')
    input_as_reference_image_url = url_for('static', filename=f'output/{output_filename_input_reference}')

    # Prepare the comparison data
    comparison_data = {
        'phi_matrix': {
            'image': phi_matrix_image_url,
            'percentage': phi_matrix_avg_percentage
        },
        'input_as_reference': {
            'image': input_as_reference_image_url,
            'percentage': input_as_reference_avg_percentage
        },
        'golden_ratio': {
            'image': golden_ratio_image_url,
            'percentage': golden_ratio_avg_percentage
        }
    }

    # Render the template with the comparison data
    return render_template('compare_report.html', data=comparison_data,filename=filename)


def generate_pdf(data, method):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []

    # Define styles
    styles = getSampleStyleSheet()

    # Method-specific content
    if method == 'phi_matrix':
        elements.append(Paragraph("Phi Matrix Report", styles['Title']))

        # Process image
        if isinstance(data, tuple) and len(data) > 0:
            img_path = data[0] if isinstance(data[0], str) else None
            if img_path:
                elements.append(Image(img_path, width=400, height=500))
                elements.append(Spacer(1, 24))
        # Process average percentage
        avg_percentage = data[2] if isinstance(data[2], (int, float)) else 0
        elements.append(Paragraph(f"Average Percentage: {avg_percentage:.3f}%", styles['Normal']))
        elements.append(Spacer(1, 24))
        # Process table data
        data_for_table = [['S.No.', 'Description', 'Distance', 'Percentage']]
        if isinstance(data[1], list):
            for i, ratio in enumerate(data[1]):
                if isinstance(ratio, dict):
                    desc1 = ratio.get('Description1', 'N/A')
                    desc2 = ratio.get('Description2', 'N/A')
                    description = f"{desc1} -\n {desc2}" if desc2 != 'N/A' else desc1
                    dist1 = ratio.get('dist1', 'N/A')
                    dist2 = ratio.get('dist2', 'N/A')
                    percentage = ratio.get('Percentage', 'N/A')
                    data_for_table.append([
                        i + 1,
                        description,
                        f"{dist1} - {dist2}",
                        f"{percentage:.3f}%"
                    ])

        # Define column widths
        col_widths = [doc.width / 14, doc.width / 1.3, doc.width / 5, doc.width / 9]
        table = Table(data_for_table, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),  # Align all text to the left
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white])
        ]))
        elements.append(table)

    elif method == 'input_as_reference':
        elements.append(Paragraph("Input As Reference Report", styles['Title']))

        # Extract image path
        image_path = data[0] if isinstance(data[0], str) else ''
        if image_path:
            elements.append(Image(image_path, width=400, height=500))
            elements.append(Spacer(1, 24))
        # Add average percentage
        avg_percentage = data[2] if isinstance(data[2], (int, float)) else 0
        elements.append(Paragraph(f"Average Percentage: {avg_percentage:.3f}%", styles['Normal']))
        elements.append(Spacer(1, 24))
        # Handle table data safely
        data_for_table = [['S.No.', 'Description', 'Distance', 'Percentage']]
        if isinstance(data[1], list):
            for i, ratio in enumerate(data[1]):
                if isinstance(ratio, dict):
                    description1 = ratio.get('Description1', 'N/A')
                    description2 = ratio.get('Description2', 'N/A')
                    distance1 = ratio.get('dist1', 'N/A')
                    distance2 = ratio.get('dist2', 'N/A')
                    percentage = ratio.get('Percentage', 'N/A')

                    if description2 != 'N/A':
                        description = f"{description1} - {description2}"
                    else:
                        description = description1

                    data_for_table.append([
                        i + 1,
                        description,
                        f"{distance1} - {distance2}",
                        f"{percentage}%"
                    ])

        # Define column widths
        col_widths = [doc.width / 12, doc.width / 1.6, doc.width / 6, doc.width / 8]
        table = Table(data_for_table, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),  # Align all text to the left
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white])
        ]))
        elements.append(table)

    elif method == 'symmetric_asymmetric':
        elements.append(Paragraph("Symmetric Asymmetric Report", styles['Title']))

        image_paths = []
        if isinstance(data, (tuple, list)) and len(data) > 2:
            image_paths = [data[0], data[1], data[2]]  # Assume first three are image paths

        for img_path in image_paths:
            if isinstance(img_path, str):
                elements.append(Image(img_path, width=400, height=500))
                elements.append(Spacer(1, 12))
        # Handle table data safely
        data_for_table = [['S.No.', 'Name', 'Distance', 'Symmetry', 'Percentage']]
        if len(data) > 3 and isinstance(data[3], list):
            for i, ratio in enumerate(data[3]):
                if isinstance(ratio, dict):
                    data_for_table.append([
                        i + 1,
                        ratio.get('Name', 'N/A'),
                        round(ratio.get('Distance', 0), 3),
                        ratio.get('Symmetry', 'N/A'),
                        f"{round(ratio.get('Percentage', 0), 3)}%" if ratio.get('Percentage') else 'N/A'
                    ])

        # Define column widths
        col_widths = [doc.width / 12, doc.width / 2.2, doc.width / 8, doc.width / 8, doc.width / 8]
        table = Table(data_for_table, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),  # Align all text to the left
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white])
        ]))
        elements.append(table)
    elif method == 'golden_ratio':
        print(data)
        elements.append(Paragraph("Golden Ratio Report", styles['Title']))

        # Add the processed image
        if isinstance(data, (tuple, list)) and len(data) > 0:
            img_path = data[0]
            if isinstance(img_path, str):
                elements.append(Image(img_path, width=400, height=500))
                elements.append(Spacer(1, 24))
        # Average percentage
        avg_percentage = data[2] if isinstance(data[2], (int, float)) else 0
        elements.append(Paragraph(f"Average Percentage: {avg_percentage:.3f}%", styles['Normal']))
        elements.append(Spacer(1, 24))
        # Table data
        data_for_table = [['S.No.', 'Name', 'Input\n Distance', 'Reference\n Distance', 'Percentage']]
        if isinstance(data[1], list):
            for i, ratio in enumerate(data[1]):
                if isinstance(ratio, dict):
                    data_for_table.append([
                        i + 1,
                        ratio.get('Name', 'N/A'),
                        ratio.get('patient_value', 'N/A'),
                        ratio.get('reference_value', 'N/A'),
                        f"{ratio.get('gr_percentage', 'N/A')}%"
                    ])

        # Define column widths
        col_widths =[doc.width / 10, doc.width / 3, doc.width / 6, doc.width / 6, doc.width / 6]
        table = Table(data_for_table, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),  # Align all text to the left
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('BACKGROUND', (0, 1), (-1, -1), colors.white),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.whitesmoke, colors.white])
        ]))
        elements.append(table)
    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

@app.route('/generate_pdf', methods=['GET'])
def generate_pdf_route():
    filename = request.args.get('filename')
    method = request.args.get('method')

    if not filename or not method:
        flash('Invalid request. Please select an image and a method.')
        return redirect(url_for('index'))

    input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    output_filename = f"processed_{filename}"
    output_filename1 = f"processed1_{filename}"
    output_filename2 = f"processed2_{filename}"

    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
    output_path1 = os.path.join(app.config['OUTPUT_FOLDER'], output_filename1)
    output_path2 = os.path.join(app.config['OUTPUT_FOLDER'], output_filename2)

    # Process based on the method
    if method == 'phi_matrix':
        data = phi_matrix_method(input_path, output_path)
    elif method == 'input_as_reference':
        data = image_per(input_path, output_path)
    elif method == 'symmetric_asymmetric':
        data = final_res(input_path, output_path, output_path1, output_path2)
    elif method == 'golden_ratio':
        data = calculate_golden_ratio(input_path, output_path)
    else:
        return 'Invalid option selected', 400

    pdf_content = generate_pdf(data, method)
    return send_file(BytesIO(pdf_content), as_attachment=True, download_name=f"{method}_report.pdf", mimetype='application/pdf')



def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

if __name__ == "__main__":
    app.run(debug=True)
