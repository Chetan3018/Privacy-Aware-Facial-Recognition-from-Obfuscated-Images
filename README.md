# FaceBlur - Django Web Application

A complete web-based face detection and blurring tool built with Django, face_recognition, and OpenCV.

## 🚀 Quick Start

### 1. Install Dependencies

Using the requirements.txt file:
```bash
pip install -r requirements.txt
```

Or manually install:
```bash
pip install django face-recognition opencv-python numpy pillow scikit-image
```

On Windows, you may also need:
```bash
pip install cmake dlib
```

### 2. Run the Application

```bash
cd faceblur_project
python manage.py runserver
```

### 3. Access the Web Interface

Open your browser and go to: `http://localhost:8000`

## 📁 Project Structure

```
faceblur_project/
├── faceblur/                  # Main Django app
│   ├── templates/             # HTML templates
│   │   └── faceblur/          # App templates
│   │       ├── base.html      # Base template
│   │       ├── home.html      # Upload form
│   │       ├── result.html    # Results page
│   │       └── about.html     # About page
│   ├── utils.py               # Face detection logic
│   ├── views.py               # Django views
│   └── urls.py               # App URLs
│
├── faceblur_project/          # Django project
│   ├── settings.py            # Project settings
│   ├── urls.py                # Main URLs
│   └── ...                    # Other Django files
│
├── media/                     # Uploaded and processed images
└── README.md                  # This file
```

## 🎯 Features

### Web Interface
- ✅ Responsive design (mobile & desktop)
- ✅ Drag-and-drop style file uploads
- ✅ Real-time file selection feedback
- ✅ Progress indicators and status messages
- ✅ Download processed images
- ✅ Multiple page navigation

### Face Processing
- ✅ Face detection using `face_recognition`
- ✅ Face matching between reference and target images
- ✅ Gaussian blur for privacy protection
- ✅ Bounding box visualization
- ✅ Configurable matching tolerance

### Image Verification (NEW!)
- ✅ Verify if a blurred image matches an original image
- ✅ Compare blur patterns using SSIM (Structural Similarity Index)
- ✅ Similarity scoring (0-100%)
- ✅ Automatic detection of matching blurred photos
- ✅ Return original image if verification succeeds

### Technical Features
- ✅ Django 4.2.11 backend
- ✅ Clean MVC architecture
- ✅ Error handling and validation
- ✅ Media file management
- ✅ Static file serving in development

## 📸 How to Use

### Mode 1: Detect & Blur Faces

#### Step 1: Upload Images
1. **Reference Image**: Upload a photo of the person you want to find
2. **Target Image**: Upload a photo where you want to find that person

#### Step 2: Process Images
- Click "Detect & Blur Faces"
- The system will analyze both images and find matching faces

#### Step 3: View Results
- See the processed image with blurred faces and bounding boxes
- Download the result or process another image

### Mode 2: Verify Blurred Images (NEW!)

#### Step 1: Upload Images
1. **Blurred Image**: Upload the blurred photo you want to verify
2. **Original Image**: Upload the original unblurred image

#### Step 2: Verify Match
- Click "Verify & Match Images"
- The system will blur the original image and compare it with the uploaded blurred image

#### Step 3: View Results
- If matched: You'll see the verified original image returned
- If not matched: You'll get a "Pictures do not match" message with similarity percentage
- Download the original image if verification succeeds

## 🔧 Configuration

### Settings
Edit `faceblur_project/settings.py` to customize:

```python
# Media settings
MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# Debug mode (for development)
DEBUG = True
```

### Face Recognition Parameters
Edit `faceblur/utils.py` to adjust:

```python
# Matching tolerance (lower = stricter)
tolerance=0.5

# Blur parameters
cv2.GaussianBlur(face, (99, 99), 30)
```

## 🎨 UI Features

### Home Page
- Clean, modern interface with card-based layout
- File upload areas with visual feedback
- Step-by-step instructions
- Use case examples

### Results Page
- Processed image display
- Download button for results
- Technical details about processing
- Option to process another image

### About Page
- Comprehensive technical documentation
- Algorithm explanations
- Use cases and limitations
- Technology stack information

## 🛠 Development

### Run Development Server
```bash
python manage.py runserver
```

### Create Superuser (for admin)
```bash
python manage.py createsuperuser
```

### Run Tests
```bash
python manage.py test
```

## 📦 Deployment

### Requirements for Production
- Python 3.7+
- Django 4.2+
- face_recognition 1.3+
- OpenCV 4.5+
- Web server (Nginx, Apache)
- WSGI server (Gunicorn, uWSGI)

### Production Settings
```python
# Set DEBUG = False
# Configure ALLOWED_HOSTS
# Set up proper static/media file serving
# Configure database for production
```

## 🔒 Security Notes

- The application is designed for local/development use
- For production deployment:
  - Set `DEBUG = False`
  - Configure proper security headers
  - Use HTTPS
  - Implement rate limiting
  - Secure file uploads

## 🤝 Contributing

This project is open for contributions. Feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

## 📝 License

This is a demonstration project. Use responsibly and in compliance with privacy laws and regulations.

---

🎉 **Ready to use!** Start the server and open `http://localhost:8000` in your browser.
