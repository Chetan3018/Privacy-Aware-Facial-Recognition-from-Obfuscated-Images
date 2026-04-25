from django.shortcuts import render, redirect
from django.conf import settings
from .utils import process_uploaded_images, match_blurred_with_original
from PIL import Image
import numpy as np
import cv2
import os
import uuid


def home(request):
    """Home page with upload form"""
    return render(request, 'faceblur/home.html')


def process_images(request):
    """Process uploaded images and return result"""
    if request.method == 'POST':

        ref_image = request.FILES.get('ref_image')
        target_image = request.FILES.get('target_image')

        if not ref_image or not target_image:
            return render(request, 'faceblur/home.html', {
                'error': 'Please upload both reference and target images'
            })

        success, message, result_file, faces_detected,confidence,processing_time = process_uploaded_images(
            ref_image, target_image
        )

        if success:
            # Ensure media directory exists
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)

            # Create unique filename (avoids overwrite)
            filename = f"result_{uuid.uuid4().hex}.jpg"
            result_path = os.path.join(settings.MEDIA_ROOT, filename)

            # SAVE ContentFile CORRECTLY
            with open(result_path, 'wb') as f:
                f.write(result_file.read())

            return render(request, 'faceblur/result.html', {
                'success': True,
                'message': message,
                'faces_detected': faces_detected,
                'confidence': confidence,
                'processing_time': processing_time,
                'result_image_url': settings.MEDIA_URL + filename
            })

        else:
            return render(request, 'faceblur/home.html', {
                'error': message
            })

    return redirect('home')


def match_images(request):
    """Match a blurred image with an original image"""
    if request.method == 'POST':
        blurred_image = request.FILES.get('blurred_image')
        original_image = request.FILES.get('original_image')

        if not blurred_image or not original_image:
            return render(request, 'faceblur/home.html', {
                'error': 'Please upload both blurred and original images'
            })

        success, message, similarity, processing_time, face_crop_bytes, face_box = match_blurred_with_original(
            blurred_image, original_image
        )

        if success:
            # If matched, save the annotated original image as result
            os.makedirs(settings.MEDIA_ROOT, exist_ok=True)
            filename = f"verified_{uuid.uuid4().hex}.jpg"
            result_path = os.path.join(settings.MEDIA_ROOT, filename)

            # Load original and optionally draw circle around matched face
            original = Image.open(original_image)
            original = original.convert('RGB')
            original_cv = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)

            if face_box:
                (top, right, bottom, left) = face_box
                center = ((left + right) // 2, (top + bottom) // 2)
                radius = max((right - left), (bottom - top)) // 2 + 15
                cv2.circle(original_cv, center, radius, (0, 255, 0), 4)
                cv2.rectangle(original_cv, (left, top), (right, bottom), (0, 255, 0), 2)

            cv2.imwrite(result_path, original_cv)

            # Save crop file if available
            face_crop_url = None
            if face_crop_bytes is not None:
                crop_name = f"matched_face_{uuid.uuid4().hex}.jpg"
                crop_path = os.path.join(settings.MEDIA_ROOT, crop_name)
                with open(crop_path, 'wb') as f:
                    f.write(face_crop_bytes)
                face_crop_url = settings.MEDIA_URL + crop_name

            return render(request, 'faceblur/result.html', {
                'success': True,
                'message': message,
                'similarity': similarity,
                'processing_time': processing_time,
                'result_image_url': settings.MEDIA_URL + filename,
                'match_mode': True,
                'face_crop_url': face_crop_url,
                'face_box': face_box
            })
        else:
            return render(request, 'faceblur/home.html', {
                'error': message
            })

    return redirect('home')


def about(request):
    """About page with information about the tool"""
    return render(request, 'faceblur/about.html')
