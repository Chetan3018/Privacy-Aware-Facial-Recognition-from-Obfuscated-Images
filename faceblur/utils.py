import face_recognition
import cv2
import numpy as np
import time
from PIL import Image
import os
from django.core.files.base import ContentFile
from skimage.metrics import structural_similarity as ssim


# =========================================================
# STRICT IMAGE LOADER (NO dlib ERRORS)
# =========================================================
def load_image_strict(path_or_file):
    """
    Load image strictly as 8-bit RGB for dlib/OpenCV compatibility.
    Accepts a filesystem path or a file-like object (UploadedFile).
    """
    if hasattr(path_or_file, 'read'):
        try:
            path_or_file.seek(0)
        except Exception:
            pass
        image = Image.open(path_or_file)
    else:
        image = Image.open(path_or_file)

    # Force RGB (remove alpha / CMYK / etc.)
    image = image.convert("RGB")

    # Convert to NumPy uint8 and ensure contiguous memory
    image_np = np.asarray(image, dtype=np.uint8)
    return np.ascontiguousarray(image_np)


# =========================================================
# FACE MATCH + BLUR
# =========================================================
def detect_and_blur_faces(ref_image_path, target_image_path, output_path):
    try:
        start_time= time.time()
        # ---------- LOAD IMAGES ----------
        ref_image = load_image_strict(ref_image_path)
        target_image = load_image_strict(target_image_path)

        # ---------- REFERENCE ENCODING ----------
        ref_encodings = face_recognition.face_encodings(ref_image)
        if len(ref_encodings) == 0:
            return False, "No face found in reference image", 0

        ref_encoding = ref_encodings[0]

        # ---------- TARGET FACES ----------
        face_locations = face_recognition.face_locations(target_image)
        if len(face_locations) == 0:
            return False, "No face found in target image", 0

        target_encodings = face_recognition.face_encodings(
            target_image, face_locations
        )

        # Convert RGB → BGR for OpenCV
        target_image_cv = cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR)

        faces_detected = 0
        best_confidence=0

        # ---------- MATCH + BLUR ----------
        for (top, right, bottom, left), face_encoding in zip(
            face_locations, target_encodings
        ):
            distance = face_recognition.face_distance(
            [ref_encoding], face_encoding
            )[0]
            confidence = (1 - distance)*100

            # Update best confidence for display
            if confidence > best_confidence:
                best_confidence = confidence

            # Relax threshold slightly for wider matching across bigger image sets
            if distance < 0.60:
                faces_detected += 1

                face = target_image_cv[top:bottom, left:right]
                blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
                target_image_cv[top:bottom, left:right] = blurred_face

        if faces_detected == 0:
            return False, "Reference person not found in target image", 0
        end_time = time.time()
        processing_time = round(end_time - start_time, 4)
        # ---------- SAVE OUTPUT ----------
        cv2.imwrite(output_path, target_image_cv)

        return True, f"Blurred {faces_detected} matching face(s)", faces_detected, round(best_confidence, 2), processing_time
    except Exception as e:
        return False, f"Processing error: {str(e)}", 0,0,0


# =========================================================
# DJANGO UPLOAD HANDLER
# =========================================================
def process_uploaded_images(ref_image, target_image):
    try:
        ref_path = "temp_ref.jpg"
        target_path = "temp_target.jpg"
        output_path = "temp_output.jpg"

        # Save uploaded reference image
        with open(ref_path, "wb") as f:
            for chunk in ref_image.chunks():
                f.write(chunk)

        # Save uploaded target image
        with open(target_path, "wb") as f:
            for chunk in target_image.chunks():
                f.write(chunk)

        # Process images
        success, message, faces_detected,confidence,processing_time = detect_and_blur_faces(
            ref_path, target_path, output_path
        )

        if not success:
            cleanup_files(ref_path, target_path, output_path)
            return False, message, None, 0,0,0

        # Read result image
        with open(output_path, "rb") as f:
            result_content = f.read()

        result_file = ContentFile(result_content, name="result.jpg")

        cleanup_files(ref_path, target_path, output_path)

        return True, message, result_file, faces_detected,confidence,processing_time

    except Exception as e:
        cleanup_files(ref_path, target_path, output_path)
        return False, f"Upload error: {str(e)}", None, 0,0,0


# =========================================================
# IMAGE MATCHING - VERIFY BLURRED PHOTO WITH ORIGINAL
# =========================================================
def match_blurred_with_original(blurred_image_source, original_image_source):
    """
    Verify if a blurred image matches the original image.

    For group originals, it scores each face blur candidate and picks the best-matching person.
    Returns (success, message, similarity, processing_time, face_crop_bytes, face_location).
    """
    try:
        start_time = time.time()

        original = load_image_strict(original_image_source)
        uploaded_blurred = load_image_strict(blurred_image_source)

        # Standardize size for comparison
        if original.shape[:2] != uploaded_blurred.shape[:2]:
            uploaded_blurred = cv2.resize(uploaded_blurred, (original.shape[1], original.shape[0]))

        # Helper: similarity between two RGB images
        def _calculate_similarity(a, b):
            a_gray = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
            b_gray = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
            score = ssim(a_gray, b_gray)
            return ((score + 1) / 2) * 100

        def _encode_jpeg(image_rgb):
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            success, buf = cv2.imencode('.jpg', image_bgr)
            return buf.tobytes() if success else None

        original_faces = face_recognition.face_locations(original)
        uploaded_faces = face_recognition.face_locations(uploaded_blurred)

        best_similarity = -1.0
        best_box = None
        best_crop_bytes = None

        # 1) Prefer direct face-encoding match in groups (blurred-person vs one of the original faces)
        if original_faces and uploaded_faces:
            original_encodings = face_recognition.face_encodings(original, original_faces)
            uploaded_encodings = face_recognition.face_encodings(uploaded_blurred, uploaded_faces)

            if original_encodings and uploaded_encodings:
                best_dist = 1.0
                matched_orig_idx = None
                matched_uploaded_idx = None

                for ui, uploaded_enc in enumerate(uploaded_encodings):
                    distances = face_recognition.face_distance(original_encodings, uploaded_enc)
                    if len(distances) == 0:
                        continue
                    candidate_idx = int(np.argmin(distances))
                    distance = float(distances[candidate_idx])
                    if distance < best_dist:
                        best_dist = distance
                        matched_orig_idx = candidate_idx
                        matched_uploaded_idx = ui

                # Threshold 0.60 is a standard face_recognition threshold for matches
                if matched_orig_idx is not None and best_dist <= 0.60:
                    top, right, bottom, left = original_faces[matched_orig_idx]
                    best_box = (top, right, bottom, left)
                    crop_region = original[top:bottom, left:right].copy()
                    best_crop_bytes = _encode_jpeg(crop_region)
                    best_similarity = max(50.0, (1 - best_dist) * 100)

                    message = f"✅ Matched blurred person to group face #{matched_orig_idx + 1} (distance {best_dist:.3f})."
                    end_time = time.time()
                    return True, message, round(best_similarity, 2), round(end_time - start_time, 4), best_crop_bytes, best_box

        # 2) Fall back to per-face blur-similarity for best candidate face in original
        for (top, right, bottom, left) in original_faces:
            candidate = original.copy()
            face_patch = candidate[top:bottom, left:right]
            if face_patch.size == 0:
                continue
            candidate[top:bottom, left:right] = cv2.GaussianBlur(face_patch, (99, 99), 30)
            sim = _calculate_similarity(candidate, uploaded_blurred)
            if sim > best_similarity:
                best_similarity = sim
                best_box = (top, right, bottom, left)
                crop_region = original[top:bottom, left:right].copy()
                best_crop_bytes = _encode_jpeg(crop_region)

        # Add fallback candidate: full image blur if no faces or if full blur more plausible
        candidate_full = cv2.GaussianBlur(original.copy(), (99, 99), 30)
        full_sim = _calculate_similarity(candidate_full, uploaded_blurred)
        if full_sim > best_similarity:
            best_similarity = full_sim
            best_box = None
            best_crop_bytes = None

        if best_similarity < 20:
            # too low to be reliable
            message = f"❌ No strong match found. Best similarity {round(best_similarity,2)}%."
            end_time = time.time()
            return False, message, round(best_similarity, 2), round(end_time - start_time, 4), None, None

        # More tolerant matching for strongly blurred photos
        match_threshold = 35
        is_match = best_similarity >= match_threshold
        end_time = time.time()
        processing_time = round(end_time - start_time, 4)

        if is_match:
            if best_box is not None:
                message = f"✅ Person match found at face region (best similarity {round(best_similarity,2)}%)."
            else:
                message = f"✅ Image-level blur match found (best similarity {round(best_similarity,2)}%)."
            return True, message, round(best_similarity, 2), processing_time, best_crop_bytes, best_box

        message = f"❌ Images do not match. Best similarity {round(best_similarity,2)}% (< threshold {match_threshold}%)."
        return False, message, round(best_similarity, 2), processing_time, None, None

    except Exception as e:
        return False, f"Matching error: {str(e)}", 0, 0, None, None


# =========================================================
# CLEANUP
# =========================================================
def cleanup_files(*paths):
    for path in paths:
        if os.path.exists(path):
            os.remove(path)
