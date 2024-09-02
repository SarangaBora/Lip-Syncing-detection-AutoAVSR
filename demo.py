import streamlit as st
import cv2
import os
import time
import tempfile
from pipelines.pipeline import InferencePipeline # Assuming this is for visual transcript
import speech_recognition as sr  # For audio transcription
import difflib



modality = "video"
model_conf =  "C:/New folder/new/Visual_Speech_Recognition_for_Multiple_Languages/content/data/LRS3_V_WER19.1/model.json"  
model_path = "C:/New folder/new/Visual_Speech_Recognition_for_Multiple_Languages/content/data/LRS3_V_WER19.1/model.pth"
pipeline = InferencePipeline(modality, model_path, model_conf, face_track=True, device="cpu")
# Function to chunk the video
def chunk_video(video_path, chunk_dir, frames_per_chunk):
    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Get video properties
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    st.write(f"FPS: {fps}, Width: {frame_width}, Height: {frame_height}")

    chunk_number = 0
    ret, frame = video_capture.read()

    while ret:
        # Create a new video writer for the chunk
         # Create a new video writer for the chunk
        chunk_path = os.path.join(chunk_dir, f'chunk_{chunk_number}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(chunk_path, fourcc, fps, (frame_width, frame_height))

        for _ in range(frames_per_chunk):
            if not ret:
                break
            out.write(frame)
            ret, frame = video_capture.read()

        out.release()
        chunk_number += 1

    video_capture.release()

# Function to process the video chunks
def process_chunks(chunk_dir):
    start_time = time.time()
    visual_transcript = ""

    # Sort chunk files in order
    #chunk_files = sorted(os.listdir(chunk_dir))
    
     # Sort chunk files in order
    chunk_files = sorted(os.listdir(chunk_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))

    # Process each chunk
    for chunk_file in chunk_files:
        if chunk_file.endswith('.mp4'):
            chunk_path = os.path.join(chunk_dir, chunk_file)
            try:
                transcript = pipeline(chunk_path)  # Replace with your actual processing function
                visual_transcript += transcript + "  *******  "
            except Exception as e:
                st.error(f"Error processing {chunk_path}: {e}")

    st.write("Visual Transcript:", visual_transcript)
    st.write("--- %s seconds ---" % (time.time() - start_time))
    return visual_transcript

# Function to extract audio from video
def extract_audio(video_path, audio_path):
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    command = f"ffmpeg -i {video_path} -ab 160k -ac 2 -ar 44100 -vn {audio_path}"
    os.system(command)

# Function to transcribe audio
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(audio_path)
    with audio_file as source:
        audio_data = recognizer.record(source)
    text = recognizer.recognize_google(audio_data)
    return text

# Streamlit app title
st.title("Lip Sync Detection Demo App")

# File uploader for video input
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_video_path = temp_file.name
    
    # Display the video file
    st.video(temp_video_path)

    # Directory to save chunks
    chunk_dir = tempfile.mkdtemp()
    frames_per_chunk = 180  # Adjust this value as needed

    # Chunk the video
    st.write("Chunking the video...")
    chunk_video(temp_video_path, chunk_dir, frames_per_chunk)
    
    # Process the video chunks
    with st.spinner("Processing chunks..."):
        visual_transcript = process_chunks(chunk_dir)
    
    # Extract audio from video
    audio_path = temp_video_path.replace('.mp4', '.wav')
    extract_audio(temp_video_path, audio_path)

    # Transcribe audio
    with st.spinner("Transcribing audio..."):
        audio_transcript = transcribe_audio(audio_path)
    
    st.write("Audio Transcript:", audio_transcript)

    # Compare transcripts
    st.write("Comparing Transcripts...")

    # Compute the similarity ratio
    matcher = difflib.SequenceMatcher(None, visual_transcript.lower(), audio_transcript)
    similarity_ratio = matcher.ratio() * 100

    # Display the result
    st.write(f"Transcript Similarity: {similarity_ratio:.2f}%")

    if similarity_ratio >= 90:
         st.success("Transcripts match closely!")
    elif similarity_ratio >= 70:
         st.warning("Transcripts have some differences.")
    else:
         st.error("Transcripts do not match!")

    # Cleanup: Remove temporary video file, audio file, and chunk directory
    os.remove(temp_video_path)
    os.remove(audio_path)
    for chunk_file in os.listdir(chunk_dir):
        os.remove(os.path.join(chunk_dir, chunk_file))
    os.rmdir(chunk_dir)
