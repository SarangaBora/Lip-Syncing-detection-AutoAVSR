import cv2 
import speech_recognition as sr  # For audio transcription
import os 
from moviepy.editor import VideoFileClip




# # Extract audio from video
# video_clip = VideoFileClip(video_path)
# video_clip.audio.write_audiofile(audio_output_path,codec='pcm_s16le')



# Function to extract audio from video
def extract_audio(video_path, audio_path):
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps

    command = f'ffmpeg -i "{video_path}" -ab 160k -ac 2 -ar 44100 -vn "{audio_path}"'
    print("FPS :", fps )
    print("FrameCount :", frame_count )
    print("Duration :", duration )
    os.system(command)


# Function to transcribe audio
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    audio_file = sr.AudioFile(audio_path)
    with audio_file as source:
        audio_data = recognizer.record(source)
    
    try: 
     text = recognizer.recognize_google(audio_data)
     return text,True
    except sr.UnknownValueError:
       return "No audio",False
    except sr.RequestError as e:
       return f"Could not request results from the service;{e}",False



 