import streamlit as st
import os
import subprocess
from pipelines.pipeline import InferencePipeline 
from divide import video_process_output
from audioextract import extract_audio, transcribe_audio
from delete_chunks import delete_chunks
from sentence_transformers import SentenceTransformer, util

# Path configurations
video_modality = "video"
v_model_conf = "C:/New folder/new/Visual_Speech_Recognition_for_Multiple_Languages/content/data/LRS3_V_WER19.1/model.json"
v_model_path = "C:/New folder/new/Visual_Speech_Recognition_for_Multiple_Languages/content/data/LRS3_V_WER19.1/model.pth"
video_folder_path = 'C:/New folder/new/Visual_Speech_Recognition_for_Multiple_Languages/content/video'
audio_path = r"C:\New folder\new\Visual_Speech_Recognition_for_Multiple_Languages\content\chunk_dir_audio\audio.wav"
chunk_dir = r'C:\New folder\new\Visual_Speech_Recognition_for_Multiple_Languages\content\chunk_dir'

# Function to get all video files from a folder
def get_video_files(folder_path):
    video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
    return video_files

# Function to process the video and return transcripts and similarity
def process_video(video_path):
    
    video_pipeline = InferencePipeline(video_modality, v_model_path, v_model_conf, face_track=True, device="cpu")
    
    video_transcript,video_detected = video_process_output(video_path, chunk_dir, video_pipeline)
    
    extract_audio(video_path, audio_path)
    audio_transcript, audio_detected = transcribe_audio(audio_path)
    similarity=None
    if audio_detected:
     audio_transcript = audio_transcript.upper() 

    if audio_detected and video_detected:
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        embeddings = model.encode([video_transcript, audio_transcript])
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()* 100
    
   
    
    return video_transcript, audio_transcript, similarity

# Streamlit interface
st.title('LIP SYNC DETECTION FROM RECORDED VIDEOS')




# File uploader
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
if uploaded_file is not None:
    # Save the uploaded file to the video folder
    save_path = os.path.join(video_folder_path, uploaded_file.name)
    
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"File saved.")
    
    # Refresh the video list after uploading
    video_files = get_video_files(video_folder_path)
else:
    # Get video files in the folder
    video_files = get_video_files(video_folder_path)


# Select the video
video_files = get_video_files(video_folder_path)
selected_video = st.selectbox('Select a video', video_files)


if selected_video:
    video_path = os.path.join(video_folder_path, selected_video)
    
    # Display the video
    st.video(video_path)
    
    # Process the video and show transcripts and similarity
    if st.button('Process Video'):
        with st.spinner('Processing...'):
            video_transcript, audio_transcript, similarity = process_video(video_path)
        
        # Display transcripts and similarity
        st.header('Audio Transcript')
        st.write(audio_transcript)
        
        st.header('Video Transcript')
        st.write(video_transcript)

        if similarity is not None:
         st.header('Similarity Score')
         st.write(f"Similarity: {similarity:.2f}%")
         if similarity < 50:
            st.markdown('<p style="color:red;">LIP SYNC DETECTED!</p>' ,unsafe_allow_html=True)
            delete_chunks(chunk_dir,audio_path)
         else:
            st.markdown('<p style="color:green;">LIP SYNC NOT DETECTED!</p>' ,unsafe_allow_html=True)
            delete_chunks(chunk_dir,audio_path)
        else:
           st.title('The Video provided has an issue with the audio or visuals')
           delete_chunks(chunk_dir,audio_path)
           
