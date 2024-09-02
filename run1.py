##Self made w
from pipelines.pipeline import InferencePipeline 
from divide import video_process_output
from audioextract import extract_audio,transcribe_audio
from delete_chunks import delete_chunks
from difflib import SequenceMatcher
from sentence_transformers import SentenceTransformer, util

import os


#pipeline config
video_modality = "video"
v_model_conf = "C:/New folder/new/Visual_Speech_Recognition_for_Multiple_Languages/content/data/LRS3_V_WER19.1/model.json"
v_model_path = "C:/New folder/new/Visual_Speech_Recognition_for_Multiple_Languages/content/data/LRS3_V_WER19.1/model.pth"

#paths to video/audio
video_path = 'C:/New folder/new/Visual_Speech_Recognition_for_Multiple_Languages/content/video/audioless.mp4' # Path to the video file(pre-exixting)
audio_path = r"C:\New folder\new\Visual_Speech_Recognition_for_Multiple_Languages\content\chunk_dir_audio\audio.wav" #Path to the extracted audio(created)
chunk_dir = r'C:\New folder\new\Visual_Speech_Recognition_for_Multiple_Languages\content\chunk_dir' # Directory to save chunks(created)

# # Function to get all video files from a folder
# def get_video_files(folder_path):
#     video_files = [f for f in os.listdir(folder_path) if f.endswith('.mp4')]
#     return video_files



if __name__== "__main__":
 
 #Pipelineee for video processing
 video_pipeline = InferencePipeline(video_modality, v_model_path, v_model_conf, face_track=True, device="cpu")
 
#  for filename in os.listdir(video_path):
#     if filename.endswith(".mp4"):
#         video_path = os.path.join(video_path, filename)
#         print(f"Video file found: {video_path}")
        
#         video_transcript,video_detected=video_process_output(video_path,chunk_dir,video_pipeline)
        
#         extract_audio(video_path, audio_path)
#         audio_transcript, audio_detected =transcribe_audio(audio_path)
        # if audio_detected:
        #   audio_transcript=audio_transcript.upper()
#         print("\n\nAudio Transcript :\n",audio_transcript)

#         if video_detected and audio_detected:
#          model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#          embeddings = model.encode([video_transcript, audio_transcript])
#          similarity = util.cos_sim(embeddings[0], embeddings[1]).item()*100
#          print(f"Similarity Score: {similarity:.2f}%")
#          if similarity<50:
#            print("Lip Syncing detected!")
#          else:
#            print("Lip Syncing Not Detected!")
#         else:
#           print('The Video provided has an issue with the audio or visuals')
#           delete_chunks(chunk_dir,audio_path)
          
                 
#     else:
#      print("No video found.")


#FOR A PARTICULAR VIDEO
#  video_path = os.path.join(video_path, filename)
 print(f"Video file found: {video_path}")
        
 video_transcript,video_detected=video_process_output(video_path,chunk_dir,video_pipeline)
        
 extract_audio(video_path, audio_path)
 audio_transcript, audio_detected =transcribe_audio(audio_path)
 if audio_detected:
  audio_transcript=audio_transcript.upper()
 print("\n\nAudio Transcript :\n",audio_transcript)

 if video_detected and audio_detected:
  model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
  embeddings = model.encode([video_transcript, audio_transcript])
  similarity = util.cos_sim(embeddings[0], embeddings[1]).item()*100
  print(f"Similarity Score: {similarity:.2f}%")
  if similarity<50:
    print("Lip Syncing detected!")
    delete_chunks(chunk_dir,audio_path)
  else:
    print("Lip Syncing Not Detected!")
    delete_chunks(chunk_dir,audio_path)
 else:
  print('The Video provided has an issue with the audio or visuals')
  delete_chunks(chunk_dir,audio_path)
  

   
 

   
   
   

 
 





 

# config_filename="C:/New folder/new/Visual_Speech_Recognition_for_Multiple_Languages/hydra_configs/default.yaml"

# # config_filename, detector="mediapipe", face_track=False, device="cpu"
# pipeline = InferencePipeline(config_filename,detector="mediapipe",face_track=True,device="cpu")
# # pipeline = InferencePipeline(modality, model_path, model_conf)

# transcript = pipeline("C:/New folder/Visual_Speech_Recognition_for_Multiple_Languages/content/video/longtest.mp4")  # removed the noisy
# print(transcript)



# modality = "video"
# model_conf = "/content/data/LRS3_V_WER19.1/model.json"
# model_path = "/content/data/LRS3_V_WER19.1/model.pth"
# pipeline = InferencePipeline(modality, model_path, model_conf, face_track=True)

