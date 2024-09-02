# Clean up by deleting the chunk files
import os

def delete_chunks(video_chunk,audio_path):


 for chunk_file in os.listdir(video_chunk):
    chunk_path = os.path.join(video_chunk, chunk_file)
    if os.path.isfile(chunk_path):
        os.remove(chunk_path)
 
 os.remove(audio_path)