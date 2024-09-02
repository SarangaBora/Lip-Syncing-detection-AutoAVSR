import cv2
import os
import time
from pipelines.pipeline import InferencePipeline




def video_process_output(video_path,chunk_dir,pipeline):
    
    os.makedirs(chunk_dir, exist_ok=True)

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)

    # Get video properties
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"FPS:{fps}\nDIMENSION: {frame_width} x {frame_height}\n")

    # Define the number of frames per chunk
    frames_per_chunk = frame_height # Adjust this value as needed
    
    print("FRAMES PER CHUNK = " , frames_per_chunk)
    

    chunk_number = 0
    ret, frame = video_capture.read()

    if not ret:
        print("Error: No frames detected in the video.")
        return "No frames detected",False
    
    print(" CREATING CHUNKS...")

    while ret:
        # Create a new video writer for the chunk
        
    
        chunk_path = f'{chunk_dir}/chunk_{chunk_number}.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(chunk_path, fourcc, fps, (frame_width, frame_height))
    
        
    
        for _ in range(frames_per_chunk):
            if not ret:
                 break
            out.write(frame)
            ret, frame = video_capture.read()

        out.release()
        chunk_number += 1
    print("\nNO. OF CHUNKS CREATED = ",chunk_number)
    video_capture.release()

    if chunk_number == 0:
        print("Error: No faces or frames detected, no chunks created.")
        return "Error: No faces or frames detected, no chunks created.",False

    # Start measuring time for processing each chunk
    start_time = time.time()
    output = ""

    # Sort chunk files in order
    chunk_files = sorted(os.listdir(chunk_dir), key=lambda x: int(x.split('_')[1].split('.')[0]))

    # Process each chunk
    for chunk_file in chunk_files:
        if chunk_file.endswith('.mp4'):
          chunk_path = os.path.join(chunk_dir, chunk_file)
          try:
                print(f"Processing chunk: {chunk_file}")
                result = pipeline(chunk_path)  # Replace with your actual processing function
                
                # Handle multiple values
                if isinstance(result, tuple):
                    transcript, *other_values = result
                    print(f"Transcript: {transcript}")
                    print(f"Other values: {other_values}")
                else:
                    transcript = result

                if isinstance(transcript, str):
                    output += transcript + "\n"
                else:
                    print(f"Unexpected output type from pipeline: {type(transcript)}")
          except Exception as e:
                print(f"Error processing {chunk_file}: {e}")
                return "",False
          

    print("Video Transcript:\n", output)
    print("--- %s seconds ---" % (time.time()-start_time))
    
    for chunk_file in os.listdir(chunk_dir):
     chunk_path = os.path.join(chunk_dir, chunk_file)
     if os.path.isfile(chunk_path):
        os.remove(chunk_path)
    
    return output,True 