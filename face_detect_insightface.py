import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import os
import warnings
import numpy as np
import mmcv, cv2
import argparse

warnings.filterwarnings('ignore')

DETECTION_SIZE = (640, 1280)

def read_and_process_video_frames(input_stream, model):
    """
    The function reads the video frames from the input video stream and perform following operations
        a) Detects faces from each face using Insightface's Buffalo_l model
        b) For each detected face, apply gaussian blur 
    args:
        input_stream: Input video stream path 
        model: face detection modle object 
    returns:
        frames_tracked: list of frames with blurred detected faces
    """
    # Reading the input video strea
    video = mmcv.VideoReader(input_stream)
    frames = [frame for frame in video]
    frames_tracked = []
    for i, frame in enumerate(frames):
        print('\rTracking frame: {}'.format(i + 1), end='')
        
        # Detect faces
        output = model.get(frame)
        
        # Draw rectangles around detected face and blur each face
        frame_draw = frame.copy()
        if len(output) != 0:
            for entry in output:
                bbox = entry.get('bbox')
                x1, y1, x2, y2 = [  int(_) for _ in bbox]
                try:
                    face = frame_draw[y1:y2, x1:x2]
                    frame_draw[y1:y2, x1:x2] = cv2.GaussianBlur(face, (99, 99), 30)
                    cv2.rectangle(frame_draw, (int(x1), int(y1)), (int(x2), int(y2)), color=(255,0,0), thickness=2)
                except Exception as e:
                    pass
        # Add to frame list
        frames_tracked.append(frame_draw)
    print('\nDone')
    return frames_tracked

def save_video_frames(frames_tracked, output_video):
    """
    The function creates a video from the tracked frames and saves into the file specified
    args:
        frames_tacked: list of tracked frames 
        output_video: output video path 
    """
    dim = frames_tracked[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'FMP4')  
    out = cv2.VideoWriter(output_video, fourcc, 25.0, (dim[1], dim[0]))
    for frame in frames_tracked:
        out.write(frame)
    out.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-video", help='path to input video')
    parser.add_argument("--output-video", help='path to output video')
    parser.add_argument("--detection-threshold",type=float,default=0.3, help='detection threshold')

    args = parser.parse_args()

    assert os.path.exists(args.input_video), f"Input Video Path {args.input_video} does not exist"

    output_video_path = args.output_video
    output_video_dir = os.path.dirname(output_video_path)
    if len(output_video_dir) == 0:
            output_video_dir = os.getcwd()
    os.makedirs(output_video_dir, exist_ok=True)

    print(f"Input Video Path : {args.input_video}")
    print(f"Output Video Path : {args.output_video}")
    print(f"Detection Threshold : {args.detection_threshold}")

    # Setting up detection model 
    app = FaceAnalysis(allowed_modules=['detection']) # enable detection model only
    app.prepare(ctx_id=0, det_thresh=args.detection_threshold, det_size=DETECTION_SIZE)

    # Processing and Saving Video 
    frames_tracked = read_and_process_video_frames(input_stream=args.input_video, model=app)
    save_video_frames(frames_tracked, output_video_path)