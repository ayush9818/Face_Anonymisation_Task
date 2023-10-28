
# Face Anonymisation

This repository contains the scripy to process video stream, detect faces from the video and blur the faces to ananoymise them.



## Run Locally

Clone the project

```bash
  git clone https://github.com/ayush9818/Face_Anonymisation_Task.git
```

Go to the project directory

```bash
  cd Face_Anonymisation_Task
```

Setup Virtual Environment and install dependencies

```bash
  python3 -m venv face_env
  source face_env/bin/activate
  pip install -r requirements.txt
```

Running the script

- To run without multi threading
```bash
  python face_detect_insightface.py --input-video <path to input video> \
  --output-video <path to output video> \
  --detection-threshold <threshold b/w 0 to 1>
```

- To run with multi threading
```bash
  python face_detect_insightface.py --input-video <path to input video> \
  --output-video <path to output video> \
  --detection-threshold <threshold b/w 0 to 1> \
  --num-workers <worker count usually b/w 1 to 10>
```



## Usage/Examples

- To run without multi threading
```bash
python face_detect_insightface.py \
    --input-video streams/sample.mp4 \
    --output-video streams/output_test.mp4 \
    --detection-threshold 0.25
```

- To run with multi threading
```bash
python face_detect_insightface_parallel.py \
    --input-video streams/sample.mp4 \
    --output-video streams/output_test.mp4 \
    --detection-threshold 0.25 \
    --num-workers 10
```


## Sample Output

![Alt Text](outputs/sample-output-gif.gif)

