param(
    [string]$Video = "video.mp4"
)

@"
mode: video
fps: 30

video:
  source: $Video

yolo:
  model: yolov8n.pt
  tracker: bytetrack.yaml
  conf: 0.15
  imgsz: 1280
  classes: [4]
"@ | Set-Content config\config.yaml -Encoding UTF8

py src\main.py
