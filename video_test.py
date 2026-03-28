import cv2

cap = cv2.VideoCapture("video.mp4")

if not cap.isOpened():
    print("Nie moge otworzyc pliku video.mp4")
    raise SystemExit(1)

print("Video OK")
print("FPS:", cap.get(cv2.CAP_PROP_FPS))
print("Rozdzielczosc:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("VIDEO TEST", frame)

    key = cv2.waitKey(30) & 0xFF
    if key in (27, ord("q")):
        break

cap.release()
cv2.destroyAllWindows()
