import cv2

cascade_path = "cascade.xml"
cascade = cv2.CascadeClassifier(cascade_path)

def detect(img_gray):
    rects = cascade.detectMultiScale(img_gray, minSize=(30, 30))

    # `rects` is () when no objects detected
    if len(rects) == 0:
        return []

    # convert [[x1, y1, width, height]] to [[x1, y1, x2, y2]]
    rects[:, 2:] += rects[:, :2]
    return rects
