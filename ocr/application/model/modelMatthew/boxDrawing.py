import cv2

def preprocess(image):
    imageLoad = cv2.imread(image)
    gray = cv2.cvtColor(imageLoad, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("UploadedFiles/gray.png", gray)

    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    cv2.imwrite("UploadedFiles/gray_blurred.png", blur)

    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cv2.imwrite("UploadedFiles/thresh.png", thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 13))
    dilate = cv2.dilate(thresh, kernel, iterations=1)
    cv2.imwrite("UploadedFiles/dilate.png", dilate)

    contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    boxes = [cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] > 1 and cv2.boundingRect(c)[3] > 1]
    img_height = imageLoad.shape[0]
    tolerance = int(0.10 * img_height)

    # Remove nested boxes
    filtered_boxes = []
    for i, (x1, y1, w1, h1) in enumerate(boxes):
        inside = False
        for j, (x2, y2, w2, h2) in enumerate(boxes):
            if i != j:
                if x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2:
                    inside = True
                    break
        if not inside:
            filtered_boxes.append((x1, y1, w1, h1))

    # Merge nearby boxes
    def boxes_are_close(b1, b2, thresh=15):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        return not (x1 + w1 + thresh < x2 or x2 + w2 + thresh < x1 or y1 + h1 + thresh < y2 or y2 + h2 + thresh < y1)

    def merge_boxes(b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        x = min(x1, x2)
        y = min(y1, y2)
        x_max = max(x1 + w1, x2 + w2)
        y_max = max(y1 + h1, y2 + h2)
        return (x, y, x_max - x, y_max - y)

    merged = True
    while merged:
        merged = False
        new_boxes = []
        skip = set()
        for i in range(len(filtered_boxes)):
            if i in skip:
                continue
            box1 = filtered_boxes[i]
            for j in range(i + 1, len(filtered_boxes)):
                if j in skip:
                    continue
                box2 = filtered_boxes[j]
                if boxes_are_close(box1, box2):
                    box1 = merge_boxes(box1, box2)
                    skip.add(j)
                    merged = True
            new_boxes.append(box1)
        filtered_boxes = new_boxes

    # Sort top to bottom, then left to right within row
    def sort_key(box):
        return (box[1] // tolerance, box[0])

    sorted_boxes = sorted(filtered_boxes, key=sort_key)

    img_w, img_h = imageLoad.shape[1], imageLoad.shape[0]

    def is_horizontal_line(box):
        x, y, w, h = box
        aspect_ratio = w / h if h > 0 else 0
        return h <= 15 and aspect_ratio > 10  # heuristic for lines

    # Filter small boxes
    final_boxes = [(x, y, w, h) for (x, y, w, h) in sorted_boxes if w >= 15 and h >= 15]

    for idx, (x, y, w, h) in enumerate(final_boxes, start=1):
        roi = imageLoad[y:y + h, x:x + w]
        color = (0, 0, 255) if is_horizontal_line((x, y, w, h)) else (36, 255, 12)
        cv2.imwrite(f"UploadedFiles/roi{idx}.png", roi)
        cv2.rectangle(imageLoad, (x, y), (x + w, y + h), color, 2)

    cv2.imwrite("UploadedFiles/boxed.png", imageLoad)

