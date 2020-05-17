import cv2
import numpy as np

# cap = cv2.VideoCapture('Bright.avi')
cap = cv2.VideoCapture('black_cut_2.mp4')
frame_rate = cap.get(cv2.CAP_PROP_FPS)

_, last_frame = cap.read()
h, w, _ = last_frame.shape
out = cv2.VideoWriter(
    "output.mp4", cv2.VideoWriter_fourcc(*"MJPG"), frame_rate, (w, h))

img_cnt = 0


def drawbox(img, left, top, width, height, color=(0, 0, 255), d=5):
    color = np.array(color, np.uint8)
    img = img.copy()
    img[top-d:top, left-d:left+width+d, :] = color
    img[top+height:top+height+d, left-d:left+width+d, :] = color
    img[top-d:top+height+d, left-d:left, :] = color
    img[top-d:top+height+d, left+width:left+width+d, :] = color
    return img


def diff(a, b):
    global img_cnt
    thres = 70  # 100
    img = b.copy()
    a = a.astype(np.int16)
    b = b.astype(np.int16)
    a = np.sum(a, 2)/3
    b = np.sum(b, 2)/3
    r = np.zeros(b.shape, np.uint8)
    r[(b > a+5)] = 255
    kernel = np.ones((5, 5), np.uint8)
    # r = cv2.morphologyEx(r, cv2.MORPH_CLOSE, kernel)
    r = cv2.morphologyEx(r, cv2.MORPH_OPEN, kernel)
    # return r
    # retval,labels=cv2.connectedComponents(r)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(r)
    for i in range(1, retval):
        l, t, w, h, area = stats[i]
        if area > w*h*0.7 and h > 1.2*w and w > 5:
            region = b[t:t+h, l:l+w]
            fregion = region.astype(np.float32)
            m = np.mean(fregion)
            v = np.var(fregion)
            if m > thres and v > 1 and np.all(region[[0, 0, -1, -1], [0, -1, 0, -1]] < m):
                cv2.imwrite(f'sample/{img_cnt}.png', img[t: t+h, l: l+w, :])
                img = drawbox(img, l, t, w, h)
                print(img_cnt, m, v)
                img_cnt += 1
            # else:
            #     img = drawbox(img, l, t, w, h, (0, 255, 0))

    return img


cnt = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    res = diff(frame, last_frame)
    out.write(res)
    cv2.imshow('diff', cv2.resize(res, (640, 400)))
    # print(cnt)
    cnt += 1
    cv2.waitKey(1)
    last_frame = frame
out.release()
