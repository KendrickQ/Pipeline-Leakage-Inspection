import cv2
import numpy as np

# cap = cv2.VideoCapture('Dark2.mp4')
cap = cv2.VideoCapture('Bright.avi')

_, last_frame = cap.read()
h, w, _ = last_frame.shape
out = cv2.VideoWriter(
    "output.mp4", cv2.VideoWriter_fourcc(*"MJPG"), 20, (w, h))

img_cnt = 0


def drawbox(img, left, top, width, height, d=5):
    color = np.array([0, 0, 255], np.uint8)
    img = img.copy()
    # img[top:top+d, left:left+width, :] = color
    # img[top+height-d:top+height, left:left+width, :] = color
    # img[top:top+height, left:left+d, :] = color
    # img[top:top+height, left+width-d:left+width, :] = color
    img[top-d:top, left-d:left+width+d, :] = color
    img[top+height:top+height+d, left-d:left+width+d, :] = color
    img[top-d:top+height+d, left-d:left, :] = color
    img[top-d:top+height+d, left+width:left+width+d, :] = color
    return img


def diff(a, b):
    # r = a-b
    # r[b > a] = -r[b > a]
    # r[r > 50] = 255
    # return r
    global img_cnt
    thres = 150
    img = b.copy()
    a = a.astype(np.int16)
    b = b.astype(np.int16)
    a = np.sum(a, 2)/3
    b = np.sum(b, 2)/3
    r = np.zeros(b.shape, np.uint8)
    r[(b > 100) & (b > a+10)] = 255
    # return r
    kernel = np.ones((5, 5), np.uint8)
    r = cv2.morphologyEx(r, cv2.MORPH_CLOSE, kernel)
    # retval,labels=cv2.connectedComponents(r)
    retval, labels, stats, centroids = cv2.connectedComponentsWithStats(r)
    for i in range(1, retval):
        l, t, w, h, area = stats[i]
        if area > w*h*0.7 and h > 1.2*w and w > 5:
            region = b[t:t+h, l:l+w]
            if np.any(region > 150) and region[0,0]<thres and region[0,-1]<thres and region[-1,0]<thres and region[-1,-1]<thres:
                img = drawbox(img, l, t, w, h)
                # cv2.imwrite(f'sample/{img_cnt}.png', img[t: t+h, l: l+w, :])
                print(img_cnt)
                img_cnt += 1

    # b[r < 128] = 0
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
