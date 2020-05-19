import cv2
import numpy as np

# cap = cv2.VideoCapture('Bright.avi')
cap = cv2.VideoCapture('black_cut_2.mp4')
frame_rate = cap.get(cv2.CAP_PROP_FPS)

_, last_frame = cap.read()
h, w, _ = last_frame.shape
out = cv2.VideoWriter(
    "output.mp4", cv2.VideoWriter_fourcc(*"MJPG"), frame_rate, (w*2, h))

img_cnt = 0


def drawbox(img, left, top, width, height, color=(0, 0, 255), d=5):
    color = np.array(color, np.uint8)
    img = img.copy()
    img[top-d:top, left-d:left+width+d, :] = color
    img[top+height:top+height+d, left-d:left+width+d, :] = color
    img[top-d:top+height+d, left-d:left, :] = color
    img[top-d:top+height+d, left+width:left+width+d, :] = color
    return img


def is_water(r):
    thres = 70  # 100
    fregion = r.astype(np.float32)
    m = np.mean(fregion)
    v = np.var(fregion)
    return m > thres and v > 1 and np.all(fregion[[0, 0, -1, -1], [0, -1, 0, -1]] < m)


def diff(a, b):
    global img_cnt
    img = b.copy()
    a = a.astype(np.int16)
    b = b.astype(np.int16)
    a = np.sum(a, 2)/3
    b = np.sum(b, 2)/3
    r = np.zeros(b.shape, np.uint8)
    r[(b > a+5)] = 255
    # r = cv2.morphologyEx(r, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    r = cv2.morphologyEx(r, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    # return r
    # retval,labels=cv2.connectedComponents(r)
    r3 = np.repeat(r.reshape(1, -1), 3).reshape(*r.shape, 3)
    retval, _, stats, _ = cv2.connectedComponentsWithStats(r)
    for i in range(1, retval):
        l, t, w, h, area = stats[i]
        # r3 = cv2.putText(r3, '%.3f' % (area/w/h), (l, t), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if area > w*h*0.5 and h>1.2*w and w > 3:
            if is_water(b[t:t+h, l:l+w]):
                # cv2.imwrite(f'sample/{img_cnt}.png', img[t: t+h, l: l+w, :])
                img = drawbox(img, l, t, w, h, (0, 255, 0))
                img_cnt += 1
            else:
                img = drawbox(img, l, t, w, h, (0, 0, 255))
    return np.concatenate([r3, img], 1)


cnt = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    res = diff(frame, last_frame)
    out.write(res)
    cv2.imshow('diff', cv2.resize(res, (800, 640)))
    # print(cnt)
    cnt += 1
    cv2.waitKey(1)
    last_frame = frame
out.release()
