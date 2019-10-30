import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt


windowName = "Chroma keying"
trackbarPatch = "Color Patch Selector"
trackbarTolerance = "Tolerance %"
trackbarSoftness = "Softness %"
trackbarCast = "Color cast"

cap = cv2.VideoCapture('greenscreen-demo.mp4')
ret, frame = cap.read()
# load an image
im = frame# cv2.imread("demo.jpg")
cap.release()
if im.shape[1]/2 > 700:
    im = cv2.resize(im, None, fx=700/im.shape[1], fy=700/im.shape[1], interpolation = cv2.INTER_CUBIC)
image = im.copy()
imHSV = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
back = cv2.imread("back2.jpg")
back = cv2.resize(back, (im.shape[:2][::-1]), interpolation = cv2.INTER_CUBIC)
button = 255 * np.ones((50, im.shape[1], im.shape[2]), dtype=np.uint8)
cv2.putText(button, 'Chose Color Patch', (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color=(0, 0, 0))
col = (0, 0, 0)
colHsv = [0, 0, 0]
cv2.rectangle(button, (350, 45), (385, 10), color=col, thickness=-1)
cv2.putText(button, 'Apply', (405, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color=(0, 0, 0))
cv2.rectangle(button, (400, 45), (490, 10), color=col, thickness=1)
im = cv2.vconcat([button, im])
g = np.linspace(0, 1, im.shape[1])
grad = np.tile(g, (50, 1))
grad = cv2.merge([np.uint8(100*grad), np.uint8(255*grad), np.uint8(100*grad)])
chose = cv2.vconcat([grad, image])
mask = np.zeros_like(image)
tolArg = 0
softArg = 0
castArg = 0
# Create a window to display results
cv2.namedWindow(windowName, cv2.WINDOW_AUTOSIZE)
cv2.imshow(windowName, im)

def selectPatch(action, x, y, flags, userdata):
  # Referencing global variables
  if action == cv2.EVENT_LBUTTONDOWN:
    global col, colHSV
    b = int(np.mean(chose[np.clip(y-17, 0, image.shape[0]):np.clip(y+17, 0, image.shape[1]),
                    np.clip(x-17, 0, image.shape[0]):np.clip(x+17, 0, image.shape[1]), 0]))
    g = int(np.mean(chose[np.clip(y-17, 0, image.shape[0]):np.clip(y+17, 0, image.shape[1]),
                    np.clip(x-17, 0, image.shape[0]):np.clip(x+17, 0, image.shape[1]), 1]))
    r = int(np.mean(chose[np.clip(y-17, 0, image.shape[0]):np.clip(y+17, 0, image.shape[1]),
                    np.clip(x-17, 0, image.shape[0]):np.clip(x+17, 0, image.shape[1]), 2]))
    col = (b, g, r)
    cv2.rectangle(im, (350, 45), (385, 10), color=col, thickness=-1)
    colHSV = cv2.cvtColor(np.array([[[b, g, r]]],dtype=np.uint8), cv2.COLOR_BGR2HSV)[0, 0]
    changeIm()



def beckDetect(hsv, rng = np.array([0 , 180])):
    imHSV = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    h = imHSV[:, :, 0].astype(float)
    if hsv == 0:
        h = h/180.0
    elif hsv == 1:
        h = imHSV[np.isin(h, np.arange(rng[0], rng[1]+1))][:, hsv].astype(float)
        h = h / 256.0
    elif hsv == 2:
        h = imHSV[np.isin(h, np.arange(rng[0], rng[1]+1))][:, hsv].astype(float)
        h = h / 256.0
    M = 100
    r =  1/M
    c = np.linspace(0,1-r, M)
    hist, bin_edges = np.histogram(h,bins=np.append(c,[1]), density=False, range=(0,r))
    p = np.column_stack((hist, c))
    p = p[np.argsort(p[:,0])]
    CR = np.array([], dtype=float)
    v0 = np.sum((p[:, 0] - np.mean(p[:, 0]) )** 2) / p.shape[0]
    v1 = v0
    vamp = 0.0001
    tgrad = 0.1
    #plt.plot(c, hist)
    #plt.hist(bin_edges[:-1], bin_edges, weights=hist)
    CH = np.array([], dtype=int)
    pp = p.copy()
    while True:
        v = np.sum((p[:, 0] - np.mean(p[:, 0]))**2)/p.shape[0]
        #print((v < vamp * v0 , abs(v - v1)/v1 < tgrad))
        if (v < vamp * v0 and abs(v - v1)/v1 < tgrad) or len(p) <= 1:
            break
        CR = np.append(CR, p[-1][1])
        CH = np.append(CH, p[-1][0])
        p = p[:-1]
        v1 = v

    #plt.plot(CR , CH, 'ro')
    #plt.show()

    pC = np.column_stack((CH, CR))
    pC = pC[np.argsort(pC[:, 1])]
    CR.sort()
    mR = pC[pC[:,0] == np.max(pC[:,0])][0, 1]
    ind = np.where(CR == mR)[0][0]
    Range = np.array([CR[ind]])
    for i in range(ind, len(CR) - 2):
        if abs(CR[(i + 1)] - CR[i]) < r*2:
            Range = np.append(Range, CR[i+1])
        else:
            break
    for i in reversed(range(1, ind+1)):
        if abs(CR[(i - 1)] - CR[i]) <= r*2:
            Range = np.append(Range, CR[i-1])
        else:
            break
    Range.sort()
    if hsv == 0:
        Range = Range*180.0
        mx = Range[-1] + (tolArg * (180 - Range[-1])) / 100
    elif hsv == 1:
        Range = Range * 256.0
        mx = Range[-1] + (tolArg * (255 - Range[-1])) / 100
    elif hsv == 2:
        Range = Range * 256.0
        mx = Range[-1] + (tolArg * (255 - Range[-1])) / 100
    mn = Range[0] - (tolArg*Range[0])/100
    return [np.floor(mn), np.ceil(mx)]


def alphaBG(br, sf, rf):
    unknown = 255- br - sf  - rf
    #imHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #kernel = np.array([[1,1,1], [1,0,1], [1,1,1]])/8
    #bg = cv2.filter2D(cv2.bitwise_and(imHSV, cv2.merge([br,br,br])), -1, kernel, (-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
    #fg = cv2.filter2D(cv2.bitwise_and(imHSV, cv2.merge([sf+rf,sf+rf,sf+rf])), -1, kernel, (-1, -1), delta=0, borderType=cv2.BORDER_DEFAULT)
    uk = cv2.bitwise_and(image, cv2.merge([unknown, unknown, unknown]))
    cond = np.abs(np.sum(uk[:,:] - col, axis=2)) < 20*tolArg/100
    new = np.zeros_like(unknown)
    new[cond] = 255
    return cv2.bitwise_and(unknown, new)


def selectPatchButton(action, x, y, flags, userdata):
  # Referencing global variables
  if action == cv2.EVENT_LBUTTONDOWN and 10 < y < 45 and 350 < x < 385:
      cv2.destroyWindow('Sample')
      cv2.imshow('Chose', chose)
      cv2.setMouseCallback('Chose', selectPatch)
  if action == cv2.EVENT_LBUTTONDOWN and 10 < y < 45 and 400 < x < 490:
      global im, image, imHSV, back
      cap = cv2.VideoCapture('greenscreen-demo.mp4')
      frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
      frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

      # Define the codec and create VideoWriter object.
      # The output is stored in 'outputChaplin.avi' file.
      out = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 25,
                            (frame_width, frame_height))

      cv2.destroyAllWindows()
      wait = np.ones_like(image)*255
      cv2.putText(wait, 'Wait', (image.shape[1]//2, image.shape[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color=(0, 0, 0))
      cv2.imshow('Wait', wait)
      i = 1
      f = False
      while (cap.isOpened()):
          # Capture frame-by-frame
          ret, frame = cap.read()
          im = frame  # cv2.imread("demo.jpg")
          image = im.copy()
          imHSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
          
          if f:
              back = cv2.imread("back1.jpg")
          else:
              back = cv2.imread("back2.jpg")
          back = cv2.resize(back, (im.shape[:2][::-1]), interpolation=cv2.INTER_CUBIC)
          button = 255 * np.ones((50, im.shape[1], im.shape[2]), dtype=np.uint8)
          im = cv2.vconcat([button, im])
          if ret == True:
              fr = changeIm(1)
              out.write(fr)
              i = i + 1
              if i > 50:
                  f = True if not f else False
                  i = 1
          # Break the loop
          else:
              break
      cap.release()
      out.release()
      cv2.destroyAllWindows()
      sys.exit()

# Callback functions
def setTol(*args):
    global tol, tolArg
    tol = np.array([int(args[0]*180/100), int(args[0]*255/100), int(args[0]*255/100)], dtype=np.uint8)
    tolArg = args[0]
    changeIm(Tol = 1)

def setSoft(*args):
    global soft, softArg
    soft = int(round(int(args[0] * (min(image.shape[0], image.shape[1])) / 100)))
    softArg = args[0]
    changeIm()

def setCast(*args):
    global cast, castArg
    castArg = args[0]
    changeIm()

hr = beckDetect(0)
sr = beckDetect(1, hr)
vr = beckDetect(2, hr)

def changeIm(isVidio = 0, Tol = 0):
    global hr, sr, vr
    if Tol:
        hr = beckDetect(0)
        sr = beckDetect(1, hr)
        vr = beckDetect(2, hr)

    t = 10 * 1 / 180
    fr = [0, hr[0] - t * 180, hr[1] + t * 180, 180] if hr[0] > t * 180 and hr[1] < 180 - t * 180 else \
        [hr[1] + t * 180 - 180, hr[0] - t * 180, 180, 180] if hr[1] > 180 - t * 180 else [hr[1] + t * 180,
                                                                                          180 + hr[0] - t * 180, 180,
                                                                                          180]
    fr.sort()
    FRmask = cv2.inRange(imHSV, np.array([fr[0], 0, 0], dtype=int), np.array([fr[1], 255, 255], dtype=int)) + \
             cv2.inRange(imHSV, np.array([fr[2], 0, 0], dtype=int), np.array([fr[3], 255, 255], dtype=int))

    vCurve = (np.array([0.1, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, 1]) * 255)
    sCurve = (np.array([1, 0.4, 0.3, 0.3, 0.2, 0.2, 0.15, 0.15]) * 255)
    fullRange = np.arange(0, 256)
    sLUT = np.interp(fullRange, vCurve, sCurve)
    GC = np.clip(sLUT[imHSV[:, :, 2]] - imHSV[:, :, 1].astype(int), 0, 255).astype(np.uint8)
    ret, GC = cv2.threshold(GC, 0.05 * 255, 255, type=cv2.THRESH_BINARY)
    inr = cv2.inRange(imHSV, np.array([hr[0], sr[0], vr[0]], dtype=int), np.array([hr[1], sr[1], vr[1]], dtype=int))
    unkBG = alphaBG(inr, GC, FRmask)
    mask = cv2.merge([inr + unkBG, inr + unkBG, inr + unkBG])
    cast = (castArg / 100)
    rf = cv2.bitwise_and(image, 255 - mask)
    bb = rf[:, :, 0]
    gg = rf[:, :, 1]
    rr = rf[:, :, 2]
    mn = (np.min([bb, rr], axis=0))
    mx = (np.max([bb, rr], axis=0))
    dvs = mx - mn
    maskAply = (cv2.bitwise_and(image, 255 - mask))
    maskAply[:,:,1] = np.where(mx<gg, np.clip(mx - (2*dvs*cast), 0, 255).astype(np.uint8), gg)
    soft = int(round(int(softArg * (min(image.shape[0], image.shape[1])) / 100)))
    if softArg > 2:
        if soft % 2 == 0:
            soft = soft + 1
        mask1 = cv2.erode(255 - mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, \
                                                                (int(softArg * image.shape[0] / 500),
                                                                 int(softArg * image.shape[1] / 450))))
        mask1 = cv2.GaussianBlur(mask1, (soft, soft), 0, 0)
        mask1 = (mask1).astype(float) / 255.0
        mask1 = np.where(mask == 0, mask1, 1 - mask.astype(float) / 255)
        background = cv2.multiply((1 - mask1), back.astype(float)) / 255.0
        maskAply = cv2.multiply(mask1, maskAply.astype(float)) / 255.0
        sample = (cv2.add(maskAply, background)*255).astype(np.uint8)
        im[50:, :, :] = maskAply[:, :, :] * 255
    else:
        background = cv2.bitwise_and(back, mask)
        sample = cv2.add(maskAply, background)
        im[50:, :, :] = maskAply

    if not isVidio:
        cv2.destroyWindow('Chose')
        cv2.imshow('Sample', sample)
        cv2.imshow(windowName, im)
    return sample

cv2.setMouseCallback(windowName, selectPatchButton)
cv2.createTrackbar(trackbarTolerance, windowName, 1, 100, setTol)
cv2.createTrackbar(trackbarSoftness, windowName, 1, 100, setSoft)
cv2.createTrackbar(trackbarCast, windowName, 1, 100, setCast)

while True:
    c = cv2.waitKey(20)
    if c==27:
        break

cv2.destroyAllWindows()