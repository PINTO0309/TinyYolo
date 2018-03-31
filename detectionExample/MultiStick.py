import sys
graph_folder="./"
if sys.version_info.major < 3 or sys.version_info.minor < 4:
    print("Please using python3.4 or greater!")
    exit(1)

if len(sys.argv) > 1:
    graph_folder = sys.argv[1]

from mvnc import mvncapi as mvnc
import numpy as np
import cv2
from os import system
import io, time
from os.path import isfile, join
from queue import Queue
from threading import Thread, Event, Lock
import re
from time import sleep
from Visualize import *
from libpydetector import YoloDetector

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

mvnc.SetGlobalOption(mvnc.GlobalOption.LOG_LEVEL, 2)

devices = mvnc.EnumerateDevices()
if len(devices) == 0:
    print("No devices found")
    quit()
print(len(devices))

devHandle   = []
graphHandle = []

with open(join(graph_folder, "graph"), mode="rb") as f:
    graph = f.read()

for devnum in range(len(devices)):
    devHandle.append(mvnc.Device(devices[devnum]))
    devHandle[devnum].OpenDevice()

    graphHandle.append(devHandle[devnum].AllocateGraph(graph))
    graphHandle[devnum].SetGraphOption(mvnc.GraphOption.ITERATIONS, 1)
    iterations = graphHandle[devnum].GetGraphOption(mvnc.GraphOption.ITERATIONS)

    dim = (320,320)
    blockwd = 9
    targetBlockwd = 9
    wh = blockwd*blockwd
    classes = 20 
    threshold = 0.3
    nms = 0.4

print("\nLoaded Graphs!!!")

cam = cv2.VideoCapture(0)
#cam = cv2.VideoCapture('/home/pi/TinyYolo/detectionExample/xxxx.mp4')

if cam.isOpened() != True:
    print("Camera/Movie Open Error!!!")
    quit()

widowWidth = 320
windowHeight = 240
cam.set(cv2.CAP_PROP_FRAME_WIDTH, widowWidth)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, windowHeight)

lock = Lock()
frameBuffer = []
results = Queue()
lastresults = None
detector = YoloDetector(1)


def init():
    glClearColor(0.7, 0.7, 0.7, 0.7)

def idle():
    glutPostRedisplay()

def resizeview(w, h):
    glViewport(0, 0, w, h)
    glLoadIdentity()
    glOrtho(-w / 1920, w / 1920, -h / 1080, h / 1080, -1.0, 1.0)

def keyboard(key, x, y):
    key = key.decode('utf-8')
    if key == 'q':
        lock.acquire()
        while len(frameBuffer) > 0:
            frameBuffer.pop()
        lock.release()
        for devnum in range(len(devices)):
            graphHandle[devnum].DeallocateGraph()
            devHandle[devnum].CloseDevice()
        print("\n\nFinished\n\n")
        sys.exit()


def camThread():   
    global lastresults

    s, img = cam.read()

    if not s:
        print("Could not get frame")
        return 0

    lock.acquire()
    if len(frameBuffer)>10:
        for i in range(10):
            del frameBuffer[0]
    frameBuffer.append(img)
    lock.release()
    res = None

    if not results.empty():
        res = results.get(False)
        if res == None:
            if lastresults == None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = img.shape[:2]
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, img)
            else:
                imdraw = Visualize(img, lastresults)
                imdraw = cv2.cvtColor(imdraw, cv2.COLOR_BGR2RGB)
                h, w = imdraw.shape[:2]
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, imdraw)
        else:
            img = Visualize(img, res)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, img)
            lastresults = res
    else:
        if lastresults == None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w = img.shape[:2]
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, img)
        else:
            imdraw = Visualize(img, lastresults)
            imdraw = cv2.cvtColor(imdraw, cv2.COLOR_BGR2RGB)
            h, w = imdraw.shape[:2]
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, w, h, 0, GL_RGB, GL_UNSIGNED_BYTE, imdraw)

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glColor3f(1.0, 1.0, 1.0)
    glEnable(GL_TEXTURE_2D)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glBegin(GL_QUADS) 
    glTexCoord2d(0.0, 1.0)
    glVertex3d(-1.0, -1.0,  0.0)
    glTexCoord2d(1.0, 1.0)
    glVertex3d( 1.0, -1.0,  0.0)
    glTexCoord2d(1.0, 0.0)
    glVertex3d( 1.0,  1.0,  0.0)
    glTexCoord2d(0.0, 0.0)
    glVertex3d(-1.0,  1.0,  0.0)
    glEnd()
    glFlush()
    glutSwapBuffers()

def inferencer(results, lock, frameBuffer, handle):
    failure = 0
    sleep(1)
    while failure < 100:

        lock.acquire()
        if len(frameBuffer) == 0:
            lock.release()
            failure += 1
            continue

        img = frameBuffer[-1].copy()
        del frameBuffer[-1]
        failure = 0
        lock.release()

        start = time.time()
        imgw = img.shape[1]
        imgh = img.shape[0]

        now = time.time()
        im,offx,offy = PrepareImage(img, dim)        
        handle.LoadTensor(im.astype(np.float16), 'user object')
        out, userobj = handle.GetResult()
        out = Reshape(out, dim)
        internalresults = detector.Detect(out.astype(np.float32), int(out.shape[0]/wh), blockwd, blockwd, classes, imgw, imgh, threshold, nms, targetBlockwd)
        pyresults = [BBox(x) for x in internalresults]
        results.put(pyresults)
        print("elapsedtime = ", time.time() - now)

def PrepareImage(img, dim):
    imgw = img.shape[1]
    imgh = img.shape[0]
    imgb = np.empty((dim[0], dim[1], 3))
    imgb.fill(0.5)

    newh = dim[1]
    neww = dim[0]
    offx = int((dim[0] - neww)/2)
    offy = int((dim[1] - newh)/2)

    imgb[offy:offy+newh,offx:offx+neww,:] = cv2.resize(img.copy()/255.0,(newh,neww))
    im = imgb[:,:,(2,1,0)]

    return im,offx,offy

def Reshape(out, dim):
    shape = out.shape
    out = np.transpose(out.reshape(wh, int(shape[0]/wh)))
    out = out.reshape(shape)
    return out

class BBox(object):
    def __init__(self, bbox):
        self.left = bbox.left
        self.top = bbox.top
        self.right = bbox.right
        self.bottom = bbox.bottom
        self.confidence = bbox.confidence
        self.objType = bbox.objType
        self.name = bbox.name


glutInitWindowPosition(0, 0)
glutInit(sys.argv)
glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE )
glutCreateWindow("DEMO")
glutFullScreen()
glutDisplayFunc(camThread)
glutReshapeFunc(resizeview)
glutKeyboardFunc(keyboard)
init()
glutIdleFunc(idle) 

print("press 'q' to quit!\n")

threads = []

for devnum in range(len(devices)):
  t = Thread(target=inferencer, args=(results, lock, frameBuffer, graphHandle[devnum]))
  t.start()
  threads.append(t)

glutMainLoop()
