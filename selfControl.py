import time
import numpy
from djitellopy import Tello
import cv2 as cv
import argparse
import math

# parsearg. setup
parser = argparse.ArgumentParser(description='Source of Tello Drone Self Control')
parser.add_argument('--face_cascade', help='Path fo face cascade',
                    default='data/haarcascades/haarcascade_frontalface_alt.xml')
parser.add_argument('--camera', help='When drone not in use, alternative camera. (default: 0)', type=int, default=int(0))
parser.add_argument('--drone', help='Is drone in use? Change this if you want to use your webcam only. (default: True)',
                    default=True)
parser.add_argument('--fly_manual', help='Do you want to fly manual without automated controls? (default: False)',
                    default=False)
parser.add_argument('--distance', help='Drone distance to targeting face or object (0-6), (default: 3)', default=3,
                    type=int)
parser.add_argument('--tx', '--tolerance_x',
                    help='Drone tolerance distance to targeting x-axis on face or object (0 - 200), (default: 100)',
                    default=100, type=int)
parser.add_argument('--ty', '--tolerance_y',
                    help='Drone tolerance distance to targeting y-axis on face or object (0 - 200), (default: 100)',
                    default=100, type=int)
parser.add_argument('--tz', '--tolerance_z',
                    help='Drone tolerance distance to targeting z-axis on face or object (0 - 200), (default: 50)',
                    default=50, type=int)
args = parser.parse_args()
print('Setted Args: ', args)


# Error function
def throw_error(msg: str):
    print('--(!){}'.format(msg))
    close()


def close():
    cv.destroyAllWindows()
    exit(0)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# -- CONSTANTS
FPS = 25
SPEED = 20

# is drone or webcam in use
manual_flight = str2bool(args.fly_manual)
use_drone = str2bool(args.drone)

# boundaries of screen target field
tDistances = [320, 290, 250, 210, 170, 140]
TARGET_SIZE = 210
if 6 > args.distance >= 0:
    TARGET_SIZE = tDistances[args.distance]

# tolerance distances to target point
tx = args.tx
ty = args.ty
tz = args.tz

# interface constants
FONT = cv.FONT_HERSHEY_SIMPLEX
LEFT1 = (10, 500)
LEFT2 = (10, 450)
LEFT3 = (10, 400)
TOPLEFT = (10, 50)
FONTSCALE = 0.5
FONTCOLOR = (255, 255, 255)
LINETYPE = 2

# recognition setup
face_cascade_name = args.face_cascade
face_cascade = cv.CascadeClassifier()

# Load the cascades
if not face_cascade.load(cv.samples.findFile(face_cascade_name)):
    throw_error('Error loading cascade file')


# Frontend Obj
class TelloSelfControl(object):
    def __init__(self):
        self.tello = Tello()
        self.is_flying = False
        self.frame_read = None
        self.cap = None
        self.faces = None
        self.displayMsg = ''
        self.drone_speed = (0, 0, 0, 0)
        self.vDistance = (0, 0, 0, 0)
        self.windowSize = None
        self.windowCenter = None
        self.objectCenter = None
        self.isOverride = False

        self.velocity_left_right = 0
        self.velocity_up_down = 0
        self.velocity_forward_backward = 0
        self.velocity_yaw = 0

        self.nFace = (0, 0, 0, 0)
        self.last_seen = (0, 0, 0, 0)
        self.last_last_seen = (0, 0, 0, 0)
        self.valid_lock = False
        self.valid_lock_count = 0
        self.is_face_lost = False

        self.imgCount = 0

    def run(self):
        # Tello setup
        if use_drone:
            self.frame_read = self.drone_setup()
        else:
            self.cap = self.video_capture_setup()
            if not self.cap.isOpened:
                throw_error('Error opening video capture')

        # init cv2 window
        cv.namedWindow('Tracker')

        # run frames
        while True:
            # reset
            self.isOverride = False
            self.displayMsg = ''
            self.velocity_left_right = 0
            self.velocity_up_down = 0
            self.velocity_forward_backward = 0
            self.velocity_yaw = 0

            # img count for status data form tello
            if self.imgCount == 100:
                if use_drone:
                    self.status_update()
                self.imgCount = 0
            else:
                self.imgCount += 1

            if use_drone:
                if self.frame_read.stopped:
                    self.frame_read.stop()
                    self.tello.streamoff()
                    throw_error('stream turned off')
                    break

                # get single frame
                frame = self.frame_read.frame
            else:
                ret, frame = self.cap.read()

            if frame is None:
                throw_error('No captured frame -- Break')
                break

            # get window properties
            self.windowSize = cv.getWindowImageRect('Tracker')
            self.windowCenter = (self.windowSize[2] // 2, self.windowSize[3] // 2)

            time.sleep(1 / FPS)

            # face detection
            self.faces = self.face_detection(frame)

            # Overridecontrols
            if use_drone:
                self.override_detection()

            # automation controls
            if not self.isOverride and not manual_flight:
                self.auto_control()

            # send rc controls
            if use_drone:
                self.tello.send_rc_control(self.velocity_left_right, self.velocity_forward_backward,
                                           self.velocity_up_down, self.velocity_yaw)

            # show frame in window
            self.display(frame)


            if self.nFace is not self.last_seen:
                self.last_last_seen = self.last_seen

            self.last_seen = self.nFace

            # Exit key esc
            key = cv.waitKey(10)
            if key == 27:
                self.tello.end()
                throw_error('Close application')

    # drone setup
    def drone_setup(self):
        if not self.tello.connect():
            throw_error('Could not connect to Tello')
        if not self.tello.streamon():
            throw_error('Could not load stream')
        print(self.tello.get_battery()[:2])
        return self.tello.get_frame_read()

    def video_capture_setup(self):
        return cv.VideoCapture(args.camera)

    # frame detection
    def face_detection(self, frame):
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        gray_frame = cv.equalizeHist(gray_frame)
        objects = face_cascade.detectMultiScale(gray_frame)
        return objects

    # frame visualsation
    def display(self, frame):
        self.drone_speed = (self.velocity_yaw, self.velocity_up_down, self.velocity_forward_backward)

        if not None:
            for (x, y, w, h) in self.faces:
                frame = cv.circle(frame, (x + w // 2, y + h // 2), 10, (78, 214, 99), 2)  # center face
                frame = cv.rectangle(frame, (x, y), (x + w, y + h), (78, 214, 99), 2)  # face
                frame = cv.putText(frame, 'Width: ' + str(w), LEFT1, FONT, FONTSCALE, FONTCOLOR, LINETYPE)
                frame = cv.putText(frame, 'Height: {0}'.format(str(h)), LEFT2, FONT, FONTSCALE, FONTCOLOR, LINETYPE)

        frame = cv.putText(frame, self.displayMsg, LEFT3, FONT, FONTSCALE, FONTCOLOR, LINETYPE)  # data message
        frame = cv.putText(frame, str(self.vDistance), TOPLEFT, FONT, FONTSCALE, FONTCOLOR,
                           LINETYPE)  # distance vektor to center
        frame = cv.putText(frame, str(self.drone_speed), (10, 100), FONT, FONTSCALE, FONTCOLOR,
                           LINETYPE)  # speed send to drone
        frame = cv.circle(frame, self.windowCenter, 10, (225, 15, 47), 2)  # center window
        frame = cv.rectangle(frame, (self.windowCenter[0] - TARGET_SIZE // 2, self.windowCenter[1] - TARGET_SIZE // 2),
                             (self.windowCenter[0] + TARGET_SIZE // 2, self.windowCenter[1] + TARGET_SIZE // 2),
                             (225, 15, 47), 2)  # distance area
        frame = cv.rectangle(frame, (self.windowCenter[0] - tx // 2, self.windowCenter[1] - ty // 2),
                             (self.windowCenter[0] + tx // 2, self.windowCenter[1] + ty // 2),
                             (255, 191, 0), 2)  # tolerance area
        (x, y, w, h) = self.last_last_seen
        frame = cv.rectangle(frame, (x, y),
                             (x + w, y + h),
                             (255, 0, 238), 2)  # last last seen face position
        (x, y, w, h) = self.last_seen
        frame = cv.rectangle(frame, (x, y),
                             (x + w, y + h),
                             (255, 191, 0), 2)  # last last seen face position


        cv.imshow('Tracker', frame)

    # drone autocontrol
    def auto_control(self):
        if isinstance(self.faces, tuple):
            self.face_lost()
            return

        # we validate which face has the highest chance to be the same face as the last seen face the valid_value
        # describes the diffrence between detected and last seen face. the face with the lowest valid_value will be
        # selected
        valid_value = math.inf
        self.nFace  = (0, 0, 0, 0)
        for face in self.faces:
            vD = self.get_dist_vector(face)
            value = abs(vD[0]) + abs(vD[1]) + abs(vD[2])
            if value < valid_value:
                self.nFace  = face
                valid_value = value

        self.displayMsg += str(valid_value) + ' / '
        (x, y, w, h) = self.nFace

        # if the last face isnt locked, we need to check the counter
        if not self.valid_lock:
            self.valid_lock_fnc(valid_value)

        # if the last face was locked and the detected valid_value has more than 200 difference to the last face,
        # we need to clal face_lost()
        if valid_value > 200 and self.valid_lock:
            self.nFace = self.last_seen
            self.face_lost()
            print('-- Face lost', valid_value)
            return

        # if the face was lost last frame, the lock is open to prevent that something wrong gonna be tracked
        if self.is_face_lost:
            self.valid_lock_count = 0
            self.valid_lock = False
            self.is_face_lost = False

        self.objectCenter = (x + w // 2, y + h // 2)  # face center

        # calculate the distance vector from center of target to center of window(camera view)
        vCenter = numpy.array((self.windowCenter[0], self.windowCenter[1], TARGET_SIZE))
        vTarget = numpy.array((self.objectCenter[0], self.objectCenter[1], self.nFace[3]))
        self.vDistance = vCenter - vTarget

        # set the right speed for the distance
        sX=sY=sZ= SPEED
        if abs(self.vDistance[0]) < tx:
            sX = sX//2
        if abs(self.vDistance[1]) < ty:
            sY = sY // 2
        if self.vDistance[2] > 2 * tz:
            sZ = sZ // 2

        # set the movement of the drone by analysing the data. (tx, ty, tz are the tolerance distances. Within these
        # distances the drone stays still)
        # X - axis correction
        if self.vDistance[0] + tx // 2 < 0:
            self.velocity_yaw = sX
            self.displayMsg += 'turn right /'
        elif self.vDistance[0] - tx // 2 > 0:
            self.velocity_yaw = - sX
            self.displayMsg += 'turn left /'

        # Y - axis correction
        if self.vDistance[1] + ty//2 < 0:
            self.velocity_up_down = - sY
            self.displayMsg += 'move down /'
        elif self.vDistance[1] - ty // 2 > 0:
            self.velocity_up_down = sY
            self.displayMsg += 'move up /'

        # Z - axis correction
        if self.vDistance[2] < 0:
            self.velocity_forward_backward = - sZ
            self.displayMsg += 'move backward /'
        elif self.vDistance[2] - tz > 0:
            self.velocity_forward_backward = sZ
            self.displayMsg += 'move forward /'


    # last seen valuation
    def get_dist_vector(self, face):
        (x, y, w, h) = face
        (lx, ly, lw, lh) = self.last_seen

        v1 = numpy.array((x + w //2, y + h // 2, w)) # vector off detected face
        v2 = numpy.array((lx + lw //2, ly + lh // 2, lw)) # vector off last valid face
        return v1 - v2

    # we need to handle the drone in case the face got lost
    def face_lost(self):
        self.is_face_lost = True
        (x, y, w, h) = self.last_seen
        (lx, ly, lw, lh) = self.last_last_seen
        self.displayMsg = 'x: {}, y: {} / x: {}, y: {}'.format(x, y, lx, ly)
        if x > lx:
            self.velocity_yaw = SPEED
        else:
            self.velocity_yaw = - SPEED

    # if a new face is detected and no other was got locked, we need to valid this face.
    # For example we need to check that the detected face is just caused of a noise image quality.
    # For this we confirm that the face was 10 frames in a row detected. if not so, an other face could get locked
    def valid_lock_fnc(self, valid_value):
        if valid_value < 200:
            self.valid_lock_count += 1
        else:
            self.valid_lock_count = 0

        print('-- valid lock count: ', self.valid_lock_count)

        if self.valid_lock_count >= 10:
            self.valid_lock = True
            print('-- valid lock: TRUE')

    # override by key input
    def override_detection(self):
        key = cv.waitKey(1)
        if key is not -1:
            self.isOverride = True
        if key == ord('t'):
            if self.tello.takeoff():
                self.is_flying = True
        elif key == ord('l'):
            self.tello.land()
            self.is_flying = False
        elif key == ord('q'):
            self.tello.land()
            self.frame_read.stop()
            self.is_flying = False
            close()
        else:
            if key == ord('w'):
                self.velocity_forward_backward = SPEED
            if key == ord('s'):
                self.velocity_forward_backward = - SPEED
            if key == ord('a'):
                self.velocity_left_right = - SPEED
            if key == ord('d'):
                self.velocity_left_right = SPEED
            if key == ord('y'):
                self.velocity_up_down = SPEED
            if key == ord('h'):
                self.velocity_up_down = -SPEED
            if key == ord('g'):
                self.velocity_yaw = - 3 * SPEED
            if key == ord('j'):
                self.velocity_yaw = 3 * SPEED

    def status_update(self):
        print('Tello Battery: {}'.format(self.tello.get_battery()[:2]))
        print('Tello Temperature: {}'.format(self.tello.get_temperature()[:2]))


selfControl = TelloSelfControl()
selfControl.run()
