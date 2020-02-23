# Tello Self Control

Tello Self Control is a simple python application for controlling a DJI Tello that uses OpenCV to detect faces and have them tracked by the drone.    The program was tested with a DJI Tello and should also be compatible with a DJI Tello EDU.

Tested with Python 3.7, but should work with all Python3 versions.

## Install
You will need the contribution version of OpenCV. 

`pip install opencv-contrib-python==4.1.0.25`

Do not use the OpenCV version `4.2.0.32`! This version has issues with the QT library.

## Usage

To use the script with drone just run `python selfControl.py`. If you want to run the script for development without drone run `python selfControl.py --drone false`.

More settings:
```
   -h, --help            show this help message and exit
  --face_cascade FACE_CASCADE
                        Path fo face cascade
  --camera CAMERA       When drone not in use, alternative camera. (default:
                        0)
  --drone DRONE         Is drone in use? Change this if you want to use your
                        webcam only. (default: True)
  --fly_manual FLY_MANUAL
                        Do you want to fly manual without automated controls?
                        (default: False)
  --distance DISTANCE   Drone distance to targeting face or object (0-6),
                        (default: 3)
  --tx TX, --tolerance_x TX
                        Drone tolerance distance to targeting x-axis on face
                        or object (0 - 200), (default: 100)
  --ty TY, --tolerance_y TY
                        Drone tolerance distance to targeting y-axis on face
                        or object (0 - 200), (default: 100)
  --tz TZ, --tolerance_z TZ
                        Drone tolerance distance to targeting z-axis on face
                        or object (0 - 200), (default: 50)

```

## Controls

- `Esc`: Quit Application (at any time)
- `T`: To Takeoff
- `L`: To Land
- `Q`: To Land and then quit

**Override Control**
- `W/S`: Fly Forward/Back
- `A/D`: Fly Left/Right
- `Y/H`: Fly Up/Down
- `G/J`: Rotate Left/Right 

## Notes

* You need as much light as possible to ensure satisfactory facial recognition.
* Furthermore a stable wifi connection between computer and drone is important to be able to transfer and evaluate enough fps.

## Further improvments

- [x] I want to add position prediction for the X - axis to prevent that the drone loses the subject.
- [ ] I want to add position prediction for the Y - axis and Z - axis to prevent that the drone loses the subject.
- [ ] I want to add facial motion detection to send commands using motion
- [ ] Send command by color recognition via webcam

## Credits

This script uses [DJITelloPy](https://github.com/damiafuentes/DJITelloPy) from **Damià Fuentes Escoté** to easily send and get data from the drone.

