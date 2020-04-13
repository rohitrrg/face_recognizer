from capture_face import capture
from detector import detect


class Face_Detector:

        print("Hello,\nWelcome to Face detection Machine\n")
        print("Choose any one :")
        print("1. Detect Faces \n2. Register Faces\n")
        start = int(input())

        if start == 2:
            name = input("Enter your first name\n")
            capture(name)

        if start == 1:
            detect()
