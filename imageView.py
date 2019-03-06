from PIL import Image
import sys

if( len(sys.argv) < 1 ):
    print("Please include image file name in command line")
else:
    imageName = sys.argv[1]
    Image.open(imageName).show()

