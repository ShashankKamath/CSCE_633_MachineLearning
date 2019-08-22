from tkinter.filedialog import askopenfilename
#from PIL import ImageTk, Image
from vgg_tkinter import vggnet
import numpy as np
import tkinter
import time
import cv2

class HEC:

    def __init__(self, window):

        self.hi()
        self.options()

    # Initiate window
    def hi(self):
        tkinter.Label(
            window,
            text="Welcome to HEC the Human Emotion Classifier!"
        ).pack()

    # Basic Menu
    def options(self):
        tkinter.Label(
            window,
            text="What would you like to do now?"
        ).pack()

        # Taking a photo
        self.takephoto = tkinter.Button(
            window,
            text="Take My Photo",
            command=self.camera
        ).pack()

        # Uploading a photo
        self.browse = tkinter.Button(
            window,
            text="Browse",
            command=self.choose_im
        ).pack()

        self.closebtn = tkinter.Button(
            window,
            text="Exit",
            command=window.quit
        ).pack()

    # Camera function
    def camera(self):
        cap = cv2.VideoCapture(0)

        time.sleep(0.1)
        ret, gray = cap.read()

        h, w, _ = gray.shape
        newh = int(0.8*h)
        gray = gray[int(0.1*h):int(0.9*h), (w-newh)//2:w-(w-newh)//2]

        cv2.imshow('This is you right now.', gray)

        im_name = "images/img_"+str(time.time())+".jpg"
        cv2.imwrite(im_name, gray)

        cap.release()
        # cv2.destroyAllWindows()

        self.clear_window()

        tkinter.Label(
            window,
            text="Retake or continue?"
        ).pack()

        self.closebtn = tkinter.Button(
            window,
            text="Continue",
            command=lambda: self.send_im(im_name)
        ).pack()

        self.closebtn = tkinter.Button(
            window,
            text="Retake",
            command=self.camera
        ).pack()

    def choose_im(self):
        self.clear_window()

        tkinter.Label(
            window,
            text="Choose your file:"
        ).pack()

        filename = askopenfilename()
        self.send_im(filename)

    def send_im(self, im_name):
        tkinter.Label(
            window,
            text="Your emotion is being classified. Please be patient..."
        ).pack()

        # reduce image to 48x48x1
        # im = cv2.imread(im_name)
        gray = cv2.imread(im_name,0)
        # gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        # h, w = gray.shape
        new_dim = (48, 48)
        # send to network
        im = cv2.resize(gray, new_dim)
        im = np.expand_dims(im, -1)
        op_class = vgop(im)

        tkinter.Label(
            window,
            text="Your emotion has been classified. Click on Results."
        ).pack()

        self.closebtn = tkinter.Button(
            window,
            text="Results",
            command=lambda: self.messages(op_class)
        ).pack()


    def messages(self, result):
        self.clear_window()
        texts = [
            "You are Angry. Calm yourself by listening to calm music.",
            "You are Disgusted. Think of things which make you feel comfortable.",
            "You are Afraid. Think of your safe place.",
            "You are Happy. Great!",
            "You are Sad. Don't be.",
            "You are Surprised. What did you expect?.",
            "You are Neutral. Ok.",
        ]

        tkinter.Label(
            window,
            text=texts[result]
        ).pack()

        self.closebtn = tkinter.Button(
            window,
            text="Close",
            command=lambda: self.Close()
        ).pack()

    def all_children(self):
        alist = window.winfo_children()
        for item in alist :
            if item.winfo_children():
                alist.extend(item.winfo_children())
        return alist

    def clear_window(self):
        widget_list = self.all_children()
        for item in widget_list:
            item.pack_forget()
        self.hi()

    def Close(self):
        self.clear_window()
        window.destroy()

def get_image():
    global window
    window = tkinter.Tk()
    window.title("Human Emotion Classifier")
    window.geometry("500x300")
    HEC(window)
    window.mainloop()

    print("The program has ended without errors.")

def vgop(image):
    return vggnet(image)

if __name__ == '__main__':
    get_image()
