import cv2
import os
from tkinter import *
from tkinter import filedialog
import subprocess
import webbrowser
import sys
from PIL import Image
from matplotlib import pyplot as plt
import pano_conv
# import time

FILEBROWSER_PATH = 'nautilus'
imput_img_folder = ""
done = 10101

window = Tk()
window.geometry('920x720')
window.title("Pano Now")
window.config(bg="black")

Label(window, text="Pano Now", font=("Arial Bold", 55), bg="black", fg="#DDDDDD").place(relx=0.5, rely=0.21, anchor="center")

def close():
    sys.exit()

def pro():
    webbrowser.open('https://github.com/MJLNSN/Pano_convertor')

def imgfileselect():
    global imput_img_folder
    open_file = filedialog.askdirectory()
    imput_img_folder = open_file

icon2 = PhotoImage(file='imp/folder.png')
icon22 = icon2.subsample(9, 12)
bt22 = Button(window, text=" Select a Folder  ", image=icon22, font=("Arial Bold", 20), compound=LEFT, command=imgfileselect)
bt22.place(relx=0.5, rely=0.50, anchor="center")

icon_close = PhotoImage(file='imp/close.png')
icon_close2 = icon_close.subsample(9, 12)
bt_close = Button(window, image=icon_close2, relief=FLAT, command=close, bg='black')
bt_close.place(relx=0.9, rely=0.01)

icon_help = PhotoImage(file='imp/help.png')
icon_help2 = icon_help.subsample(9, 12)
bt_help = Button(window, image=icon_help2, relief=FLAT, command=pro, bg='black')
bt_help.place(relx=0.01, rely=0.01)

labell = Label(window, text="", bg="black")

def success(done):
    global imput_img_folder
    if done == 1:
        labell.configure(text="Successful! Please check the folder output!!!", fg="green", bg="#FCFFE7", borderwidth=2, relief="raised")
        labell.place(relx=0.5, rely=0.9, anchor="center")
        window.update()
        labell.after(7000, lambda: labell.place_forget())
    elif done == 0:
        labell.configure(text="Process cannot be completed", fg="red", bg="#FCFFE7", borderwidth=2, relief="raised")
        labell.place(relx=0.5, rely=0.9, anchor="center")

def output_open():
    outop = "output"
    subprocess.run([FILEBROWSER_PATH, outop])

def open_cv():
    global imput_img_folder
    global done
    global Images
    folder = imput_img_folder
    if imput_img_folder == "":
        labell.configure(text="Error: Plese select a folder which comtains several input images", fg="red", bg="#FCFFE7", borderwidth=2, relief="raised")
        return
    else:
        print("input images' location:")
        print(imput_img_folder)
        labell.configure()

        path = imput_img_folder
        Images = []
        mylist = os.listdir(path)
        print("input images list:")
        print(mylist)
        for imgn in mylist:
            curimg = cv2.imread(path + '/' + imgn)
            curimg = cv2.resize(curimg, (0, 0), None, 0.5, 0.5)
            Images.append(curimg)
            status, result = pano_conv.converter(Images)

        if status == 1:
            cv2.imwrite('output/pano_out.png', result)
            print("done")
            done = 1
            plt.imshow(result)
            print("The panorama is saved in folder output!!!")
            cv2.waitKey(1)
        else:
            print("Could not perform")
            done = 0
            lb_done = Label(window, text="Could not perform", font=("Arial Bold", 15), bg="black", fg="red")
            lb_done.place(relx=0.5, rely=0.81, anchor="center")
    success(done)
    cv2.waitKey(0)

labell.configure()
labell.pack(side=LEFT, ipadx=5, ipady=5)

icon_start = PhotoImage(file='imp/pika.png')
icon_start2 = icon_start.subsample(9, 12)
bt21 = Button(window, text=" Convert Now ", compound=LEFT, command=open_cv, fg="green", font=("Arial Bold", 25, "bold"))
bt21.place(relx=0.5, rely=0.66, anchor="center")

icon_out = PhotoImage(file='imp/output.png')
icon_out2 = icon_out.subsample(9, 12)
bt211 = Button(window, image=icon_out2, compound=LEFT, text="Open Output Folder", font=("Arial Bold", 20), command=output_open)
bt211.place(relx=0.5, rely=0.82, anchor="center")

window.mainloop()


