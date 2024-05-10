import os
import subprocess
import tkinter as tk

import pytube

#                               UI
# root = tk.TK()

# root.geometry("1000x600")

# root.mainloop(TP1)

#                            End of UI

url = str(input("Paste URL:"))
yt  = pytube.YouTube(url)

vids = yt.streams.filter(file_extension='webm', type = 'audio')

for i in range(len(vids)):
    print (i,'.',vids[i])
    
vnum = int(input("Enter Video Number: "))

parent_dir = r"D:\Rand_py\YTTMP3b"
new_filename = yt.title.replace(":", "")+'.mp3'
vids[vnum].download(parent_dir,new_filename)

print(new_filename)