from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from PIL import ImageTk, Image
from ocr_tesseract import get_words_cords
import cv2
import fileApi 


COLOR_WHITE = (247, 247, 247)
COLOR_BLACK = (59, 89, 152)
COLOR_BLUE = (0, 0, 255)
COLOR_RED = (255, 0, 0)

SIZE_WIDTH = 650
SIZE_HEIGHT = 400
RESULTS_WIDTH = 768
RESULTS_HEIGHT = 1024


class ResultsWindow:
    counter = 0
    def __init__(self, master, images, img_data, mode):
        self.api = fileApi.fileAPI()
        self.img_word_cord = {}
        self.images = images
        self.master = master

        self.images_marked = [] # cv2 img
        self.images_orginal = img_data # cv2 img

        self.words_cords = get_words_cords(self.images_orginal)

        self.frame = Frame(self.master)
        next_button = Button(self.frame, text='Next image', command=self.next_image)
        previous_button = Button(self.frame, text='Previous image', command=self.previous_image)
        self.entry_word = Entry(self.frame)
        find_button = Button(self.frame, text='Find', command=self.find)
        remove_button = Button(self.frame, text='Remove image', command=self.remove)
        add_button = Button(self.frame, text='Add image(s)', command=self.add)

        img = Image.open(self.images[self.counter])
        img.thumbnail((RESULTS_WIDTH - 10, RESULTS_HEIGHT - 120), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        self.panel = Label(self.frame, image=img)
        self.panel.grid(row=0, columnspan=3, sticky=SW)

        next_button.grid(row=1, column=2)
        previous_button.grid(row=1, column=0)
        remove_button.grid(row=2, column=2)
        add_button.grid(row=2, column=0)
        self.entry_word.grid(row=3, column=0)
        find_button.grid(row=3, column=2)

        self.frame.pack()
        self.frame.mainloop()

    def set_image(self):
        img = Image.open(self.images[self.counter])
        img.thumbnail((RESULTS_WIDTH - 10, RESULTS_HEIGHT - 120), Image.ANTIALIAS)
        img = ImageTk.PhotoImage(img)
        self.panel.configure(image=img)
        self.panel.image = img

    def next_image(self):
        self.counter += 1
        if self.counter > len(self.images) - 1:
            self.counter = 0
        self.set_image()

    def previous_image(self):
        self.counter -= 1
        if self.counter < 0:
            self.counter = len(self.images) - 1
        self.set_image()

    def mark_word(self, word):
        tmp = []
        for key in self.words_cords:
            print(key)
            if word in key:
                tmp.append(self.words_cords[key])

        for t in tmp:
            for path, cords in t.items():
                for img in self.images_orginal:
                    if img[0] == path:

                        w = cv2.rectangle(img[1], cords[0], cords[1], (255, 255, 0), 3)
                        cv2.imshow('sa', w)
                        cv2.waitKey(0)
                        self.images_marked.append(w)

    def find(self):
        w = self.entry_word.get()
        if w != '':
            print(w)
            self.mark_word(w)
        #szukanie slowa na obrazie
    
    def remove(self):
        self.images.remove(self.images[self.counter])
        self.counter=0
        messagebox.showinfo('Info', 'Deleted successfully!')

    def add(self):
        files = list(filedialog.askopenfilenames(parent=self.master, title='Choose image file(s)', filetypes=(("All files", "*.*"), ("PNG", "*.png"), ("JPG", "*.jpg"), ("Bitmap", "*.bmp"))))
        if len(files) != 0:
            messagebox.showinfo('Info', 'Added image(s)!')
            #fileNames = self.api.readFiles(files)
            #imgProc.showResizedImage('Result',imgProc.straightenImage(self.api.getImage(fileNames[0])),2)
            for file in files:
                self.images.append(file)
        print(f'files: {self.images}')

class UI:
    def test(self):
        self.newWindow.grab_release()
        self.newWindow.destroy()

    root = Tk()
    var = IntVar()
    def __init__(self):
        self.api = fileApi.fileAPI()

        self.root.maxsize(SIZE_WIDTH, SIZE_HEIGHT)
        self.root.minsize(SIZE_WIDTH, SIZE_HEIGHT)
        self.root.resizable(0, 0)
        self.root.winfo_toplevel().title('OCR App')
        FONT = 'Microsoft Sans Serif'
        font_header = (FONT, 45)
        font_footer = (FONT, 10)
        header = Label(self.root, text='OCR App', font=font_header)
        footer = Label(self.root, text='by Kontowicz Piotr, Przybylowski Pawel, Szkudlarek Damian', font=font_footer)
        header.pack()
        footer.place(x=10, y=375)

        
        r1 = Radiobutton(self.root, text='Faktura', variable = self.var, value = 1)
        r2 = Radiobutton(self.root, text='Tekst naukowy', variable = self.var, value = 2)

        r1.pack()
        r2.pack()
        load_images_button = Button(self.root, text='Select image(s)', command=self.load_images, width=50, height=5)
        load_images_button.place(x=145, y=150)

        self.root.mainloop()

    def show_results_window(self, master, images, data, var):
        self.newWindow = Toplevel(master)
        self.newWindow.winfo_toplevel().title('OCR App - Preview Images')
        self.newWindow.maxsize(RESULTS_WIDTH, RESULTS_HEIGHT)
        self.newWindow.minsize(RESULTS_WIDTH, RESULTS_HEIGHT)
        self.newWindow.resizable(0, 0)
        self.newWindow.grab_set()
        self.newWindow.protocol("WM_DELETE_WINDOW", self.test)
        self.app = ResultsWindow(self.newWindow, images, data, var)

    def load_images(self):
        files = list(filedialog.askopenfilenames(parent=self.root, title='Choose image file(s)', filetypes=(("All files", "*.*"), ("PNG", "*.png"), ("JPG", "*.jpg"), ("Bitmap", "*.bmp"))))
        if len(files) != 0:
            data = []
            for path in files:
                data.append([path, cv2.imread(path)])
            messagebox.showinfo('Info', 'Loaded images!')
            self.show_results_window(self.root, files, data, self.var)

if __name__ == "__main__":
    UI()
