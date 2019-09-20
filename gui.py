from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
from PIL import ImageTk, Image
#from ocr_tesseract import get_words_cords
from new_prediction_flow import get_words_cords
import cv2
import copy
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
        self.master = master

        self.images_marked = img_data # cv2 img
        self.images_orginal = img_data # cv2 img

        self.words_cords = get_words_cords(self.images_orginal)

        self.frame = Frame(self.master)
        next_button = Button(self.frame, text='Next image', command=self.next_image)
        previous_button = Button(self.frame, text='Previous image', command=self.previous_image)
        self.entry_word = Entry(self.frame)
        find_button = Button(self.frame, text='Find', command=self.find)
        remove_button = Button(self.frame, text='Remove image', command=self.remove)
        add_button = Button(self.frame, text='Add image(s)', command=self.add)

        img = Image.fromarray( self.images_marked[self.counter][1])
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
        if len(self.images_marked) != 0:
            img = Image.fromarray( self.images_marked[self.counter][1])
            img.thumbnail((RESULTS_WIDTH - 10, RESULTS_HEIGHT - 120), Image.ANTIALIAS)
            img = ImageTk.PhotoImage(img)
            self.panel.configure(image=img)
            self.panel.image = img

    def next_image(self):
        print(self.counter)
        self.counter += 1
        if self.counter > len(self.images_marked) - 1:
            self.counter = 0
        self.set_image()

    def previous_image(self):
        print(self.counter)
        self.counter -= 1
        if self.counter < 0:
            self.counter = len(self.images_marked) - 1
        self.set_image()

    def mark_word(self, word):
        self.images_marked = []
        tmp = []
        for key in self.words_cords:
            print(key)
            if word in key:
                tmp.append(self.words_cords[key])
        test = copy.deepcopy(self.images_orginal)
        for t in tmp:
            for path, cords in t.items():
                for img in test:
                    if img[0] == path:
                        w = cv2.rectangle(img[1], cords[0], cords[1], (255, 25, 25), 6)
                        img[1] = w
        self.images_marked = test

    def find(self):
        w = self.entry_word.get()
        if w != '':
            print(w)
            self.mark_word(w)
            self.set_image()

    def remove(self):
        self.images_orginal.remove(self.images_orginal[self.counter])
        self.images_marked.remove(self.images_marked[self.counter])
        self.counter=0
        messagebox.showinfo('Info', 'Deleted successfully!')

    def add(self):
        files = list(filedialog.askopenfilenames(parent=self.master, title='Choose image file(s)', filetypes=(("All files", "*.*"), ("PNG", "*.png"), ("JPG", "*.jpg"), ("Bitmap", "*.bmp"))))
        if len(files) != 0:

            for file in files:
                self.images_orginal.append([file, cv2.imread(file)])
        self.words_cords = get_words_cords(self.images_orginal)
        self.counter = 0
        self.set_image()
        messagebox.showinfo('Info', 'Added image(s)!')

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
