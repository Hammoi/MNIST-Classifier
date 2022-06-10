import tkinter as tk
import numpy as np

from gui import send_data
#copied from some stackoverflow post (thank you)
class DrawableGrid(tk.Frame):
    def __init__(self, parent, width, height, size=5):
        super().__init__(parent, bd=1, relief="sunken")
        self.width = width
        self.height = height
        self.size = size
        canvas_width = width*size
        canvas_height = height*size
        self.canvas = tk.Canvas(self, bd=0, highlightthickness=0, width=canvas_width, height=canvas_height)
        self.canvas.pack(fill="both", expand=True, padx=2, pady=2)

        for row in range(self.height):
            for column in range(self.width):
                x0, y0 = (column * size), (row*size)
                x1, y1 = (x0 + size), (y0 + size)
                self.canvas.create_rectangle(x0, y0, x1, y1,
                                             fill="white", outline="gray",
                                             tags=(self._tag(row, column),"cell" ))
        self.canvas.tag_bind("cell", "<B1-Motion>", self.paint)
        self.canvas.tag_bind("cell", "<1>", self.paint)

    def _tag(self, row, column):
        """Return the tag for a given row and column"""
        tag = f"{row},{column}"
        return tag

    def get_pixels(self):
        row = ""
        numpy_output = np.array([[]])
        for row in range(self.height):
            output = ""
            for column in range(self.width):
                color = self.canvas.itemcget(self._tag(row, column), "fill")
                value = "1" if color == "black" else "0"
                output += value

            numpy_output = np.append(numpy_output, np.fromiter(output, (np.unicode,1)).astype(np.int))
        send_data.send(numpy_output)

    def clear(self):
        self.canvas.delete("all")
        for row in range(self.height):
            for column in range(self.width):
                x0, y0 = (column * self.size), (row*self.size)
                x1, y1 = (x0 + self.size), (y0 + self.size)
                self.canvas.create_rectangle(x0, y0, x1, y1,
                                             fill="white", outline="gray",
                                             tags=(self._tag(row, column),"cell" ))

    def paint(self, event):
        x = event.x
        y = event.y
        CONST = 20
        coordinates = [[x,y],[x+CONST,y],[x+2*CONST,y],
                       [x,y+CONST],[x+CONST,y+CONST],[x+2*CONST,y+CONST],
                       [x,y+2*CONST],[x+CONST,y+2*CONST],[x+2*CONST,y+2*CONST]]
        for i in range(0,9):
            cell = self.canvas.find_closest(coordinates[i][0], coordinates[i][1])
            self.canvas.itemconfigure(cell, fill="black")


def start_gui():
    root = tk.Tk()

    canvas = DrawableGrid(root, width=28, height=28, size=20)
    b = tk.Button(root, text="Send to Classifier", command=canvas.get_pixels)
    b.pack(side="top")
    canvas.pack(fill="both", expand=True)

    clear = tk.Button(root, text="Clear Canvas", command=canvas.clear)
    clear.pack(side="bottom")
    root.mainloop()
