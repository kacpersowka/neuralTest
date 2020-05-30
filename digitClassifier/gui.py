#Adapted from https://gist.github.com/nikhilkumarsingh/85501ee2c3d8c0cfa9d1a27be5781f06

from tkinter import *
import io
from PIL import Image
from tkinter.colorchooser import askcolor
import neural,pickle

if len(sys.argv)>1:
    net=str(sys.argv[1])
else:
    net='trained.pkl'

with open(net, "br") as fh:
    w,b = pickle.load(fh)

def process(img):
    p=[]
    fac = 0.99 / 255
    for i in img:
        p.append(((255-(i[0]+i[1]+i[2])/3))*fac+0.01)
    return p

def guess(img):
    a=neural.feedForward(img,w,b,[0 for i in w[-1]],neural.function1,neural.crossEntropy,[[neural.expRectLinear],[neural.softmax]])
    r=[]
    for i in a[-2]:
        r.append(i*100)
    return r

class Paint(object):

    DEFAULT_PEN_SIZE = 5.0
    DEFAULT_COLOR = 'black'

    def __init__(self):
        self.root = Tk()

        self.pen_button = Button(self.root, text='pen', command=self.use_pen)
        self.pen_button.grid(row=0, column=0)

        self.brush_button = Button(self.root, text='brush', command=self.use_brush)
        self.brush_button.grid(row=0, column=1)

        self.color_button = Button(self.root, text='color', command=self.choose_color)
        self.color_button.grid(row=0, column=2)

        self.eraser_button = Button(self.root, text='eraser', command=self.use_eraser)
        self.eraser_button.grid(row=0, column=3)

        self.choose_size_button = Scale(self.root, from_=1, to=10, orient=HORIZONTAL)
        self.choose_size_button.grid(row=0, column=4)
		
        self.convert_button = Button(self.root, text='convert', command=self.convert)
        self.convert_button.grid(row=0, column=5)
        
        self.clear_button = Button(self.root, text='clear', command=lambda self:self.c.delete('all'))
        self.clear_button.grid(row=0, column=6)

        self.c = Canvas(self.root, bg='white', width=28*2, height=28*2)
        self.c.grid(row=1, columnspan=5)

        self.setup()
        self.root.mainloop()

    def setup(self):
        self.old_x = None
        self.old_y = None
        self.line_width = self.choose_size_button.get()
        self.color = self.DEFAULT_COLOR
        self.eraser_on = False
        self.active_button = self.pen_button
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)

    def convert(self):
        ps=self.c.postscript(colormode='gray')
        im = Image.open(io.BytesIO(ps.encode('utf-8')))
        im=im.resize((28,28))
        px=list(im.getdata())
        g=guess(process(px))
        for i in range(len(g)):
            print(i,': ',g[i],'%')
        print('Top guess: ',g.index(max(g)))
		
    def use_pen(self):
        self.activate_button(self.pen_button)

    def use_brush(self):
        self.activate_button(self.brush_button)

    def choose_color(self):
        self.eraser_on = False
        self.color = askcolor(color=self.color)[1]

    def use_eraser(self):
        self.activate_button(self.eraser_button, eraser_mode=True)

    def activate_button(self, some_button, eraser_mode=False):
        self.active_button.config(relief=RAISED)
        some_button.config(relief=SUNKEN)
        self.active_button = some_button
        self.eraser_on = eraser_mode

    def paint(self, event):
        self.line_width = self.choose_size_button.get()
        paint_color = 'white' if self.eraser_on else self.color
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x, self.old_y, event.x, event.y,
                               width=self.line_width, fill=paint_color,
                               capstyle=ROUND, smooth=TRUE, splinesteps=36)
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x, self.old_y = None, None


if __name__ == '__main__':
    Paint()