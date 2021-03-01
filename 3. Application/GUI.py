import PySimpleGUI as sg 
import numpy as np
from PIL import ImageGrab
import utils

CANVAS_WIDTH = 545
CANVAS_HEIGHT = 450

# Define the window layout
layout = [  [sg.Text('PySimpleGUI-H.Calc.', size=(50,1), justification='left', background_color="#272533", 
            text_color='white', font=('Franklin Gothic Book', 14, 'bold'))],
            [sg.Text('0.0000', size=(18,1), justification='right', background_color='black', text_color='red', 
            font=('Digital-7', 48), relief='sunken', key="_DISPLAY_")],
            [sg.Graph(canvas_size=(CANVAS_WIDTH, CANVAS_HEIGHT), graph_bottom_left=(0,0), graph_top_right=(CANVAS_WIDTH,CANVAS_HEIGHT), 
            background_color='white',  drag_submits='TRUE', enable_events='TRUE', key='_CANVAS_')],
            [sg.Button('Clr', size=(7,2), font=('Franklin Gothic Book', 24), button_color=("black","#ECA527"), pad=(215,10))]]

# Create the window and show it without the plot
window = sg.Window('PySimpleGUI-H.Calc.', layout, background_color="#272533").Finalize()

#CALCULATOR FUNCTIONS
var: dict = {'front':[], 'back':[], 'x_val':0.0, 'y_val':0.0, 'result':0.0, 'operator':''}

#HELPER FUNCTIONS
def update_display(display_value: str):
    """Change display window to value to 4 decimal places."""
    try:
        window['_DISPLAY_'].update(value='{:,.4f}'.format(display_value))
    except:
        window['_DISPLAY_'].update(value=display_value)

def get_canvas():
    """Obtain the image from the canvas's position."""
    widget = window['_CANVAS_'].Widget
    box = (widget.winfo_rootx(), widget.winfo_rooty(), widget.winfo_rootx() + widget.winfo_width(), widget.winfo_rooty() + widget.winfo_height())
    image = ImageGrab.grab(bbox=box).convert('L')
    return image

def translate_canvas(image):
    """Return string of symbols from multiple handwritten symbols
    on the input image."""
    nums = []
    imgs_ok = []
    cxs = []
    imgs = utils.thresholding(image, CANVAS_WIDTH, CANVAS_HEIGHT)
    imgs, d = utils.detect(imgs)
    labels = set(d.values())
    labels.remove(0)
    for label in sorted(labels):
        img, cx = utils.crop(imgs, label)
        img = utils.image_resize(img)
        img = utils.image_centering(img)
        img = utils.image_pad(img)
        img = img.astype(np.float32)
        num = utils.predict(img)
        nums.append(num)
        imgs_ok.append(img)
        cxs.append(cx)
    nums = utils.sort_by_other_list(nums, cxs)
    imgs_ok = utils.sort_by_other_list(imgs_ok, cxs)
    calculation = ''.join(nums)
    display = utils.calculate(calculation)
    return display

#CLICK EVENTS
def clear_click():
    global var
    var['front'].clear()
    var['back'].clear()

#EVENT LOOP
while True:
    event, values = window.read()
    print(event)
    if event is None:
        break # exit

    if event == 'Clr':
        clear_click()
        update_display(0.0000)
        var['result'] = 0.0000
        window['_CANVAS_'].draw_rectangle(top_left=(0,CANVAS_HEIGHT), bottom_right=(CANVAS_WIDTH,0), fill_color='white')

    if event == '_CANVAS_':   
        x, y = values["_CANVAS_"]
        window['_CANVAS_'].draw_point((x,y), size=8)

    if event.endswith('+UP'):
        x, y = values["_CANVAS_"]
        image = get_canvas()
        display = translate_canvas(image)
        update_display(display)