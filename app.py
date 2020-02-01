from __future__ import annotations

import streamlit as st
import os, random
import torch, torchvision
from torch import *

from fastai import *
from fastai.vision import *
from torchvision import transforms
from torch.autograd import Variable
import torchvision.transforms.functional as F

from rangerlars import *
from PIL import Image
import pandas as pd


def file_selector(folder_path='.'):
    st.subheader("Upload a file below")
    filenames = os.listdir(folder_path)
    selected_filename = st.selectbox('Select a file', filenames)
    return os.path.join(folder_path, selected_filename)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = torchvision.models.densenet201()
# model.load_state_dict(torch.load('densenet201_97_ranger_30.pth', map_location=torch.device('cpu')))
# print('print 1 loaded')
# model = torch.load('densenet201_97_ranger_30')
# model.eval()
import os
path = os.getcwd()
learn = load_learner(path, '../densenet201_97_ranger_30.pkl')
# learn = learn

test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                     ])


def weed_checker(img):
    out = learn.predict(img)
    return str(out[0])


def basicApp():
    st.write("""
        # Weed Classifier
        #
        """)
    
    filename = st.file_uploader('Upload a file', type='jpg')

    try:
        img = open_image(filename)
        py_img = F.to_pil_image(img.data)
        st.image(py_img)
        weed_type = weed_checker(open_image(filename))

        st.write(weed_type)

        if weed_type == 'Negative':
            st.success("This is a plant.")
        else:
            st.error("This is a " + weed_type + " weed!")
    except IOError:
        st.warning("Please select an image file")
    except AttributeError:
        st.write()


def matrix():
    st.write("""
        # Weed Mapper
        #
        """)

    rows = st.slider('Number of rows', 5, 20)
    cols = st.slider('Number of columns', 5, 20)
    st.write("""# """)

    new_im = Image.new('RGB', (cols*256, rows*256))
    colored_im = Image.new('RGB', (cols*256, rows*256))

    red_im = Image.open('redsquare.png')
    red_im = Image.open('greensquare.png')

    y_offset = 0
    for i in range(0, rows):
        x_offset = 0
        colored_row_im = Image.new('RGB', (cols*256, 256))
        row_im = Image.new('RGB', (cols*256, 256))
        
        for j in range(0, cols):
            file = random.choice(os.listdir("../deepweed/images"))
            im = Image.open('../deepweed/images/' + file)
            
            row_im.paste(im, (x_offset,0))
            colored_row_im.paste(im, (x_offset,0))

            # if weed_checker(open_image("../deepweed/images/" + file) == 'Negative'):
            #     colored_row_im.paste(green_im, (x_offset,0))
            # else:
            #     colored_row_im.paste(red_im, (x_offset,0))

            x_offset += 256

        new_im.paste(row_im, (0, y_offset))
        colored_im.paste(colored_row_im, (0, y_offset))
        y_offset += 256
    
    new_im.save('aerial.jpg')
    colored_im.save('aerial-colored.jpg')
    st.image('aerial.jpg', use_column_width=True)
    st.image('aerial-colored.jpg', use_column_width=True)
    

def poultry_viz():
    st.write("""
        # Poultry Visualizer
        #
        """)

    filenames = os.listdir('../poultry1/')
    selected_filename = st.selectbox('Select a file', filenames)
    path = os.path.join('../poultry1/', selected_filename)
    
    df = pd.read_csv(path)
    # df = df.set_index('timestamp')
    st.line_chart(df)
    st.dataframe(df)
    

    
o1 = "Weed Classifier"
o2 = "Weed Mapper"
o3 = "Poultry Visualizer"
currApp = st.sidebar.selectbox("What app would you like to run", (o1, o2, o3))

if (currApp == o1):
    basicApp()
elif (currApp == o2):
    matrix()
elif (currApp == o3):
    poultry_viz()