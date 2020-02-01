from __future__ import annotations

import streamlit as st
import os, random
from PIL import Image
import torch, torchvision
from torch import *

from fastai import *
from fastai.vision import *
from torchvision import transforms
from torch.autograd import Variable

from rangerlars import *

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
    # image_tensor = test_transforms(image).float()
    # image_tensor = image_tensor.unsqueeze_(0)
    # input = Variable(image_tensor)
    img = open_image(image)
    
    print('print 3 loaded')
    # input = input.to(device)
    # output = learn.predict(input)
    output = learn.predict(img)
    index = output #.numpy().argmax()
    # print('print 4 loaded')
    return index


def basicApp():
    st.write("""
        # Weed Detector
        #
        """)
    
    filename = st.file_uploader('upload a file', type='jpg')
    st.write('You selected `%s`' % filename)

    try:
        weed_type = weed_checker(open_image(filename))
        # st.image(open('im1.jpg'))
    except IOError:
        st.write("Please select a different file")

    # invoke some Ajay function on the image

    st.write(weed_type)

    # isWeed = bool(random.getrandbits(1))
    # if isWeed:
    #     st.error("Smoke that weed!")
    # else:
    #     st.success("It's a plant brutha!")


def matrix():
    st.write("""
        # Aerial Weed Colorer
        #
        """)

    rows = st.slider('Number of rows', 5, 20)
    cols = st.slider('Number of columns', 5, 20)
    st.write("""# """)

    new_im = Image.new('RGB', (cols*256, rows*256))
    colored_im = Image.new('RGB', (cols*256, rows*256))
    y_offset = 0
    for i in range(0, rows):
        x_offset = 0
        colored_row_im = Image.new('RGB', (cols*256, 256))
        row_im = Image.new('RGB', (cols*256, 256))
        
        for j in range(0, cols):
            file = random.choice(os.listdir("../deepweed/images"))
            im = open('../deepweed/images/' + file)
            row_im.paste(im, (x_offset,0))
            colored_row_im.paste(im, (x_offset,0))
            x_offset += 256

        new_im.paste(row_im, (0, y_offset))
        y_offset += 256
    
    new_im.save('aerial.jpg')
    st.image('aerial.jpg', use_column_width=True)
    

def scroller():
    st.write("""
        # Weed Live Scroller
        #
        """)
    rows = st.slider('Number of rows', 5, 20)

    
o1 = "Weed/ Plant Classifier"
o2 = "Aerial View Weed Colorer"
o3 = "Weed Live Scroller"
currApp = st.sidebar.selectbox("What app would you like to run", (o1, o2, o3))

if (currApp == o1):
    basicApp()
elif (currApp == o2):
    matrix()
elif (currApp == o3):
    scroller()