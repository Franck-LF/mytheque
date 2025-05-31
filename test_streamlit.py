# Sources
# 
# https://docs.streamlit.io/develop/concepts/design/buttons
# 
# 


from os import listdir
from os.path import isdir, isfile

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
# import pymupdf

# from sentence_transformers import SentenceTransformer, util
# from transformers import AutoModel, AutoTokenizer

# import faiss

import streamlit as st




# -------------------------------------------------

def test1():
    st.title('My First Streamlit App')
    st.write('Welcome to my Streamlit app!')
    user_input = st.text_input('Enter a custom message:', 'Hello, Streamlit!')
    st.write('Customized Message:', user_input)

# -------------------------------------------------


def test2():
    animal_shelter = ['cat', 'dog', 'rabbit', 'bird']
    animal = st.text_input('Type an animal')
    if st.button('Check availability'):
        have_it = animal.lower() in animal_shelter
        'We have that animal!' if have_it else 'We don\'t have that animal.'

# -------------------------------------------------

def click_button1():
    st.session_state.clicked = True

def test3():
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    st.button('Click me', on_click=click_button1)
    if st.session_state.clicked:
        # The message and nested widget will remain on the page
        st.write('Button clicked!')
        st.slider('Select a value')
    
    st.write(st.session_state)

# -------------------------------------------------

def test4():
    # Magic
    df = pd.DataFrame({'col1': [1,2,3]})
    df  # 👈 Draw the dataframe
    x = 10
    'x', x  # 👈 Draw the string 'x' and then the value of x
    arr = np.random.normal(1, 1, size=100)
    fig, ax = plt.subplots()
    ax.hist(arr, bins=20)
    fig  # 👈 Draw a Matplotlib chart

# -------------------------------------------------

def click_button2():
    st.session_state.button = not st.session_state.button

def test5():
    if 'button' not in st.session_state:
        st.session_state.button = False

    st.button('Click me', on_click=click_button2)
    if st.session_state.button:
        # The message and nested widget will remain on the page
        st.write('Button is on!')
        st.slider('Select a value')
    else:
        st.write('Button is off!')
    print(st.session_state.button)


# -------------------------------------------------

def set_state(i):
    st.session_state.stage = i

def test6():
    if 'stage' not in st.session_state:
        st.session_state.stage = 0

    if st.session_state.stage == 0:
        st.button('Begin', on_click=set_state, args=[1])

    if st.session_state.stage >= 1:
        name = st.text_input('Name', on_change=set_state, args=[2])

    if st.session_state.stage >= 2:
        st.write(f'Hello {name}!')
        color = st.selectbox(
            'Pick a Color',
            [None, 'red', 'orange', 'green', 'blue', 'violet'],
            on_change=set_state, args=[3]
        )
        if color is None:
            set_state(2)

    if st.session_state.stage >= 3:
        st.write(f':{color}[Thank you!]')
        st.button('Start Over', on_click=set_state, args=[0])
        
# -------------------------------------------------

def test7():
    if st.session_state.get('clear'):
        st.session_state['name'] = ''
    if st.session_state.get('streamlit'):
        st.session_state['name'] = 'Streamlit'

    st.text_input('Name', key='name')

    st.button('Clear name', key='clear')
    st.button('Streamlit!', key='streamlit')


if __name__ == '__main__':
    test7()
