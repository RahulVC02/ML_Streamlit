# Create a simple streamlit app to show ridge regression fit with
# varying degrees and varying penalty. Use a 1d dataset. Also in the title show the magnitude of coefficients.

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
# from sklearn.pipeline import make_pipeline
import random
import librosa
from keras.applications.vgg16 import VGG16
from matplotlib.animation import FuncAnimation
import cv2
import tensorflow as tf
import streamlit.components.v1 as components



##commented to increase speed

# metadata = pd.read_csv('meta/esc50.csv')
# mask = metadata['esc10'] == True
# metadata = metadata[mask]
# my_classes = metadata['category'].unique()


my_classes = ['dog','chainsaw','crackling_fire','helicopter','rain','crying_baby', 'clock_tick','sneezing','rooster','sea_waves']
audio_paths = {'dog':['data/1-100032-A-0.wav','data/1-110389-A-0.wav','data/1-30226-A-0.wav'], 'chainsaw':['data/1-116765-A-41.wav','data/1-19898-A-41.wav','data/1-19898-B-41.wav'], 'crackling_fire':['data/1-17150-A-12.wav','data/1-17565-A-12.wav','data/1-17742-A-12.wav'], 'helicopter':['data/1-172649-A-40.wav','data/1-172649-B-40.wav','data/1-172649-C-40.wav'], 'rain':['data/1-17367-A-10.wav','data/1-21189-A-10.wav','data/1-26222-A-10.wav'], 'crying_baby':['data/1-187207-A-20.wav','data/1-211527-A-20.wav','data/1-211527-B-20.wav'], 'clock_tick':['data/1-21934-A-38.wav','data/1-21935-A-38.wav','data/1-35687-A-38.wav'], 'sneezing':['data/1-26143-A-21.wav','data/1-29680-A-21.wav','data/1-31748-A-21.wav'], 'rooster':['data/1-26806-A-1.wav','data/1-27724-A-1.wav','data/1-34119-B-1.wav'], 'sea_waves':['data/1-28135-A-11.wav','data/1-28135-B-11.wav','data/1-39901-A-11.wav']}
                          

# Create a streamlit app
st.title("2D CNN for Audio Classification")
st.write(
    "This app shows a 2D CNN model trained on the ESC-10 dataset. It displays the spectrogram of the audio file and the predicted class. It also displays an animation of the saliency maps at each layer of the network. I have used the VGG16 architecture for this task. The model was trained on the ESC-10 dataset."
) 

st.write(
    "The drop-down menu allows you to select a class and a random audio file from that class is selected. The spectrogram of the audio file is displayed and the predicted class is shown. The animation shows the saliency maps at each layer of the network. The animation is paused by default. You can click on the play button to start the animation. The animation may take a small amount of time to load."
)
    

# The sidebar contains the sliders
with st.sidebar:
    
    #create a slider to select audio class and file
    class_name = st.selectbox('Select Class', my_classes)
    # class_name = st.sidebar.slider('Select an option', my_classes, orientation='vertical')
    file_list = audio_paths[class_name]
    audio_file_path = random.choice(file_list)



audio, sr = librosa.load(audio_file_path, sr=44100)
mel_spec = librosa.feature.melspectrogram(y = audio, sr=22050, n_fft=2048, hop_length=1024, n_mels=128)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
input_shape = (224, 224)
resized_mel_spec = cv2.resize(mel_spec_db, input_shape)

fig, ax = plt.subplots()
librosa.display.specshow(resized_mel_spec, y_axis='mel', fmax=8000, x_axis='time')
plt.title(f'Mel Spectrogram for {class_name}')
plt.colorbar(format='%+2.0f dB')
st.pyplot(fig)


model = VGG16(weights='imagenet', include_top=True)
mel_spec = librosa.feature.melspectrogram(y = audio, sr=22050, n_fft=2048, hop_length=1024, n_mels=128)
mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
input_shape = (224, 224)
resized_mel_spec = cv2.resize(mel_spec_db, input_shape)

resized_mel_spec_rgb = np.stack((resized_mel_spec,) * 3, axis=-1)
resized_mel_spec_rgb = resized_mel_spec_rgb - resized_mel_spec_rgb.min()
resized_mel_spec_rgb = resized_mel_spec_rgb / resized_mel_spec_rgb.max()
resized_mel_spec_rgb = resized_mel_spec_rgb * 255
resized_mel_spec_rgb = np.ceil(resized_mel_spec_rgb).astype(np.uint8)

layers = model.layers
layer_outputs = [layer.output for layer in layers]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)
image = resized_mel_spec_rgb

# Compute the activations of all layers for the chosen image and get the output of each layer as a list of outputs of it's filters
activations = activation_model.predict(image[np.newaxis,...])
output=[]


# fig2,ax2 = plt.subplots()
# Display the saliency maps after each layer of the architecture
for i, activation in enumerate(activations):

    #corresponding to the convolutional layers
    if len(activation.shape) == 4:
        # Compute the saliency map for this layer by taking the mean of the activations for all filters
        saliency = np.abs(activation).mean(axis=-1)
        
        # Normalize the saliency map to have values between 0 and 1
        saliency -= saliency.min()
        saliency /= saliency.max()
        
        # Display the saliency map
        plt.figure()
        plt.title(f'Layer {i+1}')
        plt.axis('off')
        output.append(saliency[0])

fig2, ax2 = plt.subplots()
# Create a gif of the saliency maps
def update(frame):
    ax2.clear()
    ax2.imshow(output[frame], cmap='viridis')
    ax2.set_title(f"Layer {frame}")


anim = FuncAnimation(fig2, update, frames = 19, interval = 1000)
# st.pyplot(fig2)

#HtmlFile = line_ani.to_html5_video()
with open("myvideo.html","w") as f:
  print(anim.to_html5_video(), file=f)
  
HtmlFile = open("myvideo.html", "r")
source_code = HtmlFile.read() 
components.html(source_code, height = 500,width=900)


# Line between the model and the plot
st.write("I have used the viridis colormap for the animation. Blue shades represent lower values while yellow shades represent higher values.")
