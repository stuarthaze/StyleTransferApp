# Overview
A simple app that uses a neural network to merge the content of one image with the style of a second image according to the algorithm proposed by Gatys *et al.*  [(article link)](https://arxiv.org/abs/1508.06576)

### Implementation
This was based on the final programming assignment in the Coursera Deep Learning specialization on convolutional networks. It uses activations from a subset of six layers of the pre-trained VGG-19 convolutional network to calculate content and style cost functions for a generated image with respect to reference content and style images. The pixels of the generated image are optimized by gradient descent using tensorflow and the key parameters are exposed to the user via the interface.

### Requirements
```
streamlit
tensorflow
numpy
```

### Running
The app can be run at the command line:  
`streamlit run app.py`