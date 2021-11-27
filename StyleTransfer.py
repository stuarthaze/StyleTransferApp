import streamlit as st
import os
from PIL import Image
import numpy as np
import tensorflow as tf

st.title('Art Style Transfer')
'''
This app uses the method of neural style transfer proposed by Gatys *et al.*  [(article link)](https://arxiv.org/abs/1508.06576) to merge the content of one image with the style of a second image
'''

#----- Load Images --------


col1, col2 = st.columns(2)
with col1:
    '''
    ## Content Image
    '''
    content_img_loc = st.empty()
    content_type = st.radio('Select Content', ('Default', 'Uploaded'))
    new_content = st.file_uploader('Upload new image', type=['png','jpg'])

    if content_type == 'Default':
        content_image_orig = Image.open('images/BigBen.jpg')
    elif content_type == 'Uploaded':
        content_image_orig = Image.open(new_content)
    
    content_img_loc.image(content_image_orig)

with col2:
    '''
    ## Style image
    '''
    style_img_loc = st.empty()
    style_type = st.radio('Select Style', ('Wave', 'Ink', 'Picasso', 'Uploaded'))
    new_style = st.file_uploader('Upload new style image', type=['png','jpg'])
    
    if style_type == 'Ink':
        style_image_orig = Image.open('images/japanese_town.jpg')
    elif style_type == 'Wave':
        style_image_orig = Image.open('images/The_Great_Wave_off_Kanagawa.jpg')
    elif style_type == 'Picasso':
        style_image_orig = Image.open('images/Picasso.jpg')
    elif style_type == 'Uploaded':
        style_image_orig = Image.open(new_style)

    style_img_loc.image(style_image_orig)

#End Image Loading
#----------------------------------------
#Enter user-defined hyper parameters

'''
## Model Parameters
'''
col31, col32, col33 = st.columns(3)
with col31:
    img_size = st.number_input(label='NN Input Size', min_value=100, max_value= 1000, value=300)
    noise_intensity = st.number_input(label='Noise level on initialization', min_value=0.0, max_value= 1.0, value=0.1)
    style_ratio = st.number_input(label='Style to Content Ratio', min_value=0., max_value= 1., value=0.3)


with col32:
    optimizer_name = st.selectbox('Optimizer', ('Adam','SGD'))
    learning_rate = st.number_input(label='Learning Rate', min_value=0.00001, max_value= 1., value=0.01)
    momentum = st.number_input(label='Momentum (SGD only)', min_value=0.0001, max_value= 1., value=0.85)

with col33:
    style_wts = []
    for i in range(5):
        wt = st.number_input(label=f'Style Weight - Block{i+1}', min_value=0., max_value= 1., value=0.2)
        style_wts.append(wt)

#End of user-defined parameters
#------------------------------------------------------------------------
#Build the model

vgg = tf.keras.applications.VGG19(include_top=False,
                                  input_shape=(img_size, img_size, 3),
                                  weights='vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

vgg.trainable = False

def compute_content_cost(content_output, generated_output):
    """
    Computes the content cost
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    a_C = content_output[-1]
    a_G = generated_output[-1]
    
    # Retrieve dimensions from a_G
    _, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape a_C and a_G 
    a_C_unrolled = tf.reshape(a_C, shape=[1, n_H*n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[1, n_H*n_W, n_C])
    
    # compute the cost with tensorflow 
    J_content = tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))/(4*n_H*n_W*n_C)
    
    return J_content


def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """  
    GA = tf.matmul(A, tf.transpose(A))
    return GA


def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    # Retrieve dimensions from a_G (≈1 line)
    _, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape the images from (n_H * n_W, n_C) to have them of shape (n_C, n_H * n_W) (≈2 lines)
    a_S = tf.transpose(tf.reshape(a_S, shape=[n_H*n_W, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, shape=[n_H*n_W, n_C]))

    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss (≈1 line)
    J_style_layer = tf.reduce_sum(tf.square(tf.subtract(GS, GG)))/((2 * n_C * n_H * n_W)**2)
    
    return J_style_layer


STYLE_LAYERS = [
    ('block1_conv1', style_wts[0]),
    ('block2_conv1', style_wts[1]),
    ('block3_conv1', style_wts[2]),
    ('block4_conv1', style_wts[3]),
    ('block5_conv1', style_wts[4])]


def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    Arguments:
    style_image_output -- our tensorflow model
    generated_image_output --
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    Returns: 
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    
    # initialize the overall style cost
    J_style = 0

    # Set a_S to be the hidden layer activation from the layer we have selected.
    # The last element of the array contains the content layer image, which must not be used.
    a_S = style_image_output[:-1]

    # Set a_G to be the output of the choosen hidden layers.
    # The last element of the list contains the content layer image which must not be used.
    a_G = generated_image_output[:-1]
    for i, weight in zip(range(len(a_S)), STYLE_LAYERS):  
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])

        # Add weight * J_style_layer of this layer to overall style cost
        J_style += weight[1] * J_style_layer

    return J_style


@tf.function()
def total_cost(J_content, J_style, alpha = 0.5, beta = 0.5):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost
    
    Returns:
    J -- total cost as defined by the formula above.
    """
    J = alpha*J_content + beta*J_style

    return J


def get_layer_outputs(vgg, layer_names):
    """ Creates a vgg model that returns a list of intermediate output values."""
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]

    model = tf.keras.Model([vgg.input], outputs)
    return model


def clip_0_1(image):
    """
    Truncate all the pixels in the tensor to be between 0 and 1
    
    Arguments:
    image -- Tensor
    J_style -- style cost coded above

    Returns:
    Tensor
    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def tensor_to_image(tensor):
    """
    Converts the given tensor into a PIL image
    
    Arguments:
    tensor -- Tensor
    
    Returns:
    Image: A PIL image
    """
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)



#Prepare images for TensorFlow
content_image = np.array(content_image_orig.resize((img_size, img_size)))
content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))

style_image =  np.array(style_image_orig.resize((img_size, img_size)))
style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))

generated_image = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
noise_intensity = 0.2
noise = tf.random.uniform(tf.shape(generated_image), -noise_intensity, noise_intensity)
generated_image = tf.add(generated_image, noise)
generated_image = tf.clip_by_value(generated_image, clip_value_min=0.0, clip_value_max=1.0)
generated_image = tf.Variable(generated_image)

content_layer = [('block5_conv4', 1)]
vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)
content_target = vgg_model_outputs(content_image)  # Content encoder
style_targets = vgg_model_outputs(style_image)     # Style enconder

preprocessed_content =  tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
a_C = vgg_model_outputs(preprocessed_content)

preprocessed_style =  tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
a_S = vgg_model_outputs(preprocessed_style)

#--------Optimizer------------
if optimizer_name == 'Adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    if learning_rate > 0.1:
        st.write('Warning: Learning rate might be too high for Adam method')
else:
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)

@tf.function()
def train_step(generated_image):
    with tf.GradientTape() as tape:
        
        a_G = vgg_model_outputs(generated_image)
        
        J_style = compute_style_cost(a_S, a_G, STYLE_LAYERS=STYLE_LAYERS)
        J_content = compute_content_cost(a_C, a_G)

        J = total_cost(J_content, J_style, alpha = 1-style_ratio, beta = style_ratio)
        
    grad = tape.gradient(J, generated_image)

    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(clip_0_1(generated_image))
    return J

#End of model
#--------------------------------------------------------------------
'''
## Fit Model
'''
costs = []
epochs = st.number_input(label='Number of epochs', min_value=1, max_value=10000, value=10)

show_training = st.checkbox(label='Show training data')
train_checked = st.button('Start')

prog_bar = st.progress(0)
gen_img_loc = st.empty()
img_temp = tensor_to_image(generated_image)
gen_img_loc.image(img_temp.resize(content_image_orig.size))

if train_checked:
    for i in range(epochs):
        prog_bar.progress(int(100*(i+1)/epochs))
        train_step(generated_image)
        Jcost = train_step(generated_image)
        costs.append(Jcost)
        if i % 5 == 0 or i == (epochs-1):
            img_temp = tensor_to_image(generated_image)
            gen_img_loc.image(img_temp.resize(content_image_orig.size))

    if show_training:
        st.subheader('Cost function')
        st.line_chart(costs)

