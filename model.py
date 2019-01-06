import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D, Activation, SeparableConv2D, MaxPool2D, AveragePooling2D, concatenate
from keras.layers import BatchNormalization
from keras import backend as K

# generic model design
def model_fn(actions,N=2,filters=24):
    B = len(actions) // 4
    
    ip = Input(shape=(32, 32, 3))
    ip1= ip
    ip2= ip
    stride1=(1,1)
    stride2=(1,1)

    #normal cell
    for i in range(N):
        x=build_cell(ip1,ip2, filters, actions, B,stride1,stride2)
        ip1=ip2
        ip2=x
    
    #recudtion cell
    stride1=(2,2)
    stride2=(2,2)
    filters=filters*2
    x=build_cell(ip1,ip2, filters, actions, B,stride1,stride2)
    ip1=ip2
    ip2=x
    stride2=(1,1)

    #normal cell
    for i in range(N):
        x=build_cell(ip1,ip2, filters, actions, B,stride1,stride2)
        ip1=ip2
        ip2=x
        stride1=(1,1)
        
    #recudtion cell
    stride1=(2,2)
    stride2=(2,2)
    filters=filters*2
    x=build_cell(ip1,ip2, filters, actions, B,stride1,stride2)
    ip1=ip2
    ip2=x
    stride2=(1,1)
    
    #normal cell
    for i in range(N):
        x=build_cell(ip1,ip2, filters, actions, B,stride1,stride2)
        ip1=ip2
        ip2=x
        stride1=(1,1)

    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(ip, x)
    return model

def parse_action(ip, filters, action, strides=(1, 1)):
    '''
    Parses the input string as an action. Certain cases are handled incorrectly,
    so that model can still be built, albeit not with original specification

    Args:
        ip: input tensor
        filters: number of filters
        action: action string
        strides: stride to reduce spatial size

    Returns:
        a tensor with an action performed
    '''
    # applies a 3x3 separable conv
    if action == '3x3 dconv':
        x = SeparableConv2D(filters, (3, 3), strides=strides, padding='same')(ip)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    # applies a 5x5 separable conv
    if action == '5x5 dconv':
        x = SeparableConv2D(filters, (5, 5), strides=strides, padding='same')(ip)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    # applies a 7x7 separable conv
    if action == '7x7 dconv':
        x = SeparableConv2D(filters, (7, 7), strides=strides, padding='same')(ip)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    # applies a 1x7 and then a 7x1 standard conv operation
    if action == '1x7-7x1 conv':
        x = Conv2D(filters, (1, 7), strides=strides, padding='same')(ip)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(filters, (7, 1), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    # applies a 3x3 standard conv
    if action == '3x3 conv':
        x = Conv2D(filters, (3, 3), strides=strides, padding='same')(ip)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    # applies a 3x3 maxpool
    if action == '3x3 maxpool':
        return MaxPool2D((3, 3), strides=strides, padding='same')(ip)

    # applies a 3x3 avgpool
    if action == '3x3 avgpool':
        return AveragePooling2D((3, 3), strides=strides, padding='same')(ip)

    # attempts a linear operation (if size matches) or a strided linear conv projection to reduce spatial depth
    if strides == (2, 2):
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        input_filters = K.int_shape(ip)[channel_axis]
        x = Conv2D(input_filters, (1, 1), strides=strides, padding='same')(ip)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x
    else:
        # else just submits a linear layer if shapes match
        return Activation('linear')(ip)

"""
def build_cell(ip, filters, actions, B, stride):
    # if cell size is 1 block only
    if B == 1:
        left = parse_action(ip, filters, actions[1], strides=stride)
        right = parse_action(ip, filters, actions[3], strides=stride)
        return concatenate([left, right], axis=-1)

    # else concatenate all the intermediate blocks
    actions = []
    for i in range(B):
        left_action = parse_action(ip, filters, actions[i * 4+1], strides=stride)
        right_action = parse_action(ip, filters, actions[i * 4+3], strides=stride)
        action = concatenate([left_action, right_action], axis=-1)
        actions.append(action)

    # concatenate the final blocks as well
    op = concatenate(actions, axis=-1)
    return op
"""
def build_cell(ip1,ip2, filters, actions, B, stride1=(1,1),stride2=(1,1)):
    # if cell size is 1 block only
    inputs=[ip1,ip2]
    stride=[stride1,stride2]
    input_filters=filters
    for i in range(B):
        stride.append((1,1))
    if B == 1:
        index=actions[0]+1
        left = parse_action(inputs[index], input_filters, actions[1], strides=stride[index])
        index=actions[2]+1
        right = parse_action(inputs[index], input_filters, actions[3], strides=stride[index])
        return concatenate([left, right], axis=-1)

    # else concatenate all the intermediate blocks
    actions = []
    for i in range(B):
        index=actions[i*4]+1
        left_action = parse_action(inputs[index], input_filters, actions[i * 4+1], strides=stride[index])
        index=actions[i*4+2]+1
        right_action = parse_action(inputs[index], input_filters, actions[i * 4+3], strides=stride[index])
        action = concatenate([left_action, right_action], axis=-1)
        actions.append(action)
        inputs.append(action)

    # concatenate the final blocks as well
    op = concatenate(actions, axis=-1)
    return op


