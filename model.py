import numpy as np
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, GlobalAveragePooling2D, Activation, SeparableConv2D, MaxPool2D, AveragePooling2D
from keras.layers import BatchNormalization, concatenate, add
from keras import backend as K


# generic model design
def model_fn(actions,N=2,filters=24):
    B = len(actions) // 4

    cell_index=1
    ip = Input(shape=(32, 32, 3))
    ip1= ip
    ip2= ip
    stride1=(1,1)
    stride2=(1,1)

    #normal cell
    for i in range(N):
        x=build_cell(ip1,ip2, filters, actions, B,'cell_'+str(cell_index),stride1,stride2)
        ip1=ip2
        ip2=x
        cell_index+=1
    
    #recudtion cell
    stride1=(2,2)
    stride2=(2,2)
    filters=filters*2
    x=build_cell(ip1,ip2, filters, actions, B,'cell_'+str(cell_index),stride1,stride2)
    ip1=ip2
    ip2=x
    stride2=(1,1)
    cell_index+=1

    #normal cell
    for i in range(N):
        x=build_cell(ip1,ip2, filters, actions, B,'cell_'+str(cell_index),stride1,stride2)
        ip1=ip2
        ip2=x
        stride1=(1,1)
        cell_index+=1

    #recudtion cell
    stride1=(2,2)
    stride2=(2,2)
    filters=filters*2
    x=build_cell(ip1,ip2, filters, actions, B,'cell_'+str(cell_index),stride1,stride2)
    ip1=ip2
    ip2=x
    stride2=(1,1)
    cell_index+=1

    #normal cell
    for i in range(N):
        x=build_cell(ip1,ip2, filters, actions, B,'cell_'+str(cell_index),stride1,stride2)
        ip1=ip2
        ip2=x
        stride1=(1,1)
        cell_index+=1

    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation='softmax')(x)

    model = Model(ip, x)
    return model

def parse_action(ip, filters, action,input_name, strides=(1, 1)):
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
    assert isinstance(action, str)
    name = input_name + '_' + action.replace(' ', '_')
    
    # applies a 3x3 separable conv
    if action == '3x3 dconv':
        x = SeparableConv2D(filters, (3, 3), strides=strides, padding='same',name=name)(ip)
        x = BatchNormalization(name=name + '_bn')(x)
        x = Activation('relu')(x)
        
        return x

    # applies a 5x5 separable conv
    if action == '5x5 dconv':
        x = SeparableConv2D(filters, (5, 5), strides=strides, padding='same',name=name)(ip)
        x = BatchNormalization(name=name + '_bn')(x)
        x = Activation('relu')(x)

        return x

    # applies a 7x7 separable conv
    if action == '7x7 dconv':
        x = SeparableConv2D(filters, (7, 7), strides=strides, padding='same',name=name)(ip)
        x = BatchNormalization(name=name + '_bn')(x)
        x = Activation('relu')(x)

        return x

    # applies a 1x7 and then a 7x1 standard conv operation
    if action == '1x7-7x1 conv':
        x = Conv2D(filters, (1, 7), strides=strides, padding='same',name=input_name + '_1x7_conv')(ip)
        x = BatchNormalization(name=input_name + '_1x7_conv_bn')(x)

        x = Activation('relu')(x)
        x = Conv2D(filters, (7, 1), strides=(1, 1), padding='same',name=input_name + '_7x1_conv')(x)
        x = BatchNormalization(name=input_name + '_7x1_conv_bn')(x)

        x = Activation('relu')(x)
        return x

    # applies a 3x3 standard conv
    if action == '3x3 conv':
        x = Conv2D(filters, (3, 3), strides=strides, padding='same',name=name)(ip)
        x = BatchNormalization(name=name + '_bn')(x)
        x = Activation('relu')(x)

        return x

    # applies a 3x3 maxpool
    if action == '3x3 maxpool':
        return MaxPool2D((3, 3), strides=strides, padding='same',name=name)(ip)

    # applies a 3x3 avgpool
    if action == '3x3 avgpool':
        return AveragePooling2D((3, 3), strides=strides, padding='same',name=name)(ip)

    # attempts a linear operation (if size matches) or a strided linear conv projection to reduce spatial depth
    if strides == (2, 2):
        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        input_filters = K.int_shape(ip)[channel_axis]
        x = Conv2D(input_filters, (1, 1), strides=strides, padding='same',name=input_name + '_1x1_conv')(ip)
        x = BatchNormalization(name=input_name + '_1x1_conv_bn')(x)
        x = Activation('relu')(x)

        return x
    else:
        # else just submits a linear layer if shapes match
        return Activation('linear')(ip)


def build_cell(ip1,ip2, filters, action_list, B,name, stride1=(1,1),stride2=(1,1)):
    inputs=[ip1,ip2]
    stride=[stride1,stride2]
    for i in range(B):
        stride.append((1,1))

    # calibrate input tensor shape (number of channels)
    for i, ip in enumerate(inputs):
        if ip.shape[-1] != filters:
            x = Conv2D(filters, (1, 1), padding='same', name='{}_cali_in_conv_{}'.format(name, i))(ip)
            x = BatchNormalization(name='{}_cali_in_bn_{}'.format(name, i))(x)
            x = Activation('relu')(x)
            inputs[i] = x

    # build blocks
    actions = []
    for i in range(B):
        index=action_list[i*4]+1
        left_action = parse_action(inputs[index], filters, action_list[i * 4+1], strides=stride[index],
                                   input_name='{}_block_{}_left_{}'.format(name, i + 1,index))
        index=action_list[i*4+2]+1
        right_action = parse_action(inputs[index], filters, action_list[i * 4+3], strides=stride[index],
                                    input_name='{}_block_{}_right_{}'.format(name, i + 1,index))
        action = concatenate([left_action, right_action], axis=-1)
        actions.append(action)
        inputs.append(action)

    # calibrate output tensor shape
    for i, x in enumerate(actions):
        x = Conv2D(filters, (1, 1), padding='same', name='{}_cali_out_conv_{}'.format(name, i + 1))(x)
        actions[i] = x
    if len(actions) > 1:
        x = add(actions)
    else:
        x = actions[0]
    x = BatchNormalization(name='{}_cali_out_bn'.format(name))(x)
    x = Activation('relu')(x)
    return x
