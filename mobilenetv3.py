from keras.layers import Conv2D, DepthwiseConv2D, Dense, GlobalAveragePooling2D,MaxPooling2D
from keras.layers import Activation, BatchNormalization, Add, \
    Multiply, Reshape,Permute,Lambda,Concatenate,multiply,AveragePooling2D

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, GlobalMaxPooling2D,Reshape
from keras.utils.vis_utils import plot_model
import tensorflow as tf

class MobileNetBase:
    def __init__(self, shape, n_class, alpha=1.0):
        """Init

        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor.
            n_class: Integer, number of classes.
            alpha: Integer, width multiplier.
        """
        self.shape = shape
        self.n_class = n_class
        self.alpha = alpha

    def _relu6(self, x):
        """Relu 6
        """
        return K.relu(x, max_value=6.0)

    def _hard_swish(self, x):
        """Hard swish
        """
        return x * K.relu(x + 3.0, max_value=6.0) / 6.0

    def _return_activation(self, x, nl):
        """Convolution Block
        This function defines a activation choice.

        # Arguments
            x: Tensor, input tensor of conv layer.
            nl: String, nonlinearity activation type.

        # Returns
            Output tensor.
        """
        if nl == 'HS':
            x = Activation(self._hard_swish)(x)
        if nl == 'RE':
            x = Activation(self._relu6)(x)

        return x

    def _conv_block(self, inputs, filters, kernel, strides, nl):
        """Convolution Block
        This function defines a 2D convolution operation with BN and activation.

        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution along the width and height.
                Can be a single integer to specify the same value for
                all spatial dimensions.
            nl: String, nonlinearity activation type.

        # Returns
            Output tensor.
        """

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
        x = BatchNormalization(axis=channel_axis)(x)

        return self._return_activation(x, nl)

    def _squeeze(self, inputs):
        """Squeeze and Excitation.
        This function defines a squeeze structure.

        # Arguments
            inputs: Tensor, input tensor of conv layer.
        """
        input_channels = int(inputs.shape[-1])

        x = GlobalAveragePooling2D()(inputs)
        x = Dense(input_channels, activation='relu')(x)
        x = Dense(input_channels, activation='hard_sigmoid')(x)
        x = Reshape((1, 1, input_channels))(x)
        x = Multiply()([inputs, x])

        return x

    def _bottleneck(self, inputs, filters, kernel, e, s, squeeze, nl):
        """Bottleneck
        This function defines a basic bottleneck structure.

        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            e: Integer, expansion factor.
                t is always applied to the input size.
            s: An integer or tuple/list of 2 integers,specifying the strides
                of the convolution along the width and height.Can be a single
                integer to specify the same value for all spatial dimensions.
            squeeze: Boolean, Whether to use the squeeze.
            nl: String, nonlinearity activation type.

        # Returns
            Output tensor.
        """

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        input_shape = K.int_shape(inputs)

        tchannel = int(e)
        cchannel = int(self.alpha * filters)

        r = s == 1 and input_shape[3] == filters

        x = self._conv_block(inputs, tchannel, (1, 1), (1, 1), nl)

        x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = self._return_activation(x, nl)

        if squeeze:
            x = self._squeeze(x)

        x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)

        if r:
            x = Add()([x, inputs])

        return x

    def build(self):
        pass

class Modified_MobileNetBase:
    def __init__(self, shape, n_class, alpha=1.0):
        """Init

        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor.
            n_class: Integer, number of classes.
            alpha: Integer, width multiplier.
        """
        self.shape = shape
        self.n_class = n_class
        self.alpha = alpha

    def _relu6(self, x):
        """Relu 6
        """
        return K.relu(x, max_value=6.0)

    def _hard_swish(self, x):
        """Hard swish
        """
        return x * K.relu(x + 3.0, max_value=6.0) / 6.0

    def _return_activation(self, x, nl):
        """Convolution Block
        This function defines a activation choice.

        # Arguments
            x: Tensor, input tensor of conv layer.
            nl: String, nonlinearity activation type.

        # Returns
            Output tensor.
        """
        if nl == 'HS':
            x = Activation(self._hard_swish)(x)
        if nl == 'RE':
            x = Activation(self._relu6)(x)


        return x

    def _conv_block(self, inputs, filters, kernel, strides, nl):
        """Convolution Block
        This function defines a 2D convolution operation with BN and activation.

        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            strides: An integer or tuple/list of 2 integers,
                specifying the strides of the convolution along the width and height.
                Can be a single integer to specify the same value for
                all spatial dimensions.
            nl: String, nonlinearity activation type.

        # Returns
            Output tensor.
        """

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

        x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
        x = BatchNormalization(axis=channel_axis)(x)

        return self._return_activation(x, nl)

    def _squeeze(self, inputs):
        """Squeeze and Excitation.
        This function defines a squeeze structure.

        # Arguments
            inputs: Tensor, input tensor of conv layer.
        """
        input_channels = int(inputs.shape[-1])

        x = GlobalAveragePooling2D()(inputs)
        x = Dense(input_channels, activation='relu')(x)
        x = Dense(input_channels, activation='hard_sigmoid')(x)
        x = Reshape((1, 1, input_channels))(x)
        x = Multiply()([inputs, x])

        return x

    def cbam_block(self,inputs,ratio=8):
        x=self.channel_attention(inputs)
        x=self.spatial_attention(x)
        return x

    def channel_attention(self,inputs,ratio=8):
        input_channels = int(inputs.shape[-1])

        avg = GlobalAveragePooling2D()(inputs)
        avg = Dense(input_channels//ratio,activation='relu')(avg)
        avg = Dense(input_channels)(avg)
        avg = Reshape((1, 1, input_channels))(avg)
        max = GlobalMaxPooling2D()(inputs)
        max = Dense(input_channels//ratio, activation='relu')(max)
        max = Dense(input_channels)(max)
        max = Reshape((1, 1, input_channels))(max)
        x = Add()([avg,max])
        x = Activation('sigmoid')(x)

        if K.image_data_format() == "channels_first":
            x = Permute((3, 1, 2))(x)
        x = Multiply()([inputs, x])
        return x

    def spatial_attention(self,inputs):
        if K.image_data_format() == "channels_first":
            channel = inputs._keras_shape[1]
            x = Permute((2, 3, 1))(inputs)
        else:
            channel = inputs._keras_shape[-1]
            x = inputs
        avg = Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(x)
        max = Lambda(lambda x: K.max(x, axis=3, keepdims=True))(x)
        concat=Concatenate(axis=3)([avg,max])
        x= Conv2D(filters=1,kernel_size=7,strides=1,padding='same',activation='sigmoid')(concat)
        if K.image_data_format() == "channels_first":
            x = Permute((3, 1, 2))(x)
        return Multiply()([inputs,x])

    def CA_block(self,inputs,reduction=32):
        x_shape=inputs.get_shape().as_list()
        [b,h,w,c]=x_shape
        x_h = AveragePooling2D(pool_size=[1,w],strides=1)(inputs)
        x_w = AveragePooling2D(pool_size=[h,1],strides=1)(inputs)
        x_w = Lambda(tf.transpose,arguments={'perm':[0,2,1,3]})(x_w)
        # x_w = Permute((2,1,3))(x_w)
        y= Concatenate(axis=1)([x_h,x_w])
        mip=max(8,c//reduction)
        y = Conv2D(mip,(1,1),strides=1,padding='same')(y)
        y = BatchNormalization()(y)
        y = Activation(self._hard_swish)(y)
        x_h,x_w= Lambda(tf.split,arguments={'axis':1,'num_or_size_splits':2})(y)
        x_w = Lambda(tf.transpose, arguments={'perm': [0, 2, 1, 3]})(x_w)
        # x_w = Permute((2,1,3))(x_w)
        a_h = Conv2D(c,(1,1),strides=1,padding='same',activation='sigmoid')(x_h)
        a_w = Conv2D(c,(1,1),strides=1,padding='same',activation='sigmoid')(x_w)
        x=Multiply()([inputs,a_h,a_w])
        return x


    def _bottleneck(self, inputs, filters, kernel, e, s, attention, nl):
        """Bottleneck
        This function defines a basic bottleneck structure.

        # Arguments
            inputs: Tensor, input tensor of conv layer.
            filters: Integer, the dimensionality of the output space.
            kernel: An integer or tuple/list of 2 integers, specifying the
                width and height of the 2D convolution window.
            e: Integer, expansion factor.
                t is always applied to the input size.
            s: An integer or tuple/list of 2 integers,specifying the strides
                of the convolution along the width and height.Can be a single
                integer to specify the same value for all spatial dimensions.
            squeeze: Boolean, Whether to use the squeeze.
            nl: String, nonlinearity activation type.

        # Returns
            Output tensor.
        """

        channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
        input_shape = K.int_shape(inputs)

        tchannel = int(e)
        cchannel = int(self.alpha * filters)

        r = s == 1 and input_shape[3] == filters

        x = self._conv_block(inputs, tchannel, (1, 1), (1, 1), nl)

        x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)
        x = self._return_activation(x, nl)

        if attention=='se':
            x = self._squeeze(x)
        elif attention=='cbam':
            x = self.cbam_block(x)
        elif attention=='ca':
            x = self.CA_block(x)


        x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
        x = BatchNormalization(axis=channel_axis)(x)

        if r:
            x = Add()([x, inputs])

        return x

    def build(self):
        pass

class Modified_MobileNet(Modified_MobileNetBase):
    def __init__(self, shape, n_class, alpha=1.0, include_top=True):
        """Init.

        # Arguments
            input_shape: An integer or tuple/list of 3 integers, shape
                of input tensor.
            n_class: Integer, number of classes.
            alpha: Integer, width multiplier.
            include_top: if inculde classification layer.

        # Returns
            MobileNetv3 model.
        """
        super(Modified_MobileNet, self).__init__(shape, n_class, alpha)
        self.include_top = include_top

    def build(self, plot=False):
        """build MobileNetV3 Small.

        # Arguments
            plot: Boolean, weather to plot model.

        # Returns
            model: Model, model.
        """
        inputs = Input(shape=self.shape)
        x = Conv2D(16, (3, 3), padding='same', dilation_rate=(2, 2), strides=(1,1))(inputs)
        x = BatchNormalization()(x)
        x=self._return_activation(x, nl='HS')
        x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)#224->112

        x = self._bottleneck(x, 16, (3,3), e=16, s=2, attention='se', nl='HS')#112->56

        x = self._bottleneck(x, 24, (3,3), e=72, s=2, attention='se', nl='HS')#56->28

        x = self._bottleneck(x, 32, (3,3), e=96, s=2, attention='se', nl='HS')#28->14

        x = self._bottleneck(x, 48, (3,3), e=128, s=1, attention='se', nl='HS')

        x = self._bottleneck(x, 56, (3,3), e=256, s=2, attention='se', nl='HS')#14->7

        x = self._bottleneck(x, 56, (3,3), e=512, s=1, attention='se', nl='HS')
        x = self._bottleneck(x, 56, (3,3), e=512, s=1, attention='se', nl='HS')

        x = self._conv_block(x, 576, (1, 1), strides=(1, 1), nl='HS')
        x = GlobalAveragePooling2D()(x)
        x = Reshape((1, 1, 576))(x)

        x = Conv2D(1024, (1, 1), padding='same')(x)
        x = self._return_activation(x, 'HS')

        if self.include_top:
            x = Conv2D(self.n_class, (1, 1), padding='same', activation='softmax')(x)
            x = Reshape((self.n_class,))(x)

        model = Model(inputs, x)

        if plot:
            plot_model(model, to_file='images/MobileNetv3_small.png', show_shapes=True)

        return model

    # def build(self, plot=False):
    #     """build MobileNetV3 Small.
    #
    #     # Arguments
    #         plot: Boolean, weather to plot model.
    #
    #     # Returns
    #         model: Model, model.
    #     """
    #     inputs = Input(shape=self.shape)
    #     x = Conv2D(16, (3, 3), padding='same',  dilation_rate=(2, 2), strides=(1,1))(inputs)
    #     x = BatchNormalization()(x)
    #     x=self._return_activation(x, nl='HS')
    #     x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    #
    #     x = self._bottleneck(x, 16, (3,3), e=16, s=2, squeeze=True, nl='HS')#112->56
    #
    #     x = self._bottleneck(x, 24, (3,3), e=72, s=2, squeeze=True, nl='HS')#56->28
    #
    #     x = self._bottleneck(x, 32, (3,3), e=96, s=2, squeeze=True, nl='HS')#28->14
    #
    #     x = self._bottleneck(x, 48, (3,3), e=128, s=1, squeeze=True, nl='HS')
    #
    #     x = self._bottleneck(x, 56, (3,3), e=256, s=2, squeeze=True, nl='HS')
    #
    #     x = self._bottleneck(x, 56, (3,3), e=512, s=1, squeeze=True, nl='HS')
    #     x = self._bottleneck(x, 56, (3,3), e=512, s=1, squeeze=True, nl='HS')
    #
    #     x = self._conv_block(x, 576, (1, 1), strides=(1, 1), nl='HS')
    #     x = GlobalAveragePooling2D()(x)
    #     x = Reshape((1, 1, 576))(x)
    #
    #     x = Conv2D(1024, (1, 1), padding='same')(x)
    #     x = self._return_activation(x, 'HS')
    #
    #     if self.include_top:
    #         x = Conv2D(self.n_class, (1, 1), padding='same', activation='softmax')(x)
    #         x = Reshape((self.n_class,))(x)
    #
    #     model = Model(inputs, x)
    #
    #     if plot:
    #         plot_model(model, to_file='images/MobileNetv3_small.png', show_shapes=True)
    #
    #     return model