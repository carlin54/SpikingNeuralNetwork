#######################################
#             load mnist              #
import mnist
import numpy as np


def normalize(img):
    fac = 0.99 / 255
    return img * fac + 0.01


def digit_to_layer(digit):
    return (np.arange(10) == digit).astype(np.float)


train_images = np.array([normalize(img) for img in mnist.train_images()])
train_labels = np.array([digit_to_layer(digit) for digit in mnist.train_labels()])

test_images = np.array([normalize(img) for img in mnist.test_images()])
test_labels = np.array([digit_to_layer(digit) for digit in mnist.test_labels()])
###
import math
from functools import reduce

padding = 'valid'
padding = 'same'
padding = 'full'


# I x I x C
# O x O x K




def init_tuple_counter(count_to: tuple) -> tuple:
    return tuple(np.zeros(len(count_to.shape), dtype=int))


def adder(counter: tuple, max: tuple) -> tuple:
    if counter == max:
        return counter
    counter_array = np.array(counter)
    length = len(counter_array)
    carry = True
    for i in range(length - 1, -1, -1):
        counter_array[i] = counter_array[i] + 1
        carry = False
        if counter_array[i] > max[i]:
            counter_array[i] = 0
            carry = True

        if not carry:
            break

    counted = [max[:-1] == counter_array[:-1]]
    if carry and counted:
        counter_array = max

    return tuple(counter_array)


def conv2d(input: np.array, output: np.array, filters: np.array, stride: tuple([int, int]) = (1, 1)) \
        -> np.array:
    ## padding needs to be implemented
    ## proper strides
    kernel_y = len(filters)
    kernel_x = len(filters[0])
    kernel_channels = len(filters[0][0])
    num_filters = len(filters[0][0][0])

    batch_shape = input.shape[:-3]

    layer_shape = input.shape[-3:]
    layer_height = layer_shape[0]
    layer_width = layer_shape[1]
    layer_channel = layer_shape[2]

    stride_x = stride[0]
    stride_y = stride[1]

    padding = 0

    ## assert padding is valid I x I x K
    conv_out_height = int(((layer_height - kernel_y + 2 * padding) / stride_y)) \
                      + 1
    conv_out_width = int(((layer_width - kernel_x + 2 * padding) / stride_x)) \
                     + 1
    conv_shape = batch_shape + (conv_out_height, conv_out_width, num_filters)
    # conv_out = np.ndarray(shape=conv_shape)

    batch_idx = np.zeros(len(batch_shape), dtype=int)
    while batch_idx != batch_shape:  ## probably needs to be changed
        layer = input[tuple(batch_idx)]
        for y_idx in range(0, conv_out_height):
            y_start = y_idx * stride_y
            y_end = (y_idx * stride_y + kernel_y)
            for x_idx in range(0, conv_out_width):
                x_start = x_idx * stride_x
                x_end = (x_idx * stride_x + kernel_x)

                kernel = layer[y_start:y_end, x_start:x_end]

                for filter_idx in range(num_filters):
                    filter = filters[:, :, :, filter_idx]
                    multi = np.multiply(kernel, filter)
                    product_idx = (y_idx, x_idx, filter_idx)
                    output[tuple(batch_idx) + product_idx] = np.sum(multi)

        batch_idx = adder(batch_idx, batch_shape)

    return output


def conv_output_size(layer_dimensions: tuple, kernel_dimensions: tuple,
                     stride_dimensionsensions: tuple, padding: int):
    return (int(((layer_dimensions[0] - kernel_dimensions[0] + 2 * padding) \
                 / stride_dimensionsensions[0])) + 1,
            int(((layer_dimensions[1] - kernel_dimensions[1] + 2 * padding) \
                 / stride_dimensionsensions[1])) + 1,
            kernel_dimensions[3])


def generate_conv2d_filters(kernel_dimensions: tuple, k: float = 2.0) -> np.array:
    kernel_y = kernel_dimensions[0]
    kernel_x = kernel_dimensions[1]
    kernel_channels = kernel_dimensions[2]
    num_filters = kernel_dimensions[3]

    filters = np.ndarray(shape=kernel_dimensions)
    filter_shape = tuple([kernel_y, kernel_x, kernel_channels])

    nl = kernel_x * kernel_y * kernel_channels
    std = math.sqrt(k / nl)

    for filter_idx in range(num_filters):
        filter = np.random.normal(scale=std, size=nl)
        filter = filter.reshape(filter_shape)
        filters[:, :, :, filter_idx] = filter

    return filters


def lif_neuron(Vm: float, V_reset: float, V_th: float, tau_m: float, fire=True,
               leaky=True) -> np.array:
    if Vm >= V_th and fire:
        spike = 1
        Vm = V_reset
    else:
        spike = 0
        if leaky:
            Vm = Vm * math.exp(-1 / tau_m)

    return [Vm, spike]


def flatten(input: np.array, output: np.array, flatten_dim: int):
    self.input_shape
    batch_dimensions = input.shape[:flatten_dim]
    flattened_dimension = tuple([math.prod(input.shape[flatten_dim:])])
    output = np.reshape(input, batch_dimensions + flattened_dimension)
    return output

def lif_neuron_pool(Vin: np.array,
                    Vout: np.array,
                    spike_out: np.array,
                    Vreset: float = 0,
                    Vth: float = 0.75,
                    tau_m: int = 100,
                    fire: bool = True,
                    leaky: bool = True,
                    time_index: int = 0) -> np.array:
    # [batch][time][spike_train]
    # [batch][      Vin        ]

    # adequate dimensions to process

    # a dimensions to
    # assert (len(Vin.shape[-4]) > 2)

    #if (Vin != NULL):
    #    s = 1  # TODO: implement smth here

    # generate output arrays
    # Vout = np.zero(shape=(Vin.shape))
    # spike_out = np.zero(shape=(Vin.shape))

    assert(Vin.shape == Vout.shape)

    # process batches
    batch_dimensions = Vin.shape[:max(time_index-1,0)]
    spike_train_length = Vin.shape[time_index]
    membrane_dimensions = Vin.shape[time_index+1:]



    for batch_idx in np.ndindex(batch_dimensions):

        for neuron_idx in np.ndindex(membrane_dimensions):

            for t_idx in range(1, spike_train_length):
                # membrane voltage for this step
                t_current = batch_idx + tuple([t_idx]) + neuron_idx
                t_previous = batch_idx + tuple([t_idx - 1]) + neuron_idx
                Vm = Vin[t_current] + Vout[t_previous]

                # simulate lif-neuron
                [Vout[t_current], spike_out[t_current]] = lif_neuron(Vm, Vreset, Vth, tau_m, fire, leaky)

    return [Vout, spike_out]


def generate_spike_train(p: float, t: int) -> np.array:
    dist = np.random.uniform(1, 0, t)
    return np.array([int(item < p) for item in dist])


def generate_layer_spike_train(layer: np.array, train_length: int):
    layer_height = len(layer)
    layer_width = len(layer[0])
    spike_layer = np.ndarray(shape=(train_length, layer_height, layer_width, 1))
    for y in range(0, layer_height):
        for x in range(0, layer_width):
            train = np.array(generate_spike_train(layer[y][x], train_length))
            for t in range(0, train_length):
                spike_layer[t, y, x, 0] = train[t]
    return spike_layer


def avg_pool(input: np.array, output:np.array, kernel_size: tuple([int, int]) = (2, 2), stride: tuple([int, int]) = (1, 1)) -> np.array:

    pool = output
    ## padding needs to be implemented
    ## proper strides
    kernel_y = kernel_size[1]
    kernel_x = kernel_size[0]

    batch_shape = input.shape[:-3]

    layer_shape = input.shape[-3:]
    layer_height = layer_shape[0]
    layer_width = layer_shape[1]
    layer_channel = layer_shape[2]

    stride_x = stride[0]
    stride_y = stride[1]

    padding = 0

    pool_height = int(((layer_height - kernel_y + 2 * padding) / stride_y)) + 1
    pool_width = int(((layer_width - kernel_x + 2 * padding) / stride_x)) + 1
    pool_shape = batch_shape + (pool_height, pool_width, layer_channel)
    # pool = np.ndarray(shape=pool_shape)

    # TODO: Update this code
    batch_idx = np.zeros(len(batch_shape), dtype=int)
    while batch_idx != batch_shape:
        layer = input[tuple(batch_idx)]
        for y_idx in range(0, pool_height):
            y_start = y_idx * stride_y
            y_end = (y_idx * stride_y + kernel_y)
            for x_idx in range(0, pool_width):
                x_start = x_idx * stride_x
                x_end = (x_idx * stride_x + kernel_x)
                for channel_idx in range(0, layer_channel):
                    kernel = layer[y_start:y_end, x_start:x_end, channel_idx]
                    product = np.sum(kernel) / kernel.size
                    product_idx = (y_idx, x_idx, channel_idx)
                    pool[tuple(batch_idx) + product_idx] = product

        batch_idx = adder(batch_idx, batch_shape)

    return pool


def generate_dense_layer_weights(input_dimensions: tuple, num_neuron_output: int, k: float = 2.0) -> np.array:
    axons_per_neuron = math.prod(input_dimensions)

    synapses = np.ndarray(shape=(num_neuron_output, axons_per_neuron))
    nl = axons_per_neuron
    std = math.sqrt(k / nl)
    for i in range(num_neuron_output):
        synapses[i] = np.random.normal(scale=std, size=nl)

    return synapses


def dense_forward(input_neurons: np.array, output_neurons: np.array, weights: np.array) -> np.array:

    ins = input_neurons.shape
    ons = output_neurons.shape
    ws = weights.shape

    # [batch][spike time]
    batch_dimensions = input_neurons.shape[:-1]

    # [][]
    num_input_neurons = weights.shape[1]
    num_output_neurons = weights.shape[0]

    #[neuron y][neuron x][channel]
    for batch_idx in np.ndindex(batch_dimensions):

        for output_neuron_idx in range(num_output_neurons):
            # action_potential = 0
            # dot product
            # for input_neuron_idx in range(num_input_neurons):
            #    ax = input_neurons[batch_idx][input_neuron_idx]
            #   wx = weights[output_neuron_idx][input_neuron_idx]
            #    action_potential = action_potential + ax*wx

            output_neurons[batch_idx][output_neuron_idx] = np.dot(input_neurons[batch_idx], weights[output_neuron_idx])

    return output_neurons


def generate_membrane(membrane_dimensions: tuple, value: float = 0.0):
    membrane = np.ndarray(shape=membrane_dimensions)
    membrane.fill(value)
    return membrane


# This gains the term da_lif / d_net
def differentiate_spike_train(spike_train, Vth = 1):
    # sum of decay over time
    gamma = sum(spike_train)
    if gamma == 0:
        return 0

    tau_m = len(spike_train)
    total_decay = 0
    t = tk = 1

    for activation in spike_train:

        if activation:
            if t != tk:
                decay = math.exp(-(t - tk) / tau_m)
                total_decay = total_decay - (1 / tau_m) * decay

            tk = t + 1
        t = t + 1

    return (1/Vth) * (1 + (1/gamma) * total_decay)


class Layer:

    def __init__(self):
        self.trainable = True
        self.input_shape = None
        self.output_shape = None

    def count_parameters(self):
        raise NotImplementedError()

    def compute_output_shape(self, input_shape):
        raise NotImplementedError()

    def forward_propagate(self, A):
        raise NotImplementedError()

    def backward_propagate(self, dZ, cache):
        raise NotImplementedError()

    def get_weights(self):
        raise NotImplementedError()

    def set_weights(self, weights):
        raise NotImplementedError()

    def build(self, input_shape):
        self.input_shape = input_shape


class Dropout(Layer):

    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.mask = None

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = input_shape
        self.reset()

    def reset(self):
        self.mask = np.random.binomial(1, 1-self.probability, size=self.output_shape)

    def forward_propagate(self, A):
        masked = np.multiply(self.mask, A)
        cache = [{"mask" : self.mask,
                  "output" : masked}]

        return masked, cache

    def backward_propagate(self, dZ, cache):
        assert(dZ.shape == self.mask.shape)
        return np.multiply(self.mask * dZ)

    def compute_output_shape(self, input_shape):
        return input_shape


class AveragePool2D(Layer):

    def __init__(self, kernel_size, strides):
        super().__init__()
        assert (len(kernel_size) == 2)
        assert (len(strides) == 2)
        self.kernel_size = kernel_size
        self.strides = strides

    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = self.compute_output_shape(input_shape)

    def compute_output_shape(self, input_shape):
        # incorrect input dimensions
        assert(len(input_shape) >= 3)

        # dimensions for sample / instance in the input
        sample_shape = input_shape[-3:]
        sample_y = sample_shape[0]
        sample_x = sample_shape[1]
        sample_channels = sample_shape[2]

        kernel_x = self.kernel_size[1]
        kernel_y = self.kernel_size[0]
        stride_x = self.strides[0]
        stride_y = self.strides[1]

        padding = 0

        return (int(((sample_y - kernel_y + 2 * padding) / stride_y)) + 1,
                int(((sample_x - kernel_x + 2 * padding) / stride_x)) + 1,
                sample_channels)

    def forward_propagate(self, A):
        # padding needs to be implemented
        # separate batches
        batch_shape = A.shape[:-3]

        # unpack sample shape
        sample_shape = A.shape[-3:]
        sample_y = sample_shape[0]
        sample_x = sample_shape[1]
        sample_channels = sample_shape[2]

        # unpack kernel
        kernel_y = self.kernel_size[0]
        kernel_x = self.kernel_size[1]

        # unpack stride shape
        stride_y = self.strides[0]
        stride_x = self.strides[1]

        # unpack pooling layer shape
        pool_shape = self.compute_output_shape(A.shape)
        pool_y = pool_shape[0]
        pool_x = pool_shape[1]

        # initialize the output convolution
        Z_shape = batch_shape + pool_shape
        Z = np.zeros(shape=Z_shape)
        Z.fill(-9e99)

        # begin pooling
        for batch_idx in np.ndindex(batch_shape):
            layer = A[batch_idx]
            for y_idx in range(0, pool_y):
                y_start = y_idx * stride_y
                y_end = (y_idx * stride_y + kernel_y)
                for x_idx in range(0, pool_x):
                    x_start = x_idx * stride_x
                    x_end = (x_idx * stride_x + kernel_x)
                    for channel_idx in range(0, sample_channels):
                        kernel = layer[y_start:y_end, x_start:x_end, channel_idx]
                        product = np.sum(kernel) / kernel.size
                        product_idx = (y_idx, x_idx, channel_idx)
                        Z[batch_idx + product_idx] = product


        return Z, None


class Convolution2D(Layer):

    def __init__(self, number_of_filters, kernel_size, strides):
        super().__init__()
        self.number_of_filters = number_of_filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.filters = []
        self.kernel_shape = []
        self.padding = 0

    def build(self, input_shape):


        k = 2

        sample_shape = input_shape[-3:]
        sample_y = sample_shape[0]
        sample_x = sample_shape[1]
        sample_channels = sample_shape[2]
        self.input_shape = sample_shape

        kernel_y = self.kernel_size[0]
        kernel_x = self.kernel_size[1]
        kernel_channels = sample_channels
        kernel_filters = self.number_of_filters

        self.kernel_shape = tuple([kernel_y, kernel_x, kernel_channels, kernel_filters])

        self.output_shape = self.compute_output_shape(sample_shape)

        self.filters = np.ndarray(shape=self.kernel_shape)
        filter_shape = tuple([kernel_y, kernel_x, kernel_channels])

        nl = kernel_x * kernel_y * kernel_channels
        std = math.sqrt(k / nl)

        for filter_idx in range(self.number_of_filters):
            filter = np.random.normal(scale=std, size=nl)
            filter = filter.reshape(filter_shape)
            self.filters[:, :, :, filter_idx] = filter

    def compute_output_shape(self, input_shape):
        sample_shape = input_shape[-3:]
        batch_shape = input_shape[:-3]

        input_x = sample_shape[1]
        input_y = sample_shape[0]

        kernel_x = self.kernel_size[1]
        kernel_y = self.kernel_size[0]

        stride_x = self.strides[1]
        stride_y = self.strides[0]

        padding = 0

        return (int(((input_y - kernel_y + 2 * padding) / stride_y)) + 1,
                int(((input_x - kernel_x + 2 * padding) / stride_x)) + 1,
                self.number_of_filters)

    def forward_propagate(self, A):
        # padding needs to be implemented
        # separate batches
        batch_shape = A.shape[:-3]

        # unpack sample shape
        sample_shape = A.shape[-3:]
        sample_y = sample_shape[0]
        sample_x = sample_shape[1]
        sample_channels = sample_shape[2]
        assert(sample_shape == self.input_shape)

        # unpack kernel
        kernel_y = self.kernel_size[0]
        kernel_x = self.kernel_size[1]

        # unpack stride shape
        stride_y = self.strides[0]
        stride_x = self.strides[1]

        # unpack convolution
        conv_shape = self.compute_output_shape(A.shape)
        conv_y = conv_shape[0]
        conv_x = conv_shape[1]

        # initialize the output convolution
        output_shape = batch_shape + conv_shape
        output = np.zeros(shape= output_shape)
        output.fill(-9e99)

        # begin convolution
        for batch_idx in np.ndindex(batch_shape):
            layer = A[batch_idx]
            for y_idx in range(0, conv_y):
                y_start = y_idx * stride_y
                y_end = (y_idx * stride_y + kernel_y)
                for x_idx in range(0, conv_x):
                    x_start = x_idx * stride_x
                    x_end = (x_idx * stride_x + kernel_x)

                    kernel = layer[y_start:y_end, x_start:x_end]

                    for filter_idx in range(self.number_of_filters):
                        filter = self.filters[:, :, :, filter_idx]
                        multi = np.multiply(kernel, filter)
                        product_idx = (y_idx, x_idx, filter_idx)
                        output[batch_idx + product_idx] = np.sum(multi)

        return output, None


    def backward_propagate(self, dZ, cache):
        raise NotImplementedError()

    def get_weights(self):
        return self.filters

    def set_weights(self, weights):
        self.filters = weights


class Flatten(Layer):

    def __init__(self):
        super().__init__()
        self.input_shape = None
        self.output_shape = None

    def compute_output_shape(self, sample_shape):
        return tuple([math.prod(sample_shape)])
    
    
    def build(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = self.compute_output_shape(input_shape)

    def forward_propagate(self, A):
        sample_dimensions = len(self.input_shape)
        sample_shape = A.shape[-sample_dimensions:]
        flattened_shape = tuple([math.prod(sample_shape)])
        batch_shape = A.shape[:-sample_dimensions]

        return np.reshape(A, batch_shape + flattened_shape), None

    def backward_propagate(self, dZ, cache):
        batch_shape = input_shape[:-3]
        return np.reshape(dZ, batch_shape + self.input_shape)

    def get_weights(self):
        return []

    def set_weights(self, weights):
        pass


class Dense(Layer):

    def __init__(self, num_outputs):
        super().__init__()
        self.num_outputs = num_outputs
        self.num_inputs = 0
        self.weights = np.zeros(0)
        self.output_shape = tuple([num_outputs])

    def build(self, input_shape, k = 2):
        self.input_shape = input_shape
        sample_shape = input_shape[-3:]
        self.num_inputs = math.prod(sample_shape)

        self.weights = np.ndarray(shape=(self.num_outputs, self.num_inputs))
        nl = self.num_inputs
        std = math.sqrt(k / nl)
        for i in range(self.num_outputs):
            self.weights[i] = np.random.normal(scale=std, size=nl)

    def set_weights(self, weights):
        assert(len(weights.shape) == 2)
        self.num_inputs = weights.shape[1]
        self.num_outputs = weights.shape[0]
        return self.weights

    def get_weights(self):
        return self.weights

    def compute_output_shape(self, input_shape):
        return tuple([self.num_outputs])

    def forward_propagate(self, A):
        print(A.shape)
        print(self.weights.shape)

        num_input_dimensions = len(self.input_shape)
        sample_shape = A.shape[-num_input_dimensions:]

        assert(self.input_shape == sample_shape)

        batch_shape = A.shape[:-num_input_dimensions]
        output_shape = batch_shape + self.output_shape

        Z = np.zeros(shape=output_shape)
        Z.fill(-9e99)
        for batch_idx in np.ndindex(batch_shape):
            Z[batch_idx] = self.weights @ A[batch_idx]

        cache = { 'A' : A }
        return Z, cache

    def backward_propagate(self, dZ, cache):
        A = cache['A']
        dW = np.dot(self.weights.T,dZ)
        db = None
        dA = np.dot(dZ, np.transpose(A))

        assert (dA.shape == A.shape)
        assert (dW.shape == self.weights.shape)

        return dZ, dW, db
        
class Membrane:

    def __init__(self):
        pass

    def reset(self):
        pass

    def activate(self, Z):
        return 0

    def differentiate(self, dA, cache):
        return 0


class LeakyIntegrateAndFire(Membrane):

    def __init__(self, Vreset: float, Vth: float, tau_m: float, fire=True, leaky=True):
        super().__init__()
        self.Vreset = Vreset
        self.Vth = Vth
        self.tau_m = tau_m
        self.fire = fire
        self.leaky = leaky

        self.input_shape = None
        self.__num_input_dimensions = None
        self.output_shape = None

    def build(self, input_shape):
        self.input_shape = input_shape
        self.__num_input_dimensions = len(self.input_shape)
        self.output_shape = input_shape

    def neuron_activation(self, Vm):
        spike = None
        if Vm >= self.Vth and self.fire:
            spike = 1
            self.Vm = self.Vreset
        else:
            spike = 0
            if self.leaky:
                Vm = Vm * math.exp(-1 / self.tau_m)

        # TODO: 1 / t needs to be implemented
        return [Vm, spike]

    def activate(self, Vin):
        # this function can be optimised given that only the final Vm is required
        # assert (Vin.shape == Vout.shape)

        batch_shape = Vin.shape[:-self.__num_input_dimensions-1]
        spike_train_length = Vin.shape[-self.__num_input_dimensions-1]
        membrane_shape = Vin.shape[-self.__num_input_dimensions:]

        activation = batch_shape + tuple([spike_train_length]) + membrane_shape
        activation_shape = activation

        Vout = np.ndarray(shape=(activation_shape))
        Vout.fill(self.Vreset)

        Vp = np.ndarray(shape=(batch_shape + membrane_shape))
        Vp.fill(-9e99)

        spike_train = np.ndarray(shape=(activation_shape))
        spike_train.fill(0)

        t_current = None
        t_previous = None

        for batch_idx in np.ndindex(batch_shape):

            for neuron_idx in np.ndindex(membrane_shape):

                for t_idx in range(1, spike_train_length):
                    # membrane voltage for this step
                    t_current = batch_idx + tuple([t_idx]) + neuron_idx
                    t_previous = batch_idx + tuple([t_idx - 1]) + neuron_idx
                    Vm = Vin[t_current] + Vout[t_previous]

                    # simulate lif-neuron
                    [Vout[t_current], spike_train[t_current]] = self.neuron_activation(Vm)

                # store the final membrane voltage
                Vp_idx = batch_idx + neuron_idx
                Vp[Vp_idx] = Vout[t_current]

        cache = {
            'Vp' : Vp,
            'Vout' : Vout,
            'spike_train' : spike_train
        }

        return spike_train, cache

    def __diff_LIF(self, dA, cache):
        Vp = cache['Vp']
        spike_train = cache['spike_train']

        # sum of decay over time
        gamma = sum(spike_train)
        if gamma == 0:
            return 0

        total_decay = 0
        t = tk = 1

        for activation in spike_train:

            if activation:
                if t != tk:
                    decay = math.exp(-(t - tk) / tau_m)
                    total_decay = total_decay - (1 / tau_m) * decay

                tk = t + 1
            t = t + 1

        dZ = dA * (1 / self.Vth) * (1 + (1 / gamma) * total_decay)
        return dZ

    def __diff_LI(self, dA, cache):
        dZ = (1/tau_m) * dA
        return dZ

    def __diff_IF(self, dA, cache):
        pass

    def __diff_I(self, dA, cache):
        pass

    def differentiate(self, dA, cache):
        # LIF Neuron
        Vp = cache['Vp']
        spike_train = cache['spike_train']
        dZ = None
        tau_m = len(spike_train)

        if self.leaky:
            if self.fire:
                return self.__diff_LIF()
        else:
            # LI neuron
            if self.fire:

            else:


        return dZ

    def get_output_shape(self):
        return self.output_shape


# the idea of this model is to process everything LIF
class SpikingNeuralNetwork:

    @staticmethod
    def __traverse_batch(start, end, step):
        i = start
        while i < end:
            yield i
            i += step
        yield end

    def __init__(self):
        self.__layers = []
        self.__membrane = []
        pass

    def build(self, input_shape):
        # set input shape for model
        self.input_shape = input_shape
        self.tau_m = input_shape[0]

        for layer_idx in range(0, len(self.__layers)):
            layer = self.__layers[layer_idx]
            membrane = self.__membrane[layer_idx]

            print(str(layer_idx) + ":" + str(layer))

            layer.build(input_shape=input_shape)
            input_shape = layer.compute_output_shape(input_shape)

            if membrane is not None:
                membrane.build(input_shape)

        # last layers output shape to models output shape
        self.output_shape = input_shape

    def add_layer(self, layer: Layer, activation: Membrane = None):
        self.__layers.append(layer)
        self.__membrane.append(activation)

    def forward_propagation(self, X):
        caches = []

        A = X

        for layer_idx in range(0, len(self.__layers)):
            layer = self.__layers[layer_idx]
            membrane = self.__membrane[layer_idx]

            print(layer)
            Z, linear_cache = layer.forward_propagate(A)
            print("Z: " + str(np.amax(Z)))

            if membrane is not None:
                print(membrane)
                A, activation_cache = membrane.activate(Z)
                print("A: " + str(np.amax(A)))

                cache = { 'linear_cache' : linear_cache,
                          'activation_cache' : activation_cache }

                caches.append({ 'A': A,
                                'Z': Z,
                                'cache': cache})

            else:
                print("Z: " + str(np.amax(Z)))
                A = Z

                cache = { 'linear_cache' : linear_cache,
                          'activation_cache' : None }

                caches.append({ 'A': None,
                                'Z': Z,
                                'cache': cache})



        return A, caches

    def compute_cost(self, A, Y):
        return 0.5 * np.sum(np.power((A - Y), 2))

    def compute_loss(self, A, Y):
        return np.mean(np.square(Y - A), axis=-2)


    def backward_propagation(self, AL, caches, Y):
        grads = []
        L = len(self.__layers)
        m = AL.shape[1] ## figure this out

        # gradients
        dZ, dW, db = (None, None, None)

        # derivative of activation in final layer
        dAL = [self.compute_loss(AL, Y)]
        grad = [
            {
                "dZ": None,
                "dA": dAL,
                "dW": None,
                "db": None
            }
        ]
        grads.insert(0, grad)

        # backwards propagating the loss
        for layer_idx in range(L-1, 0, -1):
            layer = self.__layers[layer_idx]
            A, Z, cache = (caches[layer_idx]['A'], caches[layer_idx]['Z'], caches[layer_idx]['cache'])
            linear_cache, activation_cache = (cache['linear_cache'], cache['activation_cache'])
            membrane = self.__membrane[layer_idx]

            if membrane is not None:
                dZ = membrane.differentiate(dAL, activation_cache)
                dAL, dW, db = layer.backward_propagate(dZ, linear_cache)
            else:
                dAL, dW, db = layer.backward_propagate(dAL, linear_cache)

            grad = [
                {
                    "dZ":dZ,
                    "dA":dAL,
                    "dW":dW,
                    "db":db
                }
            ]
            grads.insert(0, grad)

        return grads

    def fit(self, X=None, Y=None, epochs=1, batch_size=None, learning_rate=0.002):
        # batch_size + (time, height, width, channel)
        num_input_dimensions = len(self.input_shape)
        num_output_dimensions = len(self.output_shape)

        batch_shape = X.shape[:-num_input_dimensions]
        batch_ndim = len(batch_shape)
        num_samples = math.prod(batch_shape)

        sample_shape = X.shape[-num_input_dimensions:]
        sample_label_shape = Y.shape[-batch_ndim:]

        assert(sample_label_shape == self.output_shape)

        batch_samples = np.zeros(shape=tuple([batch_size]) + sample_shape)
        batch_samples_labels = np.zeros(shape=tuple([batch_size]) + sample_label_shape)

        # output from the opperation
        output = np.zeros(shape=batch_shape+Y.shape)

        # run the training data an epochs number of times
        for epoch in range(epochs):

            # start processing and updating the network according to the batch size
            for train_start in SpikingNeuralNetwork.__traverse_batch(0, num_samples-batch_size, batch_size):

                # get the end index
                train_end = min(train_start+batch_size, num_samples)

                # prevent over indexing at the end of the array
                number_of_training_samples = train_end - train_start

                # can this be optimized
                batch_indices = []
                for train in range(number_of_training_samples):
                    batch_idx = np.unravel_index(train_start + train, batch_shape)
                    print(batch_idx)
                    batch_samples[batch_idx] = X[batch_idx]
                    batch_samples_labels[batch_idx] = Y[batch_idx]

                batch_outputs, batch_cache = self.forward_propagation(batch_samples)
                final_cache = batch_cache[len(batch_cache)-1]
                cache = final_cache['cache']
                activation_cache = cache['activation_cache']

                Vp = activation_cache['Vp']
                costs = self.compute_cost(Vp, batch_samples_labels)
                loss = self.compute_loss(Vp, batch_samples_labels)

                # this needs to be fixed
                AL = Vp
                grads = self.backward_propagation(AL, batch_cache, batch_samples_labels)
                parameters = self.update_parameters(batch_cache, grads, learning_rate)
                # select batch images


                batch_labels = np.take(Y, batch_indices)




                # batch_spike_train = Model.__generate_model_spike_train(batch_size, train_labels[0])

                # generate batch activation outputs
                # batch_outputs = self.__generate_batch_outputs(batch_size, train_labels[0])

                # generate batch cache
                # batch_cache = self.__generate_batch_cache(batch_size)

                # generate batch gradients
                # batch_gradients = self.__generate_batch_gradients(batch_size)

                # costs
                # costs = np.zeros(batch_size)



                # run batch # potentially multi threaded
                # for i in range(0, batch_size):
                    # select sample from batch
                    #train_image = batch_images[i]
                    # train_label = batch_labels[i]

                    # convert to input to spike train
                    # layer_spike_train = generate_layer_spike_train(train_image, self.spike_train_length)

                    # propagate through network
                    # batch_outputs[i], batch_cache[i] = self.forward_propagation(train_image)
                    # dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
                    # dA_prev_temp, dW_temp, db_temp

                    # calculate the cost
                    # costs[i] = self.compute_cost(Y, train_label)

                    # backwards propagate to calculate the gradients in the network
                    # grads = Model.backward_propagation(model, batch_outputs[i], Y, batch_cache[i])

                    # batch_end_idx = batch_start_idx

                # update the network using the gradients from the batch
                # parameters = Model.update_parameters(model, batch_cache, batch_gradients, learning_rate)

    def predict(self, X):
        pass




def test_case_dropout():
    probability = 0.2
    layer = Dropout(probability)
    dim = 10000
    test_shape = (dim,dim)
    layer.build(input_shape=test_shape)
    test_input = np.ones(test_shape)
    expected = math.prod(test_shape) * (1 - probability)
    print("expected: " + str(expected))

    received_output, cache = layer.forward_propagate(test_input)
    sum_received_output = np.sum(received_output)
    print("actual: " + str(sum_received_output))


test_case_dropout()


tau_m = 10
dropout_rate = 0.2

LeNet5 = SpikingNeuralNetwork()

LeNet5.add_layer(Convolution2D(kernel_size=(5,5), strides=(1,1), number_of_filters=20), LeakyIntegrateAndFire(0, 1, tau_m, fire=True, leaky=True))
LeNet5.add_layer(Dropout(probability=dropout_rate))
LeNet5.add_layer(AveragePool2D(kernel_size=(2,2),strides=(2,2)), LeakyIntegrateAndFire(0, 0.75, tau_m, fire=True, leaky=True))
LeNet5.add_layer(Dropout(probability=dropout_rate))

LeNet5.add_layer(Convolution2D(kernel_size=(5,5), strides=(1,1), number_of_filters=50), LeakyIntegrateAndFire(0, 1, tau_m, fire=True, leaky=True))
LeNet5.add_layer(Dropout(probability=dropout_rate))
LeNet5.add_layer(AveragePool2D(kernel_size=(2,2),strides=(2,2)), LeakyIntegrateAndFire(0, 1, tau_m, fire=True, leaky=True))
LeNet5.add_layer(Dropout(probability=dropout_rate))

LeNet5.add_layer(Flatten())
LeNet5.add_layer(Dense(num_outputs=200), LeakyIntegrateAndFire(0, 1, tau_m, fire=True, leaky=True))
LeNet5.add_layer(Dense(num_outputs=10), LeakyIntegrateAndFire(0, 1, tau_m, fire=False, leaky=False))

input_shape = train_images[0].shape

input_images = np.array([generate_layer_spike_train(train_images[0], tau_m), generate_layer_spike_train(train_images[1], tau_m),
                         generate_layer_spike_train(train_images[2], tau_m), generate_layer_spike_train(train_images[3], tau_m),
                         generate_layer_spike_train(train_images[4], tau_m)])

LeNet5.build(input_images[0].shape)
lbl = train_labels[0:5,:]
LeNet5.fit(input_images, train_labels[0:5,:], batch_size=2)
LeNet5.predict(test_images, test_labels)

class Model:

    @staticmethod
    def __generate_batch_outputs(batch_size, train_label):
        return np.zeros(shape=(tuple([batch_size]) + train_label.shape))

    @staticmethod
    def __generate_batch_cache(model, batch_size):
        cache = []
        time = tuple([model.spike_train_length])
        for batch_idx in range(0, batch_size):
            cache.extend([{
                "l1_activation" : np.zeros(shape= time + model.__l1_output_dimensions),
                "l1_membrane": np.zeros(shape=time + model.__l1_output_dimensions),
                "l1_spike": np.zeros(shape=time + model.__l1_output_dimensions),
                "l2_activation": np.zeros(shape=time + model.__l2_output_dimensions),
                "l2_membrane": np.zeros(shape=time + model.__l2_output_dimensions),
                "l2_spike": np.zeros(shape=time + model.__l2_output_dimensions),
                "l3_activation": np.zeros(shape=time + model.__l3_output_dimensions),
                "l3_membrane": np.zeros(shape=time + model.__l3_output_dimensions),
                "l3_spike": np.zeros(shape=time + model.__l3_output_dimensions),
                "l4_activation": np.zeros(shape=time + model.__l4_output_dimensions),
                "l4_membrane": np.zeros(shape=time + model.__l4_output_dimensions),
                "l4_spike": np.zeros(shape=time + model.__l4_output_dimensions),
                "l5_output": np.zeros(shape=time + model.__l5_output_dimensions),
                "l6_activation": np.zeros(shape=time + model.__l6_output_dimensions),
                "l6_membrane": np.zeros(shape=time + model.__l6_output_dimensions),
                "l6_spike": np.zeros(shape=time + model.__l6_output_dimensions),
                "l7_activation": np.zeros(shape=time + model.__l7_output_dimensions),
                "l7_membrane": np.zeros(shape=time + model.__l7_output_dimensions),
                "l7_spike": np.zeros(shape=time + model.__l7_output_dimensions),
            }])
        return cache

    @staticmethod
    def __generate_batch_gradients(model, batch_size):
        gradients = []
        for i in range(batch_size):
            gradients.extend([{
                "l1_gradients": np.zeros(shape=model.__l1_filters.shape),
                "l3_gradients": np.zeros(shape=model.__l1_filters.shape),
                "l6_gradients": np.zeros(shape=model.__l6_weights.shape),
                "l7_gradients": np.zeros(shape=model.__l7_weights.shape)
            }])
        return gradients


    def __init__(self, spike_train_length):
        """

        """

        self.spike_train_length = spike_train_length

        """Convoloution 1"""
        self.input_channels = 1
        self.input_y = 28
        self.input_x = 28
        self.__l1_num_channels = self.input_channels
        self.__l1_input_dim = (self.input_y, self.input_x, self.input_channels)
        self.__l1_num_filters = 20
        self.__l1_kernel_dimensions = (5, 5, self.__l1_num_channels, self.__l1_num_filters)
        self.__l1_stride_dimensions = (1, 1)
        self.__l1_padding_dimensions = 0
        self.__l1_output_dimensions = conv_output_size(self.__l1_input_dim,
                                                       self.__l1_kernel_dimensions,
                                                       self.__l1_stride_dimensions,
                                                       self.__l1_padding_dimensions)
        self.__l1_filters = generate_conv2d_filters(self.__l1_kernel_dimensions)
        self.__l1_membrane = generate_membrane(self.__l1_output_dimensions)

        """Average Pool 1"""
        self.__l2_input_dim = self.__l1_output_dimensions
        self.__l2_kernel_dimensions = (2, 2, 1, self.__l1_output_dimensions[2])
        self.__l2_stride_dimensions = (2, 2)
        self.__l2_padding_dimensions = 0
        self.__l2_output_dimensions = conv_output_size(self.__l2_input_dim,
                                                       self.__l2_kernel_dimensions,
                                                       self.__l2_stride_dimensions,
                                                       self.__l2_padding_dimensions)
        self.__l2_membrane = generate_membrane(self.__l2_output_dimensions)

        """Convoloution 2"""
        self.__l3_input_dim = self.__l2_output_dimensions
        self.__l3_num_channels = 50
        self.__l3_kernel_dimensions = (5, 5, self.__l3_input_dim[2], self.__l3_num_channels)
        self.__l3_stride_dimensions = (1, 1)
        self.__l3_padding_dimensions = 0
        self.__l3_output_dimensions = conv_output_size(self.__l3_input_dim,
                                                       self.__l3_kernel_dimensions,
                                                       self.__l3_stride_dimensions,
                                                       self.__l3_padding_dimensions)
        self.__l3_filters = generate_conv2d_filters(self.__l3_kernel_dimensions)
        self.__l3_membrane = generate_membrane(self.__l3_output_dimensions)

        """Average Pool 2"""
        self.__l4_input_dim = self.__l3_output_dimensions
        self.__l4_kernel_dimensions = (2, 2, 1, self.__l3_output_dimensions[2])
        self.__l4_stride_dimensions = (2, 2)
        self.__l4_padding_dimensions = 0

        self.__l4_output_dimensions = conv_output_size(self.__l4_input_dim,
                                                       self.__l4_kernel_dimensions,
                                                       self.__l4_stride_dimensions,
                                                       self.__l4_padding_dimensions)
        self.__l4_membrane = generate_membrane(self.__l4_output_dimensions)

        """Flatten"""
        self.__l5_input_dimensions = self.__l4_output_dimensions
        self.__l5_output_dimensions = tuple([math.prod(self.__l5_input_dimensions)])

        """Dense Layer 1"""
        self.__l6_input_dim = self.__l5_output_dimensions
        self.__l6_neurons = 200
        self.__l6_output_dimensions = tuple([self.__l6_neurons])
        self.__l6_weights = generate_dense_layer_weights(self.__l5_output_dimensions, self.__l6_neurons)
        self.__l6_membrane = generate_membrane(self.__l6_output_dimensions)

        """Dense Layer 2"""
        self.__l7_input_dim = self.__l5_output_dimensions
        self.__l7_neurons = 10
        self.__l7_output_dimensions = tuple([self.__l7_neurons])
        self.__l7_weights = generate_dense_layer_weights(self.__l6_output_dimensions, self.__l7_neurons)
        self.__l7_membrane = generate_membrane(self.__l7_output_dimensions)

        self.reset()

    @staticmethod
    def forward_propagation(model, input, cache, output):

        ## TODO: reset membrane
        ## TODO: fix conv2d, avgpool, lif and dense to not require input/output

        cache["l1_activation"] = conv2d(input, cache["l1_activation"], model.__l1_filters,  stride=model.__l1_stride_dimensions)
        [cache["l1_membrane"], cache["l1_spike"]] = lif_neuron_pool(cache["l1_activation"], cache["l1_membrane"], cache["l1_spike"], Vth=1, fire=True, leaky=True)

        cache["l2_activation"] = avg_pool(cache["l1_spike"], cache["l2_activation"], kernel_size=model.__l2_kernel_dimensions, stride=model.__l2_stride_dimensions)
        [cache["l2_membrane"], cache["l2_spike"]] = lif_neuron_pool(cache["l2_activation"], cache["l2_membrane"], cache["l2_spike"], Vth=0.75, fire=True, leaky=False)

        conv2d(cache["l2_spike"], cache["l3_activation"], model.__l3_filters, stride=model.__l3_stride_dimensions)
        [cache["l3_membrane"], cache["l3_spike"]] = lif_neuron_pool(cache["l3_activation"], cache["l3_membrane"], cache["l3_spike"], Vth=1.0, fire=True, leaky=False)

        cache["l4_activation"] = avg_pool(cache["l3_spike"], cache["l4_activation"], kernel_size=model.__l4_kernel_dimensions, stride=model.__l4_stride_dimensions)
        [cache["l4_membrane"], cache["l4_spike"]] = lif_neuron_pool(cache["l4_activation"], cache["l4_membrane"], cache["l4_spike"], Vth=0.75, leaky=False)

        cache['l5_output'] = flatten(cache["l4_spike"], cache['l5_output'], 1)

        cache["l6_activation"] = dense_forward(cache['l5_output'], cache["l6_activation"], model.__l6_weights)
        [cache["l6_membrane"], cache["l6_spike"]] = lif_neuron_pool(cache["l6_activation"], cache["l6_membrane"], cache["l6_spike"], Vth=1, fire=True, leaky=True, time_index=0)

        cache["l7_activation"] = dense_forward(cache["l6_spike"], cache["l7_activation"], model.__l7_weights)
        [cache["l7_membrane"], cache["l7_spike"]] = lif_neuron_pool(cache["l7_activation"], cache["l7_membrane"], cache["l7_spike"], Vth=1, fire=False, leaky=False, time_index=0)

        output = np.divide(cache["l7_membrane"][-1], model.spike_train_length)  # this may be problematic

        return [output, cache]

    @staticmethod
    def generate_grads():
        pass

    @staticmethod
    def lif_backward_propagation(input_error_gradient, weights, layer_spike_train, Vth=1):
        # activation derivative
        neurons = layer_spike_train.shape[1]
        time = layer_spike_train.shape[0]

        a_lif_derivative = np.zeros((neurons))

        spike_train = []

        layer_spike_train[1][0] = 1
        layer_spike_train[5][0] = 1
        layer_spike_train[10][0] = 1
        layer_spike_train[15][0] = 1
        layer_spike_train[18][0] = 1

        for neuron_idx in range(0, neurons):
            spike_train = layer_spike_train[:,neuron_idx]
            a_lif_derivative[neuron_idx] = differentiate_spike_train(spike_train, Vth)

        output_neurons = weights.shape[0]
        input_neurons = weights.shape[1]

        # (weights * error gradient)
        for output_neuron_idx in range(0, output_neurons):
            error = 0
            for input_neuron_idx in range(0, input_neurons):
                ax = loss[input_neuron_idx]                         # <-- loss from this neuron
                wx = weights[input_neuron_idx][output_neuron_idx]   # <-- its connected weights
                error = error + ax * wx

        # (weights * error gradient) . alif_dw/dt
        a_lif_derivative[neuron_idx]

    @staticmethod
    def backward_propagation(model, output: np.array, label: np.array, cache: dict):
        grads = [{"dW7" : np.zeros(shape=model.__l7_weights.shape)},
                 {"dW6" : np.zeros(shape=model.__l7_weights.shape)},
                 {"dW3" : np.zeros(shape=model.__l7_weights.shape)},
                 {"dW1" : np.zeros(shape=model.__l7_weights.shape)}]

        tau_m = len(cache["l7_spike"])

        #d (a_lif / d_net) = (1 / Vth + e) = (1/Vth) * (1 + 1/y )

        # (dE / da_lif) * (da_lif / dnet) * (dnet / dwl)
        # (da_lif / dnet) -> differentiate_spike_train
        #
        # dA1, dW2

        # final layer error gradient
        loss = (output - label) / tau_m

        model.lif_backward_propagation(loss, model.__l7_weights, cache["l7_spike"])

        # LIF error
        # spike layer derivative
        a7_lif = cache["l7_spike"]


        w = model.__l7_weights
        for neuron_idx in range(len(w)):
            weights = w[neuron_idx]
            error_in_neuron = loss[neuron_idx]
            for weight_idx in range(len(weights)):
                grad = weights[neuron_idx] * error_in_neuron
                grads["dW7"][neuron_idx][weight_idx] = grad



        # dW -> dZ @ np.transpose(A_prev)
        # dA_prev -> np.transpose(W) @ dZ
        # dZ -> dA[L] * g(Z[L])

        # a  =
        # grad_l = (w^(l)*dl^(l+1) * grad^(l+1)).a_lif(net^l)

        # m = cache["l7_membrane"]
        # s = cache["l7_spike"]
        # b = [differentiate_spike_train(train) for train in s]
        # a = cache["l7_activation"]
        # w = model.__l7_weights

        return grads

    @staticmethod
    def update_parameters(batch_cache, batch_gradients, learning_rate):
        pass

    def fit(self, train_images, train_labels, epochs, batch_size, learning_rate = 0.002):
        pass

    def predict(self, test_images, test_labels):
        pass

    def reset(self):
        self.__l1_filters = generate_conv2d_filters(self.__l1_kernel_dimensions)
        self.__l1_membrane = generate_membrane(self.__l1_output_dimensions)
        self.__l2_membrane = generate_membrane(self.__l2_output_dimensions)

        self.__l3_filters = generate_conv2d_filters(self.__l3_kernel_dimensions)
        self.__l3_membrane = generate_membrane(self.__l3_output_dimensions)
        self.__l4_membrane = generate_membrane(self.__l4_output_dimensions)

        self.__l6_weights = generate_dense_layer_weights(self.__l5_output_dimensions, self.__l6_neurons)
        self.__l6_membrane = generate_membrane(self.__l6_output_dimensions)

        self.__l7_weights = generate_dense_layer_weights(self.__l6_output_dimensions, self.__l7_neurons)
        self.__l7_membrane = generate_membrane(self.__l7_output_dimensions)



model = Model(spike_train_length = 20)
model.fit(train_images, train_labels, batch_size=2, epochs=10)



"""

## (1) Gaussian random distribution of zero-mean and standard deviation of p κ nl (n l : number of fan-in synapses) as introduced in [16]

## (2) Drop out

## Test all componentry
    -> LIF Neuron
    -> Conv2d (done)
    -> Average2d (done)
    -> Gaussian Zero Mean ()
    -> Forward Propergation

## convert weights back to flattened

import tensorflow as tf

x = np.ndarray(shape=(1, 28, 28, 1))
x.fill(1)
x = tf.constant(x, dtype=tf.float32)
kernel = np.ndarray(shape=(5, 5, 1, 20))
kernel.fill(1)
kernel = tf.constant(kernel, dtype=tf.float32)

x = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
print(x.shape)
x = tf.nn.avg_pool(x, ksize=(2,2), strides=[1, 2, 2, 1], padding='VALID')
print(x.shape)

kernel = np.ndarray(shape=(5,5,20,50))
kernel.fill(1)
kernel = tf.constant(kernel, dtype=tf.float32)
x = tf.nn.conv2d(x, kernel, strides=[1, 1, 1, 1], padding='VALID')
print(x.shape)
x = tf.nn.avg_pool(x, ksize=(2,2), strides=(1,2,2,1), padding='VALID')
print(x)


y = np.ndarray(shape=(1, 28, 28, 1))
y.fill(1)
kernel = np.ndarray(shape=(5,5,1,20))
kernel.fill(1)

kernel = np.ndarray(shape=(5,5,20,50))
kernel.fill(1)
print(y)





activation = lif_neuron


layer = np.array([[  [[0.0], [0.0], [1.0], [0.0]],
                    [[0.0], [1.0], [0.0], [1.0]],
                    [[0.0], [1.0], [0.0], [0.0]],
                    [[0.0], [0.0], [1.0], [0.0]]
                 ]])

## contains two filters
filter = np.array([

                            [[[ 0.1,-0.1]],[[ 0.3, 0.2]],[[ 0.8,-0.6]]],
                            [[[ 0.5, 0.0]],[[-0.3,-0.3]],[[ 0.0, 0.0]]],
                            [[[-0.6, 0.8]],[[ 0.2, 0.3]],[[-0.1, 0.1]]]

                    ])

membrane_potential = np.array(
                [
                    [ 0.0, 0.0],
                    [ 0.0, 0.0]
                ]
)

#X = lif_neuron_pool(layer1b[0], membrane_potential1a[0], V_reset=0, V_th=0.75, tau_m=100, fire=True)

#Vm = [X[0]]
#spike = [X[1]]

#print(Vm)
#print(spike)

### notably, there could be an error here

input_t1 = np.array([[0, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0]])

input_t2 = np.array([[0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [1, 1, 1, 0],
                     [0, 1, 0, 0]])

input_t3 = np.array([[0, 0, 1, 0],
                     [0, 1, 0, 1],
                     [0, 1, 1, 0],
                     [0, 0, 0, 0]])



print(conv2d(input_t1, filter))

Vm0 = np.zeros((2, 2))
Vm_in1 = conv2d(input_t1, filter)
Vm_in2 = conv2d(input_t2, filter)
Vm_in3 = conv2d(input_t3, filter)

[Vm1, spike] = lif_neuron_pool(Vm_in1[0], Vm0, V_reset=0, V_th=1, tau_m=100, fire=True)
print("Vm, t = 1")
print(Vm1)
print("Spike, t = 1")
print(spike)

[Vm2, spike] = lif_neuron_pool(Vm_in2[0], Vm1, V_reset=0, V_th=1, tau_m=100, fire=True)
print("Vm, t = 2")
print(Vm2)
print("Spike, t = 2")
print(spike)

[Vm3, spike] = lif_neuron_pool(Vm_in3[0], Vm2, V_reset=0, V_th=1, tau_m=100, fire=True)
print("Vm, t = 3")
print(Vm3)
print("Spike, t = 3")
print(spike)


"""
