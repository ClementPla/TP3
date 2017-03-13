import itertools
from fully_connected_layer import HiddenLayer
from logistic_regression import LogisticRegression


class MLP(object):
    """Multi-Layer Perceptron Class

    A multilayer perceptron is a feedforward artificial neural network model
    that has one layer or more of hidden units and nonlinear activations.
    Intermediate layers usually have as activation function tanh or the
    sigmoid function (defined here by a ``HiddenLayer`` class)  while the
    top layer is a softmax layer (defined here by a ``LogisticRegression``
    class).
    """

    def __init__(self, rng, input, n_in, fc_layers, n_out):
        """Initialize the parameters for the multilayer perceptron

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
        architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
        which the datapoints lie

        :type n_hidden: int
        :param n_hidden: number of hidden units

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
        which the labels lie

        """

        # Since we are dealing with a one hidden layer MLP, this will translate
        # into a HiddenLayer with a tanh activation function connected to the
        # LogisticRegression layer; the activation function can be replaced by
        # sigmoid or any other nonlinear function
        self.hiddenLayerList = [HiddenLayer(rng=rng,
                                            input=input,
                                            n_in=n_in,
                                            n_out=fc_layers[0][0],
                                            activation=fc_layers[0][1])]

        for index, layer in enumerate(fc_layers[1:]):
            self.hiddenLayerList.append(
                HiddenLayer(rng=rng,
                            input=self.hiddenLayerList[index].output,
                            n_in=fc_layers[index][0],
                            n_out=layer[0],
                            activation=layer[1])
            )

        # The logistic regression layer gets as input the hidden units
        # of the hidden layer
        self.logRegressionLayer = LogisticRegression(
            input=self.hiddenLayerList[-1].output,
            n_in=fc_layers[-1][0],
            n_out=n_out
        )

        # L1 norm ; one regularization option is to enforce L1 norm to
        # be small
        self.L1 = abs(self.logRegressionLayer.W).sum()
        self.L2_sqr = (self.logRegressionLayer.W ** 2).sum()

        for layer in self.hiddenLayerList:
            self.L1 += abs(layer.W).sum()
            self.L2_sqr += (layer.W ** 2).sum()


        # square of L2 norm ; one regularization option is to enforce
        # square of L2 norm to be small


        # negative log likelihood of the MLP is given by the negative
        # log likelihood of the output of the model, computed in the
        # logistic regression layer
        self.negative_log_likelihood = (
            self.logRegressionLayer.negative_log_likelihood
        )
        # same holds for the function computing the number of errors
        self.errors = self.logRegressionLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = list(itertools.chain(*[layer.params for layer in self.hiddenLayerList] ))+ self.logRegressionLayer.params
