import argparse
import lbann
import data.mnist
import lbann.contrib.args
import lbann.contrib.launcher

# ----------------------------------
# Command-line arguments
# ----------------------------------

desc = ('Train LeNet on MNIST data using LBANN.')
parser = argparse.ArgumentParser(description=desc)
lbann.contrib.args.add_scheduler_arguments(parser)
parser.add_argument(
    '--job-name', action='store', default='lbann_lenet', type=str,
    help='scheduler job name (default: lbann_lenet)')
args = parser.parse_args()

# ----------------------------------
# Construct layer graph
# ----------------------------------

# Input data
input_ = lbann.Input(target_mode='classification')
images = lbann.Identity(input_)
labels = lbann.Identity(input_)

# LeNet
x = lbann.Convolution(images,
                      num_dims = 2,
                      num_output_channels = 6,
                      num_groups = 1,
                      conv_dims_i = 3,
                      conv_strides_i = 1,
                      conv_dilations_i = 1,
                      has_bias = True)
x = lbann.Relu(x)
x = lbann.Pooling(x,
                  num_dims = 2,
                  pool_dims_i = 1,
                  pool_strides_i = 1,
                  pool_mode = "max")
x = lbann.Convolution(x,
                      num_dims = 2,
                      num_output_channels = 16,
                      num_groups = 1,
                      conv_dims_i = 3,
                      conv_strides_i = 1,
                      conv_dilations_i = 1,
                      has_bias = True)
x = lbann.Relu(x)
x = lbann.Pooling(x,
                  num_dims = 2,
                  pool_dims_i = 2,
                  pool_strides_i = 2,
                  pool_mode = "max")
x = lbann.FullyConnected(x, num_neurons = 120, has_bias = True)
x = lbann.Relu(x)
x = lbann.FullyConnected(x, num_neurons = 84, has_bias = True)
x = lbann.Relu(x)
x = lbann.FullyConnected(x, num_neurons = 10, has_bias = True)
probs = lbann.Softmax(x)

# Loss function and accuracy
loss = lbann.CrossEntropy(probs, labels)
acc = lbann.CategoricalAccuracy(probs, labels)

# ----------------------------------
# Setup experiment
# ----------------------------------

# Setup model
mini_batch_size = 64
num_epochs = 20
model = lbann.Model(num_epochs,
                    layers=lbann.traverse_layer_graph(input_),
                    objective_function=loss,
                    metrics=[lbann.Metric(acc, name='accuracy', unit='%')],
                    callbacks=[lbann.CallbackPrint(),
                               #lbann.CallbackTimer(),
                               lbann.CallbackPrintModelDescription()])

# lbann.CallbackPrintModelDescription()

# Setup optimizer
opt = lbann.SGD(learn_rate=0.01, momentum=0.9)

# Setup data reader
data_reader = data.mnist.make_data_reader()

# Setup the training algorithm
RPE = lbann.RandomPairwiseExchange
SGD = lbann.BatchedIterativeOptimizer
ES = lbann.RandomPairwiseExchange.ExchangeStrategy(
         strategy="checkpoint_binary",
         weights_names=[],
         exchange_hyperparameters=False,
         checkpoint_dir=None)
MS = lbann.MutationStrategy(strategy="replace_kernel_conv")
         #strategy="replace_activation")
metalearning = RPE(
    metric_strategies={'accuracy': RPE.MetricStrategy.HIGHER_IS_BETTER},
    exchange_strategy=ES,
    mutation_strategy=MS)
ltfb = lbann.LTFB("ltfb",
                   metalearning=metalearning,
                   local_algo=SGD("local sgd",
                                   num_iterations=4000),
                   metalearning_steps=3)

trainer = lbann.Trainer(mini_batch_size,
                        training_algo=ltfb)
# Setup trainer
#trainer = lbann.Trainer(mini_batch_size=mini_batch_size)

# ----------------------------------
# Run experiment
# ----------------------------------
lbann_args="--procs_per_trainer=2"
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
lbann.contrib.launcher.run(trainer, model, data_reader, opt,
                           lbann_args = lbann_args, nodes=1,
                           job_name=args.job_name, #batch_job=True,
                           **kwargs)
