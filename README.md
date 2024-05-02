# Paper Reproduction | DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Video

## Austin Phillips, Tilen Potočnik, Sebastien Van den Abeele

This repo is intended to prove the reproducibility of the paper *End-to-End CNN inference of Sparse Frame Differences in Videos* [1]*.* The initial goal was to reproduce the Pose-Estimation results implemented on a ResNet architecture that were shown in the paper, re-making the network architecture ourselves and using the authors’ implementation of the DeltaCNN layers. Due to difficulties with compatibility of existing libraries and the inaccessibility of datasets, a much simple network architecture was tested on the MNIST dataset. 

We start with a short **Introduction** to shed light on the topic of the paper. Thereafter,  in **DeltaCNN theory** the information the reader requires is explained briefly. Next, the code implementation and the related problems are shown in **DeltaCNN Framework Integration** and **ResNet DeltaCNN Implementation**. The results of reproducibility study can be found in **Results**. Finally, we end with a **Conclusion** on the paper reproduction.

## Introduction

The use of convolutional neural networks (CNN) has become the mainstream method for the processing of image and video. Applications such as object detection and human pose estimation use CNNs to perform inference. However, when it comes to video-processing, the task becomes much more computationally expensive. Video-processing is usually done by feeding each frame into the network, resulting in a batch of 60 images that needs to be processed every second in a real-time system. This requires more computational resources, more power, and more expensive hardware.

Rather than processing the entire frame, what if only the **changes** (or the **deltas**) across the frames were processed? In particular for static video capture, for example in surveillance and security video in which the camera is not moving, skipping regions of the frames that are identical and ignoring insignificant pixel updates, the amount of computational operations can be significantly reduced. In practice, it is difficult to take advantage of these sparse updates.

Since graphics processing units (GPUs) are typically used for CNN inference due to their ability to perform a **s**ingle-**i**nstruction on **m**ultiple **d**ata points (they are SIMD devices), the conditional statements currently used to skip the sparse updates are less efficient on GPUs than on CPUs and thus cannot take full advantage. The authors of the paper “**DeltaCNN: End-to-End CNN Inference of Sparse Frame Differences in Videos**” present the first fully sparse CNN, *DeltaCNN*, which achieves real speedups in practice and outperforms the state-of-the-art *cuDNN* dense inference by a factor of up to 7x.

## DeltaCNN Theory

With the paper, the authors enabled sparse end-to-end neural networks for video processing. By sparsifying the data the neural network processes, the number of computations and overhead can be reduced as unimportant bits are left out. At the time the paper was released, other existing architectures only sparsified the convolutional layers, making it not very effective as extra computations were required to switch between a sparse and dense states at different layers.

To give an overview of the forward pass of a sparse neural network, a simple model is shown in the figure below. The first frame is processed as if it where a normal neural network. This serves later on as backbone to contain the irrelevant information (e.g. the static background) that is no longer processed in the next frame. For the second frame, the difference between the first and the second frame is used as sparse input for the network. The peculiarities related to the processing of the sparse data is explained later. The output of the second frame is a combination of the dense first layer’s output and sparse output of the second frame. 

![Forward pass simple DeltaCNN [1]](https://github.com/austinjphillips/deltacnn-paper-reproduction/blob/main/deltacnn-forward-pass.png?raw=true) Forward pass simple DeltaCNN [1]

For this sparse end-to-end neural network to function effectively in video applications, there are two problems to be solved. Firstly, GPUs are the main processing units for neural networks.  Sparsifying requires more conditional control operations which are not well-performed by GPUs, increasing the processing time. To effectively use GPUs, operations should be adapted to the strengths of GPU, which are matrix operations. To do so, the authors introduced update masks and hybrid kernels. Secondly, non-linear layers require previous input data to obtain accurate intermediate sparse result. Here, buffers are introduced to keep track of this past data.

A first contribution to the solution of the first problem are the update masks. Such a mask is a 2d tensor which is propagated next to the delta values (see figure above). As a result, a next layer can know which pixels need to be updated without loading and checking the entire input to the layer. The mask acts as a filter that selects only the relevant information and thereby this operation can be executed efficiently by a GPU. 

The second contribution to the solution are the hybrid kernels. These are deployed in convolutional layers which are the most computational demanding part of the neural network. Images are stored in tiles for efficient memory use. If a pixel inside a tile is updated, the computational effort to update all the pixels compared to updating a single pixel is almost as high due to the nature of the convolutional operation. In order to reduce the required computations an hybrid kernel is introduced. Based on the number of updates in a tile, a tile is skipped completely, processed sparsely or processed densely. In the sparse processing mode, the convolutional operation occurs only on a small subset, stored in a separate array.

To enable end-to-end sparse neural networks, all types of layers need to be able to handle sparse data. Sparsification can be seen as a linear operation (a subtraction) and therefore all linear layers just remain the same. Some layers are non-linear and to make these conform to the sparse framework, every non-linear layer stores its own buffer of the previous accumulated  input (see formula below). 

$$
\delta y = f(x_{-1}+dx) -f(x_{-1})
$$

## DeltaCNN Framework Integration

We were unable to install the DeltaCNN library according to the instructions provided in the [DeltaCNN GitHub repository](https://github.com/facebookresearch/DeltaCNN). Although the module was correctly installed and added to the path, Colab could not find it when importing. We received the error:
`ModuleNotFoundError: No module named 'deltacnn’`

Rather, we propose an alternative installation approach by **using pip** to install the DeltaCNN library: `pip install /PathTo/DeltaCNN`. Using this approach, we were able to successfully import the module. The total installation time took about 9 minutes.

Now that DeltaCNN is installed, there are a few housekeeping tasks to perform before one can start modifying the standard CNN architecture to use the DeltaCNN functions. Firstly, it is important to move the network to the GPU and convert the filters into DeltaCNN format:

```python
# Try using GPU instead of CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Move network to GPU and set the network in 'channels last mode'
net.to(device, memory_format=torch.channels_last)
# Convert filters into DeltaCNN format
net.process_filters()
```

Failing to convert the filters into DeltaCNN format will result in the error:
`RuntimeError: Caught an unknown exception!`

The final step before modifying the layers of the CNN architecture is to ensure that your training and testing samples are moved to the GPU prior to feeding them into the network. This can be done using `x.to(device), y.to(device)` , otherwise the following error will arise:
`RuntimeError: input must be a CUDA tensor`

Now, one can start modifying the standard CNN architecture to employ the DeltaCNN functions. In essence, the only changes that must be made are to add a *dense-to-sparse* (`DCSparsify()`) layer at the beginning and a *sparse-to-dense* (`DCDensify()`) layer at the end. The standard CNN architecture from Assignment 3 and the modified DeltaCNN architecture can be seen below:

```python
### PyTorch
from torch import nn
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()

		self.conv1 = nn.Conv2d(...)
		self.relu1 = nn.ReLU(...)
		self.max_pool1 = nn.MaxPool2d(...)

		self.conv2 = nn.Conv2d(...)
		self.relu2 = nn.ReLU(...)
		self.max_pool2 = nn.MaxPool2d(...)

		self.fc = nn.Linear(...)

	def forward(self, x):
		# First convolutional layer
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.max_pool1(x)
		# Second convolutional layer
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.max_pool2(x)
		# Flatten feature vector
		x = x.view(x.size(0), -1)
		# Fully-connected layer
		x = self.fc(x)

		return x

### DeltaCNN
import deltacnn
class DCNet(deltacnn.DCModule):
	def __init__(self):
		super(DCNet, self).__init__()

		self.conv1 = deltann.DCConv2d(...)
		self.relu1 = deltann.DCActivation(...)
		self.max_pool1 = deltann.DCMaxPooling(...)

		self.conv2 = deltann.DCConv2d(...)
		self.relu2 = deltann.DCActivation(...)
		self.max_pool2 = deltann.DCMaxPooling(...)

		self.fc = nn.Linear(...)

		# Sparsify and densify functions
		self.sparsify = deltacnn.DCSparsify()
		self.densify = deltacnn.DCDensify()

	def forward(self, x):
		# Sparsify input <------ ADDED
		x = self.sparsify(x)
		# First convolutional layer
		x = self.conv1(x)
		x = self.relu1(x)
		x = self.max_pool1(x)
		# Second convolutional layer
		x = self.conv2(x)
		x = self.relu2(x)
		x = self.max_pool2(x)
		# Densify output <------ ADDED
		x = self.densify(x)
		# Flatten feature vector
		x = x.view(x.size(0), -1)
		# Fully-connected layer
		x = self.fc(x)

		return x
	
```

One of the final quirks in the implementation of the DeltaCNN framework was the need for equal training and testing batch sizes. Otherwise, the following error is received for a training batch size of 8 and a testing batch size 16:

> `RuntimeError: The size of tensor a (8) must match the size of tensor b (16) at non-singleton dimension 0`
> 

## ResNet DeltaCNN Implementation

Although we were not successful at getting the Pose-ResNet to work, we were still able to get a ResNet-50 architecture running with MNIST. This section discusses which errors we ran into, and which ones we were able to resolve. We will show the implementation of the DeltaCNN functions, but will not report any results obtained with it.
We used the ResNet-50 architecture from [Microsoft’s Pose-ResNet](https://github.com/microsoft/human-pose-estimation.pytorch/). The changes made follow similarly as explained above. Pay careful attention to modifying the functions of the network; in particular, the fully-connected layer remains an `nn.Linear` and one should not modify the `nn.Sequential` in the make_layer function:

```python
def make_layer(self, block, planes, blocks, stride):
	downsample = None
	if stride != 1 or self.inplanes != planes * block.expansion:
		# Do not modify the nn.Sequential!
		downsample = nn.Sequential(deltacnn.DCConv2d(...), deltacnn.DCBatchNorm2d(...))
	...
```

Furthermore, the dense-to-sparse and sparse-to-dense layers should only be added within the main feedforward loop of the ResNet class. They should not be added to the Basic and Bottleneck blocks, since the input is already sparsified when it is passed to the ResNet blocks and should be in a sparse state when leaving the ResNet blocks:

```python
class DCResNet(deltacnn.DCModule):
	def forward(self, x):
		# Sparsify the input
		x = self.sparsify(x) # <---- ADD HERE
		# ResNet layers:
		...
		# Densify the output
		x = self.densify(x) # <----- ADD HERE
		# Flatten feature vector, followed by fully-connected layer
		...

class BasicBlock(deltacnn.DCModule): # or the class Bottleneck(deltacnn.DCModule)
	def forward(self, x):
		# DO NOT SPARSIFY THE INPUT 'X', it is already sparsified
		residual = x
		...
		# DO NOT DENSIFY THE OUTPUT 'X', it needs to be in a sparse state
		return x
```

If you try to sparsify an already sparsified input, you will receive the following error:
`AttributeError: 'tuple' object has no attribute 'clone'`

Furthermore, the DeltaCNN library has a specific function to perform the residual connection within the ResNet blocks. Rather than using x = x + residual, one should use the `deltacnn.DCAdd()` otherwise the sparse delta and sparse update mask will not be respectively added properly at the residual connection.

When performing the final activation function after the residual connection in the ResNet block, we determine that one must specify the activation within the DCAdd() function. If the residual connection and activation are performed on separate lines, one will receive the following error:
`RuntimeError: CUDA error: an illegal memory access was encountered`

It is not known as to why this error arises. As an example, see the following:

```python
## SEPARATE residual connection and activation
def __init__(self)
	self.sparse_add = deltacnn.DCAdd()
	self.relu = deltacnn.DCActivation()

def forward(self, x)
	...
	# Perform residual connection
	out = self.sparse_add(out, residual)
	# Perform activation
	out = self.relu(out)
	return out

# --> Throws "RuntimeError: CUDA error: an illegal memory access was encountered"

## COLLECTIVE residual connection and activation
def __init__(self)
	self.sparse_add = deltacnn.DCAdd(activation="relu") # <--- Specify activation within function

def forward(self, x)
	...
	# Perform residual connection and activation
	out = self.sparse_add(out, residual)
	return out

# --> Does not throw errors
```

With these changes, it was possible to run the DeltaCNN ResNet-50 architecture on the MNIST dataset. The error that we ran into preventing us from further development on performing Pose Estimation using the Pose-ResNet is the following: 
`torch.jit._trace.TracingCheckError: Tracing failed sanity checks!`

We could not figure out how to resolve this error. Another error that were encountered includes:
`Warning: Kernel sizes other than 7x7, 5x5, 3x3 and 1x1 not supported. Got 3x7`
`RuntimeError: Caught an unknown exception!`

This was resolved by remembering to convert the filters into DeltaCNN format (see above).

## Results

This section will present the results and the ablation study on the MNIST dataset using a simple 6 layer convolutional neural network. The ablation study was performed on a select set of parameters, discussed below, with the intention of achieving the best performance and seeing which parameter influences the results most.

The parameters studied are:

- **Batch Size**
- **Learning Rate**
- **Weight Decay**
- **Momentum**
- **Damping**
- **Nr. of Epochs**

Each of these parameters was varied individually, to see its effects, on five runs. The table below provides the results achieved.

|  | Batch Size | LR | WD | MOM | DMP | Epoch # | Train Accuracy | Test Accuracy | Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Batch Size | 2 | 0.001 | 0.05 | 0.9 | 0.9 | 5 | 92.27 | 92.81 | 840 |
|  | 4 | 0.001 | 0.05 | 0.9 | 0.9 | 5 | 90.73 | 91.65 | 419 |
|  | 8 | 0.001 | 0.05 | 0.9 | 0.9 | 5 | 90.47 | 91.4 | 290 |
|  | 16 | 0.001 | 0.05 | 0.9 | 0.9 | 5 | 89.64 | 90.6 | 236 |
|  | 32 | 0.001 | 0.05 | 0.9 | 0.9 | 5 | 88.17 | 89.08 | 203 |
|  |  |  |  |  |  |  |  |  |  |
| LR | 8 | 0.0001 | 0.05 | 0.9 | 0.9 | 5 | 84.91 | 85.74 | 287 |
|  | 8 | 0.0005 | 0.05 | 0.9 | 0.9 | 5 | 88.15 | 89.49 | 291 |
|  | 8 | 0.001 | 0.05 | 0.9 | 0.9 | 5 | 90.47 | 91.4 | 290 |
|  | 8 | 0.0015 | 0.05 | 0.9 | 0.9 | 5 | 90.61 | 91.62 | 289 |
|  | 8 | 0.002 | 0.05 | 0.9 | 0.9 | 5 | 85.41 | 86.58 | 294 |
|  |  |  |  |  |  |  |  |  |  |
| WD | 8 | 0.0015 | 0.0001 | 0.9 | 0.9 | 5 | 92.56 | 93.41 | 293 |
|  | 8 | 0.0015 | 0.001 | 0.9 | 0.9 | 5 | 93.82 | 94.28 | 294 |
|  | 8 | 0.0015 | 0.01 | 0.9 | 0.9 | 5 | 92.8 | 93.28 | 289 |
|  | 8 | 0.0015 | 0.05 | 0.9 | 0.9 | 5 | 90.61 | 91.62 | 289 |
|  | 8 | 0.0015 | 0.1 | 0.9 | 0.9 | 5 | 88.38 | 89.29 | 289 |
|  |  |  |  |  |  |  |  |  |  |
| MOM | 8 | 0.0015 | 0.001 | 0.6 | 0.9 | 5 | 86.97 | 88.06 | 289 |
|  | 8 | 0.0015 | 0.001 | 0.7 | 0.9 | 5 | 88.54 | 89.16 | 290 |
|  | 8 | 0.0015 | 0.001 | 0.8 | 0.9 | 5 | 91.09 | 91.89 | 291 |
|  | 8 | 0.0015 | 0.001 | 0.9 | 0.9 | 5 | 93.82 | 94.28 | 294 |
|  | 8 | 0.0015 | 0.001 | 0.95 | 0.9 | 5 | 93.41 | 93.96 | 293 |
|  |  |  |  |  |  |  |  |  |  |
| DMP | 8 | 0.0015 | 0.001 | 0.9 | 0.9 | 5 | 93.82 | 94.28 | 294 |
|  | 8 | 0.0015 | 0.001 | 0.9 | 0.6 | 5 | 94.72 | 94.77 | 291 |
|  | 8 | 0.0015 | 0.001 | 0.9 | 0.4 | 5 | 95.45 | 95.63 | 291 |
|  | 8 | 0.0015 | 0.001 | 0.9 | 0.2 | 5 | 95.9 | 96.04 | 294 |
|  | 8 | 0.0015 | 0.001 | 0.9 | 0 | 5 | 95.95 | 95.9 | 291 |
|  |  |  |  |  |  |  |  |  |  |
| Epoch # | 8 | 0.0015 | 0.001 | 0.9 | 0.2 | 5 | 95.4 | 95.29 | 291 |
|  | 8 | 0.0015 | 0.001 | 0.9 | 0.2 | 10 | 95.98 | 95.87 | 582 |
|  | 8 | 0.0015 | 0.001 | 0.9 | 0.2 | 15 | 96.21 | 96.09 | 873 |
|  | 8 | 0.0015 | 0.001 | 0.9 | 0.2 | 20 | 96.23 | 95.84 | 1164 |
|  | 8 | 0.0015 | 0.001 | 0.9 | 0.2 | 25 | 96.34 | 96.05 | 1445 |

From the ablation results it can be seen that the peak performance hovers around **96.2%** on the train set accuracy and **96%** on the test set accuracy. This performance when compared to the regular CNN implementation showed worse results. The comparison is seen in the table below.

|  | DeltaCNN | Regular CNN |
| --- | --- | --- |
| Train Accuracy | 96.21 | 98.99 |
| Test Accuracy | 96.09 | 98.63 |
| Time | 873 | 797 |

It can be seen that in addition to achieving a lower accuracy it also took longer to execute. While DeltaCNN should be able to decrease the compute time of a neural network, this is not the case here. It can be concluded that while it does perform relatively well, DeltaCNN is not designed to be used on such an architecture.

### Ablation Analysis

The results of the ablation study are provided in the table above. This section will explore the effect each of the parameters has on the accuracy. The initial values were set to be the typical values for the parameters of the SGD, presented in the first run of the study, with the exception of batch size and epoch number, which were chosen arbitrarily. The values that affect the performance the least are the dampening and number of epochs. From the number of epochs we can see that, as the network is relatively shallow, that it reaches a plateau in performance at 15 epochs, therefore training for more than that is not necessary.

The batch size, while not affecting the performance significantly, has a great impact on the time it takes to train the network. With increasing the batch size, the accuracy dropped as well as the time required. A compromise between the time and accuracy was selected and a batch size of 8 deemed optimal.

Learning rate, weight decay and momentum are the variables most affecting the result. The outlier within them is the lowest weight decay. It performs worse than that of one order higher, however that may be due to the limited epoch number, not enabling the weights to reach their final values. 

## Discussion and Conclusion

This project was started with the goal of reproducing the results of the original DeltaCNN paper, and performing an ablation study to try and improve the performance achieved. This has proven to not be within the scope of the timeline, namely due to the numerous errors that arose with the implementation. The authors of the paper do provide a GitHub repository of the DeltaCNN, referring to the paper, however there is no implementation provided. The aforementioned lack of implementation, has pushed us to implement the library on the Microsoft Human Pose Estimation framework, as done in the paper, with little references. This led us down a rabbit hole of errors, which were not able to be resolves, as was the case with other teams working with the same paper. This led to a redirection of the project goals mid-way.

The implementation of the DeltaCNN layers on the MNIST dataset yielded acceptable results, peaking at 96% accuracy. This success was however overshadowed by an accuracy of nearly 99%, with a faster training time, of an identical network using pytorch functions. The results of the comparison were to be expected as DeltaCNN thrives in sequential datasets (eg. camera feed) and the implementation on MNIST was far from its intended purpose. 

In conclusion, the implementation of the DeltaCNN layers was deemed out of scope of the project, due to a lack of documentation and support. The library was successfully implemented on a simple convolutional neural network with the MNIST dataset, which showed that the DeltaCNN layers do not offer any improvement over conventional layers, when used for this purpose.

### References

1.Parger, M., Tang, C., Twigg, C. D., Keskin, C., Wang, R., & Steinberger, M. (2022). DeltaCNN: end-to-end CNN inference of sparse frame differences in videos. In *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (* pp. 12497-12506).
