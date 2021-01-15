---
jupyter:
  celltoolbar: Slideshow
  jupytext:
    cell_metadata_json: true
    formats: ipynb,md,py:percent
    notebook_metadata_filter: all
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.9.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    codemirror_mode:
      name: ipython
      version: 3
    file_extension: .py
    mimetype: text/x-python
    name: python
    nbconvert_exporter: python
    pygments_lexer: ipython3
    version: 3.9.1
  rise:
    scroll: true
    theme: black
  toc-autonumbering: true
  toc-showcode: false
  toc-showmarkdowntxt: false
---


Tensors
--------------------------------------------

Tensors are a specialized data structure that are very similar to arrays
and matrices. In PyTorch, we use tensors to encode the inputs and
outputs of a model, as well as the model’s parameters.

Tensors are similar to NumPy’s ndarrays, except that tensors can run on
GPUs or other specialized hardware to accelerate computing. If you’re familiar with ndarrays, you’ll
be right at home with the Tensor API. If not, follow along in this quick
API walkthrough.




```python jupyter={"outputs_hidden": false}
import torch
import numpy as np
```

```python
print(torch.cuda.is_available())
print(torch.cuda.device_count())
```

```python
%run -i 'plotting.py'
```

Tensor Initialization
~~~~~~~~~~~~~~~~~~~~~

Tensors can be initialized in various ways. Take a look at the following examples:

**Directly from data**

Tensors can be created directly from data. The data type is automatically inferred.



```python jupyter={"outputs_hidden": false}
data = [[1, 2],[3, 4]]
x_data = torch.tensor(data)
```

```python
x_data
```

```python
type(data)
```

```python
type(x_data)
```

**From a NumPy array**

Tensors can be created from NumPy arrays (and vice versa - see `bridge-to-np-label`).



```python jupyter={"outputs_hidden": false}
np_array = np.array(data)
x_np = torch.from_numpy(np_array)
```

```python
type(np_array)
```

```python
type(x_np)
```

**From another tensor:**

The new tensor retains the properties (shape, datatype) of the argument tensor, unless explicitly overridden.



```python jupyter={"outputs_hidden": false}
x_ones = torch.ones_like(x_data) # retains the properties of x_data
print(f"Ones Tensor: \n {x_ones} \n")

x_rand = torch.rand_like(x_data, dtype=torch.float) # overrides the datatype of x_data
print(f"Random Tensor: \n {x_rand} \n")
```

```python
x_ones.dtype
```

```python
type(x_rand)
```

**With random or constant values:**

``shape`` is a tuple of tensor dimensions. In the functions below, it determines the dimensionality of the output tensor.



```python jupyter={"outputs_hidden": false}
shape = (2,3,)
rand_tensor = torch.rand(shape)
ones_tensor = torch.ones(shape)
zeros_tensor = torch.zeros(shape)

print(f"Random Tensor: \n {rand_tensor} \n")
print(f"Ones Tensor: \n {ones_tensor} \n")
print(f"Zeros Tensor: \n {zeros_tensor}")
```

--------------





Tensor Attributes
~~~~~~~~~~~~~~~~~

Tensor attributes describe their shape, datatype, and the device on which they are stored.



```python jupyter={"outputs_hidden": false}
tensor = torch.rand(3,4)

print(f"Shape of tensor: {tensor.shape}")
print(f"Datatype of tensor: {tensor.dtype}")
print(f"Device tensor is stored on: {tensor.device}")
```

--------------





Tensor Operations
~~~~~~~~~~~~~~~~~

Over 100 tensor operations, including transposing, indexing, slicing,
mathematical operations, linear algebra, random sampling, and more are
comprehensively described
`here <https://pytorch.org/docs/stable/torch.html>`__.

Each of them can be run on the GPU (at typically higher speeds than on a
CPU). If you’re using Colab, allocate a GPU by going to Edit > Notebook
Settings.




```python jupyter={"outputs_hidden": false}
# We move our tensor to the GPU if available
if torch.cuda.is_available():
  tensor = tensor.to('cuda')
```

```python
print(f"Device tensor is stored on: {tensor.device}")
```

```python
tensor.dtype
```

```python
type(tensor)
```

Try out some of the operations from the list.
If you're familiar with the NumPy API, you'll find the Tensor API a breeze to use.





**Standard numpy-like indexing and slicing:**



```python jupyter={"outputs_hidden": false}
tensor = torch.ones(4, 4)
tensor[:,1] = 0
print(tensor)
```

```python
tensor.device
```

```python
tensor = tensor.to('cuda')
tensor.device
```

**Joining tensors** You can use ``torch.cat`` to concatenate a sequence of tensors along a given dimension.
See also `torch.stack <https://pytorch.org/docs/stable/generated/torch.stack.html>`__,
another tensor joining op that is subtly different from ``torch.cat``.



```python jupyter={"outputs_hidden": false}
t1 = torch.cat([tensor, tensor, tensor], dim=1)
print(t1)
```

```python
t1 = torch.cat([tensor, tensor, tensor], dim=0)
print(t1)
```

**Multiplying tensors**



```python jupyter={"outputs_hidden": false}
# This computes the element-wise product
print(f"tensor.mul(tensor) \n {tensor.mul(tensor)} \n")
# Alternative syntax:
print(f"tensor * tensor \n {tensor * tensor}")
```

This computes the matrix multiplication between two tensors



```python jupyter={"outputs_hidden": false}
print(f"tensor.matmul(tensor.T) \n {tensor.matmul(tensor.T)} \n")
# Alternative syntax:
print(f"tensor @ tensor.T \n {tensor @ tensor.T}")
```

**In-place operations**
Operations that have a ``_`` suffix are in-place. For example: ``x.copy_(y)``, ``x.t_()``, will change ``x``.



```python jupyter={"outputs_hidden": false}
print(tensor, "\n")
tensor.add_(5)
print(tensor)
```

<div class="alert alert-info"><h4>Note</h4><p>In-place operations save some memory, but can be problematic when computing derivatives because of an immediate loss
     of history. Hence, their use is discouraged.</p></div>




--------------






Bridge with NumPy
~~~~~~~~~~~~~~~~~
Tensors on the CPU and NumPy arrays can share their underlying memory
locations, and changing one will change	the other.




Tensor to NumPy array
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



```python jupyter={"outputs_hidden": false}
t = torch.ones(5)
print(f"t: {t}")
n = t.numpy()
print(f"n: {n}")
```

A change in the tensor reflects in the NumPy array.



```python jupyter={"outputs_hidden": false}
t.add_(1)
print(f"t: {t}")
print(f"n: {n}")
```

NumPy array to Tensor
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^



```python jupyter={"outputs_hidden": false}
n = np.ones(5)
t = torch.from_numpy(n)
```

Changes in the NumPy array reflects in the tensor.



```python jupyter={"outputs_hidden": false}
np.add(n, 1, out=n)
print(f"t: {t}")
print(f"n: {n}")
```

```python

```
