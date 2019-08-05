# Neural Network / Network Potential Format Documentation

The neural network files used by pyfit are part of a custom format used by the research group that develops and maintains this code. They are often referred to as network potential files, because they can act as standalone interatomic potentials. 

## General Format

Network potential files are plaintext, with no particular encoding enforced. All numeric values in the files are in standard notation eg. `1.324234` or in scientific notation eg. `-8.83409436e-02`. 

## Fields

The code that parses the header to these files (lines 1 to 8) is [here](https://github.com/derangedhk417/pyfit/blob/1302d3d2ef8edbb3a60bc828328f925410a60e84/src/config.py#L21)

### Line 1
You will almost always want line one to be the following:
```
1 0.5 1
```

- The first cell is an integer, indicating the mode of the structure parameter calculations. It is maintained for backwards compatibililty, treat it as reserved.  
- The second cell is a shift value applied to all values before their logarithm is taken. You probably shouldn't change it.
- This controls whether or not the activation function for the neural network shifts all values down by 0.5 before applying the sigmoid function. `1 = shifted`, `0 = unshifted`. In practice, this choice doesn't seem to have much effect on error. You probably shouldn't change it.  

### Line 2
This is the number of elements the potential is meant to be used with. At the time of this writing, this is always
```
1
```

### Line 3
The name of the element the potential is for, followed by its atomic mass. Example:
```
Si 28.0855000
```

### Line 4
The cells hold these values
1) Whether or not to randomize the weights and biases when loading the file (`1 = true` and `0 = false`)
2) The maximum absolute value of weights and biases when randomizing. Values will be uniformly distributed from -`this value` to +`this value`
3) The standard MD cutoff radius in Angstroms.
4) The truncations distance in Angstroms.
5) The value to use for sigma in the gaussian term of the local structure parameter calculations.

Example line:
```
0 0.1000000 4.5000000 1.0000000 1.0000000
```

### Line 5

This is the number of Legendre polynomial orders followed by the actual orders to use for the local structure parameter calculations.
Example:
```
5 0 1 2 4 6
```

This example would indicate that 5 Legendre polynomial orders should be used, with specific orders 0, 1, 2, 4 and 6.

### Line 6

These are the r_0 values (gaussian center locations) in the local structure parameter calculations. The first number indicates how many values there are. Example:
```
8 1.5 1.8571 2.2143 2.5714 2.9286 3.2857 3.6429 4.0
```

### Line 7
These are bond order potential parameters kept for compatibility with other code. At the time of this writing they are not used. You should always make line 7 the following:
```
 1 10.787010 5.237710 4.040920 1.365000 0.104528 0.979074 0.891061 0.803526
```

### Line 8
The number of layers in the neural network followed by the size of each layer. It is **important** to note that the first "layer" is not actually a layer, but the number of inputs to the first layer. Example:

```
 6 40 16 16 16 16 1
```

This example is a network with 3 hidden layers and a total of 40 inputs to the first layer. The total number of actual layers is 5. **note**, the number of inputs (in this case 40) must be the number of legendre polynomial orders times the number of r_0 values.
