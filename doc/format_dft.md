# DFT Input File Format Documentation


pyfit expects the DFT data that it uses to be in a single file that is essentially a sequential list of poscar files. 

## Format

Each atomic structure for which you have calculated a DFT energy should have one entry in this file. Each entry has the following format. The code that parses this file is [here](https://github.com/derangedhk417/pyfit/blob/9fd3577d6c49e981ca98bdd888a14803c8130148/src/poscar.py#L28) and [here](https://github.com/derangedhk417/pyfit/blob/9fd3577d6c49e981ca98bdd888a14803c8130148/src/poscar.py#L118)

### Line 1
This can technically be anything, but you should fill it with a string of characters that identifies the structure in some way. Some of the tools that come with pyfit treat this as an identifier for the structural subgroup that the structure is a part of. For example, the silicon dataset used by this research group calls all structures that are unpterturbed diamond cubic silicon `Si_B1`. Example:

```
Si_B1
```

### Line 2
The scale factor for all vectors in the structure. This is a floating point value. Typical example:
```
1.0
```

### Lines 3, 4 and 5
These are the lattice vectors for the structure. Typical example:
```
2.1874856000 2.1874856000 0.0000000000 
0.0000000000 2.1874856000 2.1874856000 
2.1874856000 0.0000000000 2.1874856000 
```

### Line 6
The number of atoms in the structure. Typical example:
```
2
```

### Line 7
This line specifies whether the coordinates are cartesian or direct (scaled). Only the first letter matters. `c = cartesian`, `d = direct (scaled)`.

Typical example:
```
carestian or direct (scaled), only the first letter matters
```

### Lines 8 to second to last line
All lines from line 8 until the second to last line should be coordinates of the atoms in the structure, often called the basis vectors. Typical example: 

```
0.0000000000 0.0000000000 0.0000000000
1.0937428000 1.0937428000 1.0937428000
```
You should have a number of entries corresponding to the number of atoms in the structure.

### Last line
This is the energy of the structure in eV, as predicted by DFT.

## Final Notes
You should have one entry per structure, conforming to the specification above. There should be no empty lines between structures. Here is an example file. This file contains four structures.

```poscar
Si_B1
1.000000
2.1874856000 2.1874856000 0.0000000000 
0.0000000000 2.1874856000 2.1874856000 
2.1874856000 0.0000000000 2.1874856000 
2
carestian or direct (scaled), only the first letter matters
0.0000000000 0.0000000000 0.0000000000
1.0937428000 1.0937428000 1.0937428000
-2.880543
Si_B1
1.000000
2.4637114602 2.4637114602 0.0000000000 
0.0000000000 2.4637114602 2.4637114602 
2.4637114602 0.0000000000 2.4637114602 
2
carestian or direct (scaled), only the first letter matters
0.0000000000 0.0000000000 0.0000000000
1.2318557301 1.2318557301 1.2318557301
-9.457406
Si_B1
1.000000
2.4944032224 2.4944032224 0.0000000000 
0.0000000000 2.4944032224 2.4944032224 
2.4944032224 0.0000000000 2.4944032224 
2
carestian or direct (scaled), only the first letter matters
0.0000000000 0.0000000000 0.0000000000
1.2472016112 1.2472016112 1.2472016112
-9.795690
Si_B1
1.000000
2.5250949847 2.5250949847 0.0000000000 
0.0000000000 2.5250949847 2.5250949847 
2.5250949847 0.0000000000 2.5250949847 
2
carestian or direct (scaled), only the first letter matters
0.0000000000 0.0000000000 0.0000000000
1.2625474923 1.2625474923 1.2625474923
-10.077705
```
