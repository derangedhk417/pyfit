#!/bin/bash
export PATH="$PATH:$PWD/src"
echo "export PATH=\"\$PATH:$PWD/src\"" >> ~/.bashrc
chmod +x $PWD/src/pyfit.py
echo "Remove the export line referencing $PWD from your .bashrc to reverse the installation procedure."
echo "Done"