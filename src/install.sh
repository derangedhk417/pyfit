#!/bin/bash
export PATH="$PATH:$PWD"
echo "export PATH=\"\$PATH:$PWD\"" >> ~/.bashrc
echo "Remove the export line referencing $PWD from your .bashrc to reverse the installation procedure."
echo "Done"