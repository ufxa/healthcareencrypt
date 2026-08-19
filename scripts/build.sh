#!/bin/bash
# Build the paper using tectonic
cd "$(dirname "$0")/../paper"
tectonic Artigo14.tex
echo "Build complete: paper/Artigo14.pdf"
