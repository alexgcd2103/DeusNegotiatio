#!/bin/bash

# Try to find SUMO_HOME if not set
if [ -z "$SUMO_HOME" ]; then
    # Check if we are in a conda env
    if [ -n "$CONDA_PREFIX" ]; then
        POSSIBLE_HOME="$CONDA_PREFIX/share/sumo"
        if [ -d "$POSSIBLE_HOME" ]; then
            export SUMO_HOME="$POSSIBLE_HOME"
            echo "Found SUMO_HOME in Conda: $SUMO_HOME"
        fi
    fi
    
    # Check standard Homebrew path (Apple Silicon)
    if [ -z "$SUMO_HOME" ] && [ -d "/opt/homebrew/share/sumo" ]; then
        export SUMO_HOME="/opt/homebrew/share/sumo"
        echo "Found SUMO_HOME in Homebrew: $SUMO_HOME"
    fi
fi

if [ -z "$SUMO_HOME" ]; then
    echo "Error: SUMO_HOME environment variable is not set."
    echo "Please install SUMO (or wait for the installation to finish) and set SUMO_HOME."
    exit 1
fi

echo "Building SUMO Network..."
netconvert \
  --node-files=oxford_hydepark.nod.xml \
  --edge-files=oxford_hydepark.edg.xml \
  --connection-files=oxford_hydepark.con.xml \
  --tllogic-files=oxford_hydepark.tll.xml \
  --output-file=oxford_hydepark.net.xml \
  --verbose

if [ $? -eq 0 ]; then
    echo "Network built successfully: oxford_hydepark.net.xml"
else
    echo "Network build failed."
    exit 1
fi
