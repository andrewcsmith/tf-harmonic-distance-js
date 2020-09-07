# tf-harmonic-distance-js

This is a (work-in-progress) translation of a portion of harmonic distance and
chord rationalization research into TensorFlowJS, from a previous implementation
in Python. This aims to allow the running of the harmonic distance and chord
rationalization algorithms on `npm`, for use in Max 8 and possibly the browser.

## Running on Windows 10

Make sure to set the proper environment variable, given here for PowerShell.

```
$Env:TF_FORCE_GPU_ALLOW_GROWTH = "true"
```

## Building it

Assumes you have a typescript compiler installed, as this will run tsc

```
# Install nodejs and npm on whatever's right for your system
sudo apt install nodejs npm
# Install all the packages
yarn install
# Build the typescript files to dist/
yarn build
```