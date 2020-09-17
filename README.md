
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
npm install -g ts-node typescript
# Install nodejs and npm on whatever's right for your system
sudo apt install nodejs npm
# Install yarn for packaging
sudo npm install -g yarn
# Install all the packages
yarn install
# Build the typescript files to dist/
yarn build
```

## Stress-testing

There are lots of instances where you might get an OOM error, so I've included a
stress_test.ts script to help identify the choke-points. If you get an error,
please open an issue (including the params you called `new Minimizer` with) so
that I can implement batch-processing where possible.

To run with typescript:

```
node -r ts-node/register ./stress-test.ts
```


Alternatively, if you're using a debugger like VSCode and want to break on
entry:

```
node -r ts-node/register --inspect --inspect-brk ./stress-test.ts
```