# Model / system card

## Intended use

Train a two-layer neural network with backpropagation to recognize noisy geometric image patterns. Educational local experiment; no production deployment is implied.

## Data

Synthetic arrays, manually written documents or replayed JSON. No real user records. Seeds and sample inputs are recorded in app.py and sample_input.json.

## Method

An 8x8 image is flattened into 64 features, projected through a 24-unit tanh hidden layer, and classified with softmax. Minimize cross entropy using full-batch gradient descent.

## Evaluation protocol

Generate independent training and test noise around vertical, horizontal and diagonal patterns. Compare test accuracy with a training-majority baseline and inspect a numerical gradient check.

## Known limits

This is an MLP on simple synthetic images, not a CNN, natural-image benchmark or pretrained vision model.

## Verification status

Local tests and default experiment are executed during packaging. Live model generation is not exercised in the build. A successful validation result is limited to the checks implemented in code.

## Further work

Add rotated shapes and unseen translations, mini-batching, checkpoints, confusion matrices and a comparison with a convolutional model.
