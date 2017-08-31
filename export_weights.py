
""" Extract neural network weights and output JSON """

import json
import numpy as np
import torchvision.models.alexnet


def get_weights(model, layers):

    exported_weights = []
    for layer_num, layer_name in enumerate(layers):

        if layer_name.split('.')[0] == 'features':

            filters = len(state[layer_name + '.bias'])
            for filter_num in range(filters):
                bias = state[layer_name + '.bias'][filter_num]
                weights = state[layer_name + '.weight'][filter_num].tolist()
                exported_weights.append({
                    'filterIndex' : filter_num,
                    'layerIndex' : layer_num,
                    'weights' : weights,
                    'type' : 'Conv2d',
                    'bias' : bias
                })

        if layer_name.split('.')[0] == 'classifier':

            neurons = len(state[layer_name + '.bias'])
            for neuron_num in range(neurons):
                bias = state[layer_name + '.bias'][neuron_num]
                weights = state[layer_name + '.weight'][neuron_num].tolist()
                exported_weights.append({
                    'neuronIndex' : neuron_num,
                    'layerIndex' : layer_num,
                    'weights' : weights,
                    'type' : 'Linear',
                    'bias' : bias
                })

    return exported_weights


if __name__ == '__main__':

    model = torchvision.models.alexnet(pretrained=True)
    state = model.state_dict()

    layers = [
        'features.0',
        'features.3',
        'features.6',
        'features.8',
        'features.10',
        'classifier.1',
        'classifier.4',
        'classifier.6'
    ]

    weights = get_weights(model, layers)
    for w in weights: print(json.dumps(w))
