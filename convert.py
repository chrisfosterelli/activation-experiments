
import sys
import json
import pickle
import base64

src = 'n01645776_10130_adv.pickle'
imgsrc = 'n01645776_10130_adv.png'

layers = pickle.load(open(src, 'rb'))
image = base64.b64encode(open(imgsrc, 'rb').read())
image = 'data:image/png;base64,' + image

for layer_index, layer in enumerate(layers):

    print(json.dumps({
        'imageData'       : image,
        'layerIndex'      : layer_index,
        'layerShape'      : layer['activation'][0].shape,
        'activation'      : layer['activation'][0].tolist(),
        'layerType'       : layer['layer_name'].split('(')[0],
        'layerName'       : layer['layer_name'],
        'activationMapID' : 'n01645776_10130',
        'folderName'      : 'n01645776',
        'imageHeight'     : 224,
        'imageWidth'      : 224
    }))
