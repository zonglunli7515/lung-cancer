import json
import itertools

params_file_name = 'params_to_select'

network_type = 'lstm'
optimizer = ['adam']
epochs = [10]
hidden_size = [24]
batch_size = [5]
lr = [0.001]
n_layers = [1]
dropout = [0.0]

params_grid = list(itertools.product(epochs, hidden_size, batch_size, lr, n_layers, dropout, optimizer))
args = dict()
args['models_grid'] = {'model_%d' % step:
{    
    'epochs': epochs,
    'hidden size': hidden_size,
    'batch size': batch_size,
    'lr': lr,
    'n_layers': n_layers,
    'dropout': dropout,
    'optimizer': optimizer
} 
for step, (epochs, hidden_size, batch_size, lr, n_layers, dropout, optimizer) in enumerate(params_grid)
}
args['network_type'] = network_type

import io

with io.open('../data/' + params_file_name + '.json', 'w', encoding='utf8') as outfile:
    str_ = json.dumps(args,
                      indent = 4, sort_keys=True,
                      separators=(',', ': '), ensure_ascii=False)
    outfile.write(str(str_))


"""
results = dict()
results['model0'] = [1,2,3]
results['model1'] = [2,3,4]

with io.open('../data/' + 'list' + '.json', 'w', encoding='utf8') as outfile:
    str_ = json.dumps(results,
                      indent = 4, sort_keys=True,
                      separators=(',', ': '), ensure_ascii=False)
    outfile.write(str(str_))
"""