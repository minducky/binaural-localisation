import torch
import torchinfo
from models.ducky import *
from models.yang import *
from models.dummy import *
import yaml

def get_model(config):
    if config['MODEL'] == 'Yang':
        model = YangModel(config)
    elif config['MODEL'] == 'YangMLP':
        model = YangMLP(config)
    elif config['MODEL'] == 'DuckyItd':
        model = DuckyItdModel(config)
    elif config['MODEL'] == 'DuckyIldOnset':
        model = DuckyIldOnsetModel(config)
    elif config['MODEL'] == 'DuckyIldWhole':
        model = DuckyIldWholeModel(config)
    elif config['MODEL'] == 'DuckyItdIldWhole':
        model = DuckyItdIldWholeModel(config)
    elif config['MODEL'] == 'Ducky':    # ITDILDOnset
        model = DuckyModel(config)
    elif config['MODEL'] == 'DuckyItdMultiScale':
        model = DuckyItdMultiScaleModel(config)
    elif config['MODEL'] == 'DuckyIldMultiScale':
        model = DuckyIldMultiScaleModel(config)
    elif config['MODEL'] == 'DuckyMultiScale':
        model = DuckyMultiScaleModel(config)
    elif config['MODEL'] == 'Dummy':
        model = Dummy(config)
    elif config['MODEL'] == 'Dummy1M':
        model = Dummy1M(config)
    elif config['MODEL'] == 'Dummy5M':
        model = Dummy5M(config)
    elif config['MODEL'] == 'Dummy10M':
        model = Dummy10M(config)
    else:
        raise ValueError(f'Unrecognised Model : {config["MODEL"]}')

    device = torch.device(f"cuda:{config['GPU_NUM']}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    input_size = (1, 31, 24, 12) if config['MODEL'] == 'YangMLP' else (1, int(44100*1.3), 2)
    torchinfo.summary(model, input_size=input_size,
            col_names=["input_size", "output_size", "num_params", "mult_adds"])

    return model

if __name__ == '__main__':
    with open('config_model.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    get_model(config)