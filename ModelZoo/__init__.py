import os
import torch

MODEL_DIR = '/home/dlg/SISR/LAM_Demo/ModelZoo/models/'


NN_LIST = [
    'RCAN',
    'CARN',
    'RRDBNet',
    'RNAN', 
    'SAN',
    'TRANS',
    'SWIN',
    'SCET',
    'ELAN'
]


MODEL_LIST = {
    'RCAN': {
        'Base': 'RCAN.pt',
    },
    'CARN': {
        'Base': 'CARN_7400.pth',
    },
    'RRDBNet': {
        'Base': 'RRDBNet_PSNR_SRx4_DF2K_official-150ff491.pth',
    },
    'SAN': {
        'Base': 'SAN_BI4X.pt',
    },
    'RNAN': {
        'Base': 'RNAN_SR_F64G10P48BIX4.pt',
    },
    'TRANS': {
        'Base': 'model_best.pt',
    },
    'SWIN': {
        'Base': '34000_G.pth',
    },
    'SCET': {
        'Base': 'SCETx4.pth',
    },
    'ELAN': {
        'Base': 'ELAN_model_x4_122.pt',
    },
    
    
}

def print_network(model, model_name):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Network [%s] was created. Total number of parameters: %.1f kelo. '
          'To see the architecture, do print(network).'
          % (model_name, num_params / 1000))


def get_model(model_name, factor=4, num_channels=3):
    """
    All the models are defaulted to be X4 models, the Channels is defaulted to be RGB 3 channels.
    :param model_name:
    :param factor:
    :param num_channels:
    :return:
    """
    print(f'Getting SR Network {model_name}')
    if model_name.split('-')[0] in NN_LIST:

        if model_name == 'RCAN':
            from .NN.rcan import RCAN
            net = RCAN(factor=factor, num_channels=num_channels)

        elif model_name == 'CARN':
            from .CARN.carn import CARNet
            net = CARNet(factor=factor, num_channels=num_channels)

        elif model_name == 'RRDBNet':
            from .NN.rrdbnet import RRDBNet
            net = RRDBNet(num_in_ch=num_channels, num_out_ch=num_channels)

        elif model_name == 'SAN':
            from .NN.san import SAN
            net = SAN(factor=factor, num_channels=num_channels)

        elif model_name == 'RNAN':
            from .NN.rnan import RNAN
            net = RNAN(factor=factor, num_channels=num_channels)
            
        elif model_name == 'TRANS':
            from .NN.transenet import TransENet
            net = TransENet()
            
        elif model_name == 'SWIN':
            from .NN.swinir import SwinIR
            net = SwinIR()
            
        elif model_name == 'SCET':
            from .NN.scet import SCET
            net = SCET()
        
        elif model_name == 'ELAN':
            from .NN.elan import ELAN
            net = ELAN()

        else:
            raise NotImplementedError()

        print_network(net, model_name)
        return net
    else:
        raise NotImplementedError()


def load_model(model_loading_name):
    """
    :param model_loading_name: model_name-training_name
    :return:
    """
    splitting = model_loading_name.split('@')
    if len(splitting) == 1:
        model_name = splitting[0]
        training_name = 'Base'
    elif len(splitting) == 2:
        model_name = splitting[0]
        training_name = splitting[1]
    else:
        raise NotImplementedError()
    assert model_name in NN_LIST or model_name in MODEL_LIST.keys(), 'check your model name before @'
    net = get_model(model_name)
    state_dict_path = os.path.join(MODEL_DIR, MODEL_LIST[model_name][training_name])
    print(f'Loading model {state_dict_path} for {model_name} network.')
    
    state_dict = torch.load(state_dict_path, map_location='cpu')
    
    
    #针对ELAN模型权重模型的字段：
    #del state_dict['epoch']
    #state_dict=state_dict['model_state_dict']

    #state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    
    
    #针对SCET模型权重模型的字段：
    #del state_dict['meta']
    #del state_dict['optimizer']
    #n_state_dict = state_dict['state_dict']
    #new_pretrained_dict = {}
    #for key, value in n_state_dict.items():
    #    if key.startswith("generator."):
    #        new_key = key.replace("generator.", "")
    #        new_pretrained_dict[new_key] = value
    #   else:
    #        new_pretrained_dict[key] = value
    #net.load_state_dict(new_pretrained_dict)

    net.load_state_dict(state_dict)
    return net




