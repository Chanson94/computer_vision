import torch

dict = torch.load('experiments/self_training/LKDN-S_adan_x4_scratch_L1_fine_tune_L2/models/net_g_1000000.pth')  # load the pth file

for key in list(dict.keys()):
    if key == 'params':
        del dict[key]

new_dict = {}

for key in list(dict['params_ema'].keys()):
    new_dict[key] = dict['params_ema'][key]

torch.save(new_dict, 'experiments/self_training/LKDN-S_adan_x4_scratch_L1_fine_tune_L2/models/LKDN-S_del_x4.pth')

changed_dict = torch.load('experiments/self_training/LKDN-S_adan_x4_scratch_L1_fine_tune_L2/models/LKDN-S_del_x4.pth')
for key in list(changed_dict.keys()):
    print(key)
