import numpy as np
import os
import configargparse
import json
import torch
import torch.nn as nn
import numpy as np
from fastdtw import fastdtw
from dataloader import get_metadata,Dataset,collate_fn1,collate_fn2
from torch.utils.data import DataLoader
from model import EncoderDecoderModel
from sdtw_cuda_loss import SoftDTW
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")

parser = configargparse.ArgumentParser(
    description='ICASSP2033 model',
    config_file_parser_class=configargparse.YAMLConfigFileParser,
    formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', type=str, default="demo", help='name of current experiment')
parser.add_argument('--arch', type=str, default="chinese_hubert_large", help='name of current experiment')

# args for common settings
parser.add_argument('--gpu', default=0, type=int,help='GPU id to use.') 
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--val_batch_size", type=int, default=8)
parser.add_argument("--loss_type", type=str, default="SoftDTW", help='L1 L2 CE SoftDTW')

# args for using_soft_dtw
parser.add_argument("--gamma", type=float, default=0.001)

# args for data
parser.add_argument("--val_blendshape_path", type=str,)
parser.add_argument("--root_dir", type=str, help='root dir of the dataset')
parser.add_argument("--feature_dir", type=str, default='hubert_large')
parser.add_argument("--freq", type=int, default=50, help='audio feature frequency')
parser.add_argument("--val_speaker", type=str, default='org', help='speaker for validation')

# args for model
parser.add_argument("--feature_dim", type=int, default=128, help='64 for vocaset; 128 for BIWI')
parser.add_argument("--cov_dim", type=int, default=128, help='dimension of the convolutional feature')
parser.add_argument("--input_dim", type=int, default=39, help='dimension of the convolutional feature')
parser.add_argument("--seed", type=int, default=0, help='seed for random')
parser.add_argument("--feature_combine",action="store_true",help="whether to combine the features")

args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)

if not os.path.exists('exp'):
    os.makedirs('exp')
args.ckpt = os.path.join('exp', args.name, 'checkpoints')
assert os.path.exists(args.ckpt),'checkpoint not found!'

args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def validate(args, val_loader, model):

    save_path = args.ckpt

    # load or create checkpoint 
    init_epoch = 0
    assert os.path.exists(save_path), 'no experiment in {} !'.format(save_path)
    
    if args.loss_type == "L1":
            criterion = nn.L1Loss(reduction='mean') 
    elif args.loss_type == "L2":
        criterion = nn.MSELoss(reduction='mean') 
    elif args.loss_type == "SoftDTW":
        criterion = SoftDTW(use_cuda=True, gamma=args.gamma)
    else:
        print("check loss type")
        exit(0)
    rate = args.freq//25
    print('audio feature is {} Hz!'.format(rate*25))
    epoches = []
    val_best_loss = 1000  
    score_dict =dict()
    for filename in os.listdir(save_path):
        if filename.endswith('.pth'):
            model.load_state_dict(torch.load(os.path.join(save_path, filename),map_location=args.device))
            model = model.cuda(args.gpu)      
            model.eval()

            val_distance_log = []
            val_loss_log = []
        
            for audio_features, blendshape_labels, audio_lengths, blendshape_lengths, audio_file_names in val_loader:
                # to gpu
                audio, vertice= audio_features.cuda(args.gpu), blendshape_labels.cuda(args.gpu)
                outputs,blendshapes = model(audio,  vertice, audio_lengths, blendshape_lengths)
                loss = 0

                for i in range(outputs.shape[0]):
                    blendshape_length = blendshape_lengths[i]
                    audio_length = audio_lengths[i]//rate
                    single_output, single_blendshape = outputs[i], blendshapes[i]
                    single_output, single_blendshape = single_output[:audio_length,].unsqueeze(0), single_blendshape[:blendshape_length,].unsqueeze(0)

                    y_true = single_blendshape.squeeze()
                    y_pred = single_output.squeeze()
                    _y_true = y_true.cpu().detach().numpy()
                    _y_pred = y_pred.cpu().detach().numpy()
                    distance, result = fastdtw(_y_true, _y_pred, dist=mean_squared_error)
                    val_distance_log.append(distance)

                    single_loss = criterion(single_output, single_blendshape)
                    loss += single_loss
                    val_loss_log.append(single_loss.item())

            current_loss = np.mean(val_loss_log)
            score_dict[filename] = current_loss
            if current_loss < val_best_loss:
                val_best_loss = current_loss
                best_name = filename
                print("BEST: epoch: {}, Total loss:{:.7f}, dtw dis:{:.4f}".format(filename, np.mean(val_loss_log),np.mean(val_distance_log))) 
    score_dict['best'] = best_name
    score_dict['best_score'] = val_best_loss
    with open(args.json_path,'w') as f:
        json.dump(score_dict,f)
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):
    val_data_path = os.path.join(args.root_dir, 'wav_features', args.val_speaker, args.feature_dir,'val_1')
    val_data_list = get_metadata(args.val_blendshape_path, val_data_path, args.arch, 2, args.device)
    val_data = Dataset(val_data_list,'val')
    if args.arch == 'chinese_hubert_large':
        collate_fn = collate_fn1
    elif args.arch in ['mfcc','ppg']:
        collate_fn = collate_fn2
    else:
        print('check collate_fn type!')
        exit(1)
    val_loader = DataLoader(dataset=val_data, batch_size=args.val_batch_size, shuffle=False, collate_fn=collate_fn, num_workers = args.num_workers)
    
    #build model
    model = EncoderDecoderModel(args)
    print("model parameters: ", count_parameters(model))

    # to cuda
    assert torch.cuda.is_available()
    print("Use GPU: {} for validation".format(args.gpu))
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    
    json_dir =os.path.join('validate', args.name,args.val_speaker)
    os.makedirs(json_dir,exist_ok=True)
    args.json_path = os.path.join(json_dir, 'score.json')
    
    validate(args, val_loader, model)

    
if __name__=="__main__":
    main(args)
