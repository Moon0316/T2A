import numpy as np
from tqdm import tqdm
import os
import configargparse
import wandb,json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from fastdtw import fastdtw
from dataloader import get_dataloaders
from model import EncoderDecoderModel
from sdtw_cuda_loss import SoftDTW
from fastdtw import fastdtw
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
import csv

parser = configargparse.ArgumentParser(
    description='ICASSP2033 model',
    config_file_parser_class=configargparse.YAMLConfigFileParser,
    formatter_class=configargparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--name', type=str, default="demo", help='name of current experiment')
parser.add_argument('--arch', type=str, default="chinese_hubert_large", help='name of current experiment')
parser.add_argument("--feature_combine",action="store_true",help="whether to combine the features")

# args for common settings
parser.add_argument('--gpu', default=0, type=int,help='GPU id to use.') 
parser.add_argument("--use_wandb", type=bool, default=False, help='whether to use wandb')
parser.add_argument("--use_tensorboard", type=bool, default=False, help='whether to use tensorboard')
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--train_batch_size", type=int, default=8)
parser.add_argument("--val_batch_size", type=int, default=1)
parser.add_argument("--test_batch_size", type=int, default=1)

# args for training
parser.add_argument("--lr", type=float, default=0.001, help='learning rate')
parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='gradient accumulation')
parser.add_argument("--max_epoch", type=int, default=10, help='number of epochs')
parser.add_argument("--loss_type", type=str, default="SoftDTW", help='L1 L2 CE SoftDTW')

# args for using_soft_dtw
parser.add_argument("--gamma", type=float, default=0.1)

# args for testing
parser.add_argument("--test_mode", type=bool, default=False, help='test the lastest model')
parser.add_argument("--test_epoch", type=str, default='best', help='test the lastest model')
parser.add_argument("--output_path", type=str, default="result", help='path to the predictions')
parser.add_argument("--test_input_path", type=str, default=None, help='explicitly assign path to test data when output path is not result')

# args for data
parser.add_argument("--train_blendshape_path", type=str,)
parser.add_argument("--val_blendshape_path", type=str,)
parser.add_argument("--root_dir", type=str, help='root dir of the dataset')
parser.add_argument("--feature_dir", type=str, default='hubert_large')
parser.add_argument("--train_speaker_list", nargs='+', type=str, default=['org'], help='list of train speaker name')
parser.add_argument("--train_json", type=str, default="dataset_json/train_small_50.json", help='path to the json file which include train data list')
parser.add_argument("--freq", type=int, default=50, help='audio feature frequency')

# args for model
parser.add_argument("--feature_dim", type=int, default=64, help='64 for vocaset; 128 for BIWI')
parser.add_argument("--cov_dim", type=int, default=128, help='dimension of the convolutional feature')
parser.add_argument("--input_dim", type=int, default=39, help='dimension of the convolutional feature')
parser.add_argument("--seed", type=int, default=0, help='seed for random')
parser.add_argument("--rnn_type", type=str, default="LSTM", help='RNN cell type, RNN|LSTM|GRU')
parser.add_argument("--transformer", action='store_true', help='whether to use transformer')
parser.add_argument("--n_heads", type=int, default=4, help='number of heads in transformer')
parser.add_argument("--hidden_size", type=int, default=128, help='hidden size of RNN/Transformer')
parser.add_argument("--dim_feedforward", type=int, default=2048, help='feedforward size of Transformer')

args = parser.parse_args()
print(args)

torch.manual_seed(args.seed)

if not os.path.exists('exp'):
    os.makedirs('exp')
args.ckpt = os.path.join('exp', args.name, 'checkpoints')
if not os.path.exists(args.ckpt):
    os.makedirs(args.ckpt)
if not args.test_mode:
    with open(os.path.join('exp', args.name, 'config.json'), "w") as outfile:
        json.dump(vars(args), outfile, indent=4)

args.device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")


def trainer(args, train_loader, dev_loader, model, optimizer, epoch=1000):

    save_path = args.ckpt
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma=0.95) #TODO

    # load or create checkpoint 
    init_epoch = 0
    if os.path.exists(save_path):
        epoches = []
        for filename in os.listdir(save_path):
            if not filename.startswith('best'):
                epoches.append(filename)

        if len(epoches) > 0:
            epoch_index = np.argmax([int(epoch.split("_")[0]) for epoch in epoches])
            init_epoch = int(epoches[epoch_index].split("_")[0])
            scheduler.step(int(init_epoch))
            model.load_state_dict(torch.load(os.path.join(save_path, epoches[epoch_index]),map_location=args.device))
            model = model.cuda(args.gpu)

    else:
        print("Modles will be saved to:", args.save_path)
        os.makedirs(save_path)

    iteration = 0
    val_best_loss = 100

    if args.loss_type == "L1":
        criterion = nn.L1Loss(reduction='mean') 
    elif args.loss_type == "L2":
        criterion = nn.MSELoss(reduction='mean') 
    elif args.loss_type == "SoftDTW":
        criterion = SoftDTW(use_cuda=True, gamma=args.gamma)
    else:
        print("check loss type")
        exit(1)

    rate = args.freq//25
    print('audio feature is {} Hz!'.format(rate*25))
    
    for e in range(int(init_epoch),epoch):

        distance_log = []
        loss_log = []

        model.train()
        pbar = tqdm(enumerate(train_loader),total=len(train_loader))
        length = len(train_loader)

        for i, (audio_features, blendshape_labels, audio_lengths, blendshape_lengths, audio_file_names) in pbar:
            optimizer.zero_grad()
            
            iteration += 1
            audio, vertice, = audio_features.cuda(args.gpu), blendshape_labels.cuda(args.gpu)
            outputs,blendshapes = model(audio,  vertice, audio_lengths, blendshape_lengths)

            loss = 0

            for i in range(outputs.shape[0]):
                blendshape_length = blendshape_lengths[i]
                audio_length = blendshape_length
                if args.loss_type == "SoftDTW":
                    audio_length = audio_lengths[i]//rate     
                single_output, single_blendshape = outputs[i], blendshapes[i]
                single_output, single_blendshape = single_output[:audio_length,].unsqueeze(0), single_blendshape[:blendshape_length,].unsqueeze(0)

                y_true = single_blendshape.squeeze()
                y_pred = single_output.squeeze()
                _y_true = y_true.cpu().detach().numpy()
                _y_pred = y_pred.cpu().detach().numpy()
                distance, result = fastdtw(_y_true, _y_pred, dist=mean_squared_error)
                distance_log.append(distance)

                single_loss = criterion(single_output, single_blendshape)
                loss += single_loss
                loss_log.append(single_loss.item())

            loss.backward()
            optimizer.step()
            
            pbar.set_description("(Epoch {}, LR {}, iteration {}) {} LOSS:{:.7f}, DTW DIS:{:.4f}".format((e+1), optimizer.param_groups[0]['lr'], iteration , args.loss_type, np.mean(loss_log),np.mean(distance_log)))

        scheduler.step()

        model.eval()

        val_distance_log = []
        val_loss_log = []
        

        for audio_features, blendshape_labels, audio_lengths, blendshape_lengths, audio_file_names in dev_loader:
            # to gpu
            audio, vertice= audio_features.cuda(args.gpu), blendshape_labels.cuda(args.gpu)
            outputs,blendshapes = model(audio,  vertice, audio_lengths, blendshape_lengths)

            loss = 0

            for i in range(outputs.shape[0]):
                blendshape_length = blendshape_lengths[i]
                single_output, single_blendshape = outputs[i], blendshapes[i]
                single_output, single_blendshape = single_output[:blendshape_length,].unsqueeze(0), single_blendshape[:blendshape_length,].unsqueeze(0)

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
        pbar.set_description("(VAL Epoch {}, LR {}, iteration {}) {} LOSS:{:.7f}, DTW DIS:{:.4f}".format((e+1), optimizer.param_groups[0]['lr'], iteration , args.loss_type, np.mean(val_loss_log),np.mean(val_distance_log)))
        if current_loss < val_best_loss:
            val_best_loss = current_loss
            print("BEST: epoch: {}, Total loss:{:.7f}, dtw dis:{:.4f}".format(e+1, np.mean(val_loss_log),np.mean(val_distance_log))) 
        torch.save(model.state_dict(), os.path.join(save_path,'{}_model_{:0.4f}.pth'.format(e+1,np.mean(val_loss_log))))
    return model

@torch.no_grad()
def test(args, model, dataset, dataset_type, test_epoch='best'):
    os.makedirs(os.path.join(args.output_path, dataset_type, args.name), exist_ok=True)   
    save_path = args.ckpt
    
    # find best epoch and print
    if test_epoch == 'best':
        score = 1000
        for file in os.listdir(save_path):
            if file.endswith('.pth'):
                tmp = float(file[:-4].split('_')[2])
                if tmp < score:
                    score = tmp
                    file_path = os.path.join(save_path, file)
                    test_epoch = file[:-4]
        print('load best model from {}'.format(file_path))

    model.load_state_dict(torch.load(os.path.join(save_path, '%s.pth'%(test_epoch)),map_location=args.device))
    model = model.cuda(args.gpu)
    model.eval()
    
    if dataset_type == "test":
        data_loader = dataset["test"]
    elif dataset_type == "val":
        data_loader = dataset["valid"]
    else:
        data_loader = dataset["train"]

    for audio_features, blendshape_labels, audio_lengths, blendshape_lengths, audio_file_names in data_loader:
        # to gpu
        audio_features = audio_features.cuda(args.gpu)
        prediction = model.predict(audio_features)
        prediction = prediction.squeeze() # (b, seq_len, V*3) 
        
        print(audio_file_names[0])       
        if args.output_path == 'result':
            csv_path = os.path.join('result', dataset_type, args.name, str(audio_file_names[0]) + ".csv")                        
        else:
            os.makedirs(os.path.join(args.output_path, args.name),exist_ok=True)
            csv_path = os.path.join(args.output_path, args.name, str(audio_file_names[0]) + ".csv")
        
        data = prediction.detach().cpu().numpy()
        with open(csv_path,'w',newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow("JawForward,JawRight,JawLeft,JawOpen,MouthClose,MouthFunnel,MouthPucker,MouthRight,MouthLeft,MouthSmileLeft,MouthSmileRight,MouthFrownLeft,MouthFrownRight,MouthDimpleLeft,MouthDimpleRight,MouthStretchLeft,MouthStretchRight,MouthRollLower,MouthRollUpper,MouthShrugLower,MouthShrugUpper,MouthPressLeft,MouthPressRight,MouthLowerDownLeft,MouthLowerDownRight,MouthUpperUpLeft,MouthUpperUpRight,BrowDownLeft,BrowDownRight,BrowInnerUp,BrowOuterUpLeft,BrowOuterUpRight,CheekPuff,CheekSquintLeft,CheekSquintRight,NoseSneerLeft,NoseSneerRight".split(','))
            csv_writer.writerows(data)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main(args):
    #load data
    if args.test_mode:
        args.train_batch_size = 1
        args.val_batch_size = 1
        args.test_batch_size = 1
    dataset = get_dataloaders(args)

    #build model
    model = EncoderDecoderModel(args)
    print("model parameters: ", count_parameters(model))

    # to cuda
    assert torch.cuda.is_available()
    print("Use GPU: {} for training".format(args.gpu))
    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)
    
    if  args.test_mode:
        if args.output_path == 'result':
            # test(args, model, dataset, "train", args.test_epoch)
            # test(args, model, dataset, "val", args.test_epoch)
            test(args, model, dataset, "test", args.test_epoch)
        else:
            test(args, model, dataset, "test", args.test_epoch)
        print("csv has been generated to the result dir.")
        exit(0)

    # train
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), lr=args.lr, weight_decay=1e-5)
    model = trainer(args, dataset["train"], dataset["valid"], model, optimizer, epoch=args.max_epoch)
    test(args, model, dataset, "test", args.test_epoch)

    
if __name__=="__main__":
    if args.use_wandb:
        config_dictionary = dict(yaml=args.config)
        with wandb.init(project="T2A", config=config_dictionary):
            assert wandb.run is not None
            wandb.run.name = args.name
            main(args)
    else:
        main(args)
