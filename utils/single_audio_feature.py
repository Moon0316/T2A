
import os
import torch
import torch.nn.functional as F
import soundfile as sf
from fairseq import checkpoint_utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str, default='./data/wavs/org/train/a8.wav')
parser.add_argument('--output_path', type=str, default='./data/wav_features/org/a8.pt')
parser.add_argument('--hubert_path', type=str, default='./data/pretrained_models/chinese-hubert-large-fairseq-ckpt.pt')

args = parser.parse_args()

def postprocess(feats, normalize=False):
    if feats.dim() == 2:
        feats = feats.mean(-1)

    assert feats.dim() == 1, feats.dim()

    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    return feats

def handle_wav(wav_path, saved_to_path):
    wav, sr = sf.read(wav_path)
    assert os.path.exists(wav_path)
    feat = torch.from_numpy(wav).float()
    feat = postprocess(feat, normalize=saved_cfg.task.normalize)
    feats = feat.view(1, -1)
    padding_mask = (
        torch.BoolTensor(feats.shape).fill_(False)
    )
    inputs = {
        "source": feats.half().to(device),
        "padding_mask": padding_mask.to(device),
    }

    with torch.no_grad():
        logits = model.extract_features(**inputs)
        torch.save(logits[2], saved_to_path)

if __name__ == '__main__':
    for model_path, model_short_alias in [(args.hubert_path,"hubert_large")]:
        print(model_path)

        start = time.time()

        print("loading model(s) from {}".format(model_path))
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            [model_path],
            suffix="",
        )

        model = models[0]
        model = model.to(device)
        model = model.half()
        model.eval()

        args.output_dir = os.path.join(os.path.dirname(args.output_path))
        os.makedirs(args.output_dir,exist_ok=True)
        handle_wav(args.input_path, args.output_path)

        print("Done!")

    end = time.time() - start
    print(end)

