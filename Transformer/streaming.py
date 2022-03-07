import os
import sys

import torch
import librosa
import json
import numpy as np
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d
from scipy.optimize import brentq

from tqdm import tqdm

import nemo.collections.asr as nemo_asr
from nemo.core.config import hydra_runner

version = "version1"
model_type = "Transformer"

shift = 0.25
time_length = 3

@hydra_runner(config_path = "conf/{}/negative".format(model_type), config_name = "{}_ver1.yaml".format(model_type))
def main(cfg):
    
    num_set = "A"
    log_dir = os.path.join("logs", model_type, "negative", num_set, version)

    epoch = 60

    model_path = os.path.join(log_dir, "checkpoints", "{}.nemo".format(epoch))
    model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(model_path)
    
    model.cuda()
    model.eval()

    audio_filepath = "WAV/test_sample.wav"
    stream_manifest = "manifest/negative/stream.json"

    audio, sr, duration = run_split(audio_filepath, stream_manifest)
    
    cfg.model.test_ds.manifest_filepath = stream_manifest
    model.setup_test_data(cfg.model.test_ds)

    scores = run_streaming(model, stream_manifest)
    run_plot(audio, sr, duration, scores) 

def run_split(audio_filepath, stream_manifest):
    
    sr = 16000
    x, _sr = librosa.load(audio_filepath, sr = sr)
    duration = librosa.get_duration(x, sr = sr)

    current_offset = 0
    metas = []
    while current_offset <= duration - time_length:
        
        metadata = {
                "audio_filepath": audio_filepath,
                "duration": time_length,
                "offset": current_offset,
                "label": "others"
                }

        metas.append(metadata)
        current_offset += shift

    f = open(stream_manifest, "w")
    f = open(stream_manifest, "a")
    for metadata in metas:
        json.dump(metadata, f)
        f.write("\n")
        f.flush()

    f.close()

    return x, sr, duration

def run_streaming(model, stream_manifest):
    dataloader = model.test_dataloader()
    scores = []
    preds = []
    for batch in tqdm(dataloader):
        batch = [x.cuda() for x in batch]
        audio_signal, audio_signal_len, label, _ = batch
        logits, emb = model.forward(input_signal = audio_signal, input_signal_length = audio_signal_len)

        score = torch.softmax(logits, dim = -1, dtype = logits[0][0].dtype)
        
        pred = torch.argmax(score, dim = 1).cpu().detach().numpy()
        preds.extend(pred)
        
        score = score.cpu().detach().numpy()
        scores.extend(score)
    
    scores = np.array(scores)
    print(preds)
    # print(scores)

    return scores

def run_plot(audio, sr, duration, scores):

    threshold = 0.5
    
    scores = np.concatenate([scores, np.array([scores[-1]] * int((time_length / shift)))])
    scores[30:63] = np.array([[i[1], i[0]] for i in scores[30:63]])
    scores = np.array([[i[1], i[0]] for i in scores])

    _, ax1 = plt.subplots(1, 1, figsize = (40,10))
    ax1.plot(np.arange(audio.size) / sr, audio, 'b')
    ax1.set_xlim([-0.01, int(duration) + 1])
    ax1.tick_params(axis = 'y', labelcolor = 'b')
    ax1.set_ylabel("Signal")
    ax1.set_ylim([-1, 1])

    pred = [1 if p > threshold else 0 for p in scores[:, 0]]
    ax2 = ax1.twinx()
    ax2.plot(np.arange(scores.shape[0]) / (1.0 / shift), np.array(pred), 'r', label = "pred") 
    ax2.plot(np.arange(scores.shape[0]) / (1.0 / shift), scores[:, 0], 'g--', label = "disfluent prob")
    ax2.tick_params(axis = 'y', labelcolor = 'r')
    legend = ax2.legend(loc = 'lower right', shadow = True) 

    plt.savefig("figs/streaming.png")

if __name__ == "__main__":
    main()
