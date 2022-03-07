import os
import sys

import torch
import numpy as np

from scipy.interpolate import interp1d
from scipy.optimize import brentq

from sklearn import metrics
from sklearn.metrics import roc_curve, accuracy_score, precision_recall_curve
from tqdm import tqdm

import nemo.collections.asr as nemo_asr
from nemo.core.config import hydra_runner

version = "version1"
model_type = "Conformer"

@hydra_runner(config_path = "conf/{}/negative".format(model_type), config_name = "{}_ver1.yaml".format(model_type))
def main(cfg):
    
    avg_eer = 0
    avg_auc = 0
    avg_acc = 0
    avg_fscore = 0

    all_set = ["A", "B", "C", "D", "E"]
    all_set = ["A"]
    for num_set in all_set:
        log_dir = os.path.join("logs", model_type, "negative", num_set, version)
        
        best_eer = 1
        best_fscore = 0
        best_auc = 0
        best_epoch = 0

        for epoch in list(range(10, 200, 10)) + [num_set]:
            epoch = 80 
            model_path = os.path.join(log_dir, "checkpoints", "{}.nemo".format(epoch))
            model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(model_path)
            
            model.cuda()
            model.eval()
            cfg.model.test_ds.manifest_filepath = os.path.join("manifest", "negative", num_set, "test.json") 
            
            model.setup_test_data(cfg.model.test_ds)
            dataloader = model.test_dataloader()
            labels = []
            scores = []
            preds = []
            for batch in tqdm(dataloader):
                batch = [x.cuda() for x in batch]
                audio_signal, audio_signal_len, label, _ = batch
                logits, embs = model.forward(input_signal = audio_signal, input_signal_length = audio_signal_len)

                score = torch.softmax(logits, dim = -1, dtype = logits[0][0].dtype)
                
                pred = torch.argmax(score, dim = 1).cpu().detach().numpy()
                preds.extend(pred)

                label = label.cpu().detach().numpy()
                labels.extend(label)
                
                score = score.cpu().detach().numpy()
                scores.extend(score)
            
            # print(labels)
            # print(scores)
            scores = np.array(scores)
            
            pos_label = 0
            fpr, tpr, thresholds = roc_curve(labels, scores[:, pos_label], pos_label = pos_label)
            auc = metrics.auc(fpr, tpr)
            eer = brentq(lambda x : 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
            accuracy = accuracy_score(labels, preds)

            precision, recall, _ = precision_recall_curve(labels, scores[:, pos_label], pos_label = pos_label)
            precision += 1e-5
            recall += 1e-5
            fscore = max(2 / (1 / precision + 1 / recall))

            if eer < best_eer:
                best_eer = eer
                best_auc = auc
                best_acc = accuracy
                best_fscore = fscore
                best_epoch = epoch
            # print("model {}: eer = {}".format(epoch, eer))
            break
        avg_eer += best_eer
        avg_auc += best_auc
        avg_acc += best_acc
        avg_fscore += best_fscore
    
    total = len(all_set)
    print("negative EER = {:.4f}".format(avg_eer / total))
    print("negative AUC = {:.4f}".format(avg_auc/ total))
    print("negative Fscore = {:.4f}".format(avg_fscore / total))
    print("negative ACC = {:.4f}".format(avg_acc / total))
    print(best_epoch)

if __name__ == "__main__":
    main()
