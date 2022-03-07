import os
import sys

import torch
import numpy as np

# torch.set_flush_denormal(True)

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
    
    num_set = "A"
    log_dir = os.path.join("logs", model_type, "negative", num_set, version)

    epoch = 80

    model_path = os.path.join(log_dir, "checkpoints", "{}.nemo".format(epoch))
    model = nemo_asr.models.EncDecSpeakerLabelModel.restore_from(model_path)
    
    model.cuda()
    model.eval()
    cfg.model.test_ds.manifest_filepath = os.path.join("manifest", "negative", num_set, "test_single.json") 
    
    model.setup_test_data(cfg.model.test_ds)
    dataloader = model.test_dataloader()
    labels = []
    scores = []
    preds = []
    embs = []
    idx = 0
    for batch in tqdm(dataloader):
        batch = [x.cuda() for x in batch]
        audio_signal, audio_signal_len, label, _ = batch
        logits, emb = model.forward(input_signal = audio_signal, input_signal_length = audio_signal_len)
        
        embs.append(emb[0].cpu().detach().numpy())
        
        score = torch.softmax(logits, dim = -1, dtype = logits[0][0].dtype)
        
        pred = torch.argmax(score, dim = 1).cpu().detach().numpy()
        preds.extend(pred)

        label = label.cpu().detach().numpy()
        labels.extend(label)
        
        score = score.cpu().detach().numpy()
        scores.extend(score)
                    
        if idx == 2:
            break
        idx += 1
        
    
    print(preds)
    print(labels)
    print(scores)
    
    # plot_emb(embs, labels)

def plot_emb(embs: list, labels: list):
    embs = np.array(embs)
    labels = np.array(labels)

    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
    from sklearn.metrics.pairwise import paired_euclidean_distances, euclidean_distances
    from sklearn import manifold
    import matplotlib.pyplot as plt

    # distance_matrix = euclidean_distances(X = embs[labels == 0], Y = embs[labels == 1])
    means_sub = np.array([embs[labels == 0].mean(axis = 0), embs[labels == 1].mean(axis = 0)]) - embs.mean(axis = 0)    
    covs = np.array([np.cov(embs[labels == 0].T), np.cov(embs[labels == 1].T)])
    ns = np.array([(labels == 0).sum(), (labels == 1).sum()])
    inter_var = np.matmul(means_sub.T * (ns / len(labels)), means_sub)
    intra_var = np.sum(covs * ((ns - 1) / len(labels))[:, None, None], axis = 0)
    eigv_inter, _ = np.linalg.eigh(inter_var)
    eigv_intra, _ = np.linalg.eigh(intra_var)
    # print(np.linalg.det(np.linalg.inv(intra_var)@inter_var))
    print(eigv_inter.mean() / eigv_intra.mean())
    sys.exit(0)

    n_components = 3
    # pca = PCA(n_components = n_components)
    # embs_3d = pca.fix(embs).transform(embs)
    t_sne = manifold.TSNE(n_components = n_components, init = "pca", random_state = 0)
    embs_3d = t_sne.fit_transform(embs)
    colors = ["navy", "darkorange"]

    fig = plt.figure(figsize = (15,15))
    
    if n_components == 3:

        ax = fig.add_subplot(projection='3d')
        for index, data in enumerate(embs_3d):
            ax.scatter(xs = embs_3d[index][0], ys = embs_3d[index][1], zs = embs_3d[index][2], 
                       color = colors[labels[index]], label = labels[index])
    
    elif n_components == 2:
        for index, data in enumerate(embs_3d):
            plt.scatter(x = embs_3d[index][0], y = embs_3d[index][1],
                       color = colors[labels[index]], label = labels[index])
        
    plt.savefig("figs/transformer/embedding.png")

if __name__ == "__main__":
    main()
