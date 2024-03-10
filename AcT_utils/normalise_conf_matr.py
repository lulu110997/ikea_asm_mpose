import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import pickle_wrapper as _pw


PATH = "/240214182212_layernorm/results/model_micro_data_2d_body_conf_matr.pickle"
SAVE_PATH = True
LABELS_VO = ['NA', 'align leg screw with table thread', 'align side panel holes with front panel dowels',
             'attach drawer back panel', 'attach drawer side panel', 'attach shelf to table', 'flip shelf',
             'flip table', 'flip table top', 'insert drawer pin', 'lay down back panel', 'lay down bottom panel',
             'lay down front panel', 'lay down leg', 'lay down shelf', 'lay down side panel', 'lay down table top',
             'other', 'pick up back panel', 'pick up bottom panel', 'pick up front panel', 'pick up leg', 'pick up pin',
             'pick up shelf', 'pick up side panel', 'pick up table top', 'position the drawer right side up', 'push table',
             'push table top', 'rotate table', 'slide bottom of drawer', 'spin leg', 'tighten leg']

LABELS_V = ['NA', 'align', 'attach', 'flip', 'insert', 'lay down', 'other', 'pick up', 'position', 'push', 'rotate',
            'slide', 'spin', 'tighten']
LABELS = LABELS_V
FONT_SIZE = 5
########################################################################################################################
def norm_matr(cm):
    return(np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2))

def get_acc_scores(cm):
    acc = np.sum(np.diag(cm))/np.sum(cm)
    mAcc = sum((np.diag(cm)/np.sum(cm, axis=1))) / len(LABELS)
    return acc, mAcc

def plot_cm(cm):
    X_SCALE = 1 if len(LABELS) == 33 else 1.4
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot()
    fig = plt.gcf()
    fig.set_size_inches(10.8, 10.8)
    ax = plt.gca()
    ax.set_title(f"Confusion matrix")
    ax.set_xticklabels(LABELS, rotation=25, ha='right', fontsize=X_SCALE*FONT_SIZE)
    ax.set_yticklabels(LABELS, fontsize=1.4 * FONT_SIZE)
    for labels in disp.text_.ravel():
        labels.set_fontsize(1.4 * FONT_SIZE)
    if SAVE_PATH:
        p = PATH.replace(".pickle", "_norm.jpg")
        plt.savefig(p, dpi=100, bbox_inches='tight')
    else:
        plt.show()
    plt.close('all')

max = [0, 0]; max_idx = 0
for i in range(250):
    try:
        conf_matr = _pw.open_pickle(PATH.replace('trial_0', f'trial_{i}'))
        scores = get_acc_scores(conf_matr)
        if scores[1] > max[1]:
            max = scores
            max_idx = i
    except FileNotFoundError:  # Incomplete trials
        pass
print(f"{max} at trial #{max_idx}")
