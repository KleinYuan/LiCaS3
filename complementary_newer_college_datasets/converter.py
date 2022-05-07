import fire
import pandas as pd
import os


def convert(root_dir):
    data_dir = '{}/datasets/{}/testing'
    label_fp = '{}/datasets/{}/agreed_labels.txt'
    sub_folders = ['0', '1']
    annotations = {
        'seq_id': [],
        'baseline': [],
        'label': []
    }
    for _sub_folder in sub_folders:
        testing_dir = data_dir.format(root_dir, _sub_folder)
        label_fp = label_fp.format(root_dir, _sub_folder)
        labels = pd.read_csv(label_fp)
        for _subroot, _subdirs, _subfiles in os.walk(testing_dir):
            for _subfile in _subfiles:
                info = _subfile.split('_')
                sub_folder_id = int(info[0])
                seq_id = int(info[2])
                baseline = int(info[-1][2])
                label = labels.label[sub_folder_id]
                annotations['seq_id'].append(seq_id)
                annotations['baseline'].append(baseline)
                annotations['label'].append(label)
    pd.DataFrame(annotations).to_csv('annotation.csv')


if __name__ == '__main__':
    fire.Fire()
