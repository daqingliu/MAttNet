from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import sys
import json
import numpy as np
import h5py
import time
import argparse
from tqdm import tqdm

# model
import _init_paths
from layers.joint_match import JointMatching
from loaders.gt_mrcn_loader import GtMRCNLoader

# torch
import torch


def load_model(checkpoint_path, opt):
    tic = time.time()
    model = JointMatching(opt)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model'].state_dict())
    model.eval()
    model.cuda()
    print('model loaded in %.2f seconds' % (time.time()-tic))
    return model


def extract_feats(params):
    # set up loader
    data_json = osp.join('cache/prepro', params['dataset_splitBy'], 'data.json')
    data_h5 = osp.join('cache/prepro', params['dataset_splitBy'], 'data.h5')
    loader = GtMRCNLoader(data_h5=data_h5, data_json=data_json)

    # load mode info
    model_prefix = osp.join('output', params['dataset_splitBy'], params['id'])
    infos = json.load(open(model_prefix+'.json'))
    model_opt = infos['opt']
    model_path = model_prefix + '.pth'
    model = load_model(model_path, model_opt)

    # loader's feats
    feats_dir = '%s_%s_%s' % (model_opt['net_name'], model_opt['imdb_name'], model_opt['tag'])
    args.imdb_name = model_opt['imdb_name']
    args.net_name = model_opt['net_name']
    args.tag = model_opt['tag']
    args.iters = model_opt['iters']
    loader.prepare_mrcn(head_feats_dir=osp.join('cache/feats/', model_opt['dataset_splitBy'], 'mrcn', feats_dir), 
                      args=args) 
    ann_feats = osp.join('cache/feats', model_opt['dataset_splitBy'], 'mrcn', 
                       '%s_%s_%s_ann_feats.h5' % (model_opt['net_name'], model_opt['imdb_name'], model_opt['tag']))
    loader.loadFeats({'ann': ann_feats})

    # check model_info and params
    assert model_opt['dataset'] == params['dataset']
    assert model_opt['splitBy'] == params['splitBy']

    split = params['split']
    model.eval()
    loader.resetIterator(split)

    pbar = tqdm(total=len(loader.split_ix[split]))
    while True:
        data = loader.getTestBatch(split, model_opt)
        ann_ids = data['ann_ids']
        sent_ids = data['sent_ids']
        Feats = data['Feats']
        labels = data['labels']
        image_id = data['image_id']

        for i, sent_id in enumerate(sent_ids):
            feat_h5 = osp.join('./data/matt_gt_feats', params['dataset_splitBy'], str(image_id)+'_'+str(sent_id)+'.h5')

            if os.path.exists(feat_h5):
                continue

            # expand labels
            label = labels[i:i+1]      # (1, label.size(1))
            max_len = (label != 0).sum().data[0]
            label = label[:, :max_len] # (1, max_len) 
            expanded_labels = label.expand(len(ann_ids), max_len) # (n, max_len)

            # forward
            sub_feat, loc_feat = model.extract_sub_loc_feats(Feats, expanded_labels)
            sub_feat = sub_feat.data.cpu().numpy()
            loc_feat = loc_feat.data.cpu().numpy()

            f = h5py.File(feat_h5, 'w')
            f.create_dataset('sub_feat', dtype=np.float32, data=sub_feat)
            f.create_dataset('loc_feat', dtype=np.float32, data=loc_feat)
            f.close()

        pbar.update(1)
        # if we already wrapped around the split
        if data['bounds']['wrapped']:
            break
    pbar.close()

    return


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='refcocog', help='dataset name: refclef, refcoco, refcoco+, refcocog')
    parser.add_argument('--splitBy', type=str, default='umd', help='splitBy: unc, google, berkeley')
    parser.add_argument('--split', type=str, default='test', help='split: testAB or val, etc')
    parser.add_argument('--id', type=str, default="mrcn_cmr_with_st", help='model id name')
    args = parser.parse_args()
    params = vars(args)

    # make other options
    params['dataset_splitBy'] = params['dataset'] + '_' + params['splitBy']
    extract_feats(params)
