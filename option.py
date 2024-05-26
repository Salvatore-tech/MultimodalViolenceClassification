import argparse

parser = argparse.ArgumentParser(description='WeaklySupAnoDet')
parser.add_argument('--modality', default='MIX2', help='the type of the input, AUDIO,RGB,FLOW, MIX1, MIX2, or MIX3, MIX_ALL')
parser.add_argument('--rgb-list', default='list/rgb.list', help='list of rgb features ')
parser.add_argument('--flow-list', default='list/flow.list', help='list of flow features')
parser.add_argument('--audio-list', default='list/audio.list', help='list of audio features')
parser.add_argument('--test-rgb-list', default='list/newsplit/test/rgb_test.list', help='list of test rgb features ')
parser.add_argument('--test-flow-list', default='list/newsplit/test/flow_test.list', help='list of test flow features')
parser.add_argument('--test-audio-list', default='list/newsplit/test/audio_test.list', help='list of test audio features')
parser.add_argument('--gt', default='list/newsplit/test/gt.npy', help='file of ground truth ')
parser.add_argument('--gpus', default=0, type=int, choices=[-1, 0, 1], help='gpus')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument('--batch-size', type=int, default=128, help='number of instances in a batch of data (default: 128)')
parser.add_argument('--workers', default=4, help='number of workers in dataloader')
parser.add_argument('--model-name', default='wsanodet', help='name to save model')
parser.add_argument('--pretrained-ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--feature-size', type=int, default=1024+128, help='size of feature (default: 2048)')
parser.add_argument('--num-classes', type=int, default=1, help='number of class')
parser.add_argument('--dataset-name', default='XD-Violence', help='dataset to train on (default: )')
parser.add_argument('--max-seqlen', type=int, default=200, help='maximum sequence length during training (default: 750)')
parser.add_argument('--max-epoch', type=int, default=50, help='maximum iteration to train (default: 100)')
parser.add_argument('--manifold', default='Lorentz', help='which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall, Lorentz]')
