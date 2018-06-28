import argparse
from visdom import Visdom

from arguments import get_args
from visualize import visdom_plot

args = get_args()

viz = Visdom(port=args.port)
win = None

try:
    # Sometimes monitor doesn't properly flush the outputs
    win = visdom_plot(viz, win, args.log_dir, args.env_name,
                      args.algo, args.num_frames)
except IOError:
    pass