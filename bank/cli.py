import signal
import sys
import os
import argparse
from functools import partial
import numpy as np

from bank.commands import cmd_test, cmd_train



def parse_args():
    from bank import checkpoints
    def _parse_command(cmd, args): return (cmd, args)

    save_path = os.path.join(checkpoints.path(), 'bank')

    parser = argparse.ArgumentParser(prog='bank', description="homework of machine learning")
    sparser = parser.add_subparsers()

    train = sparser.add_parser('train', help='train models')
    train.set_defaults(func=partial(_parse_command, 'train'))
    train.add_argument('-s', '--save-path', type=str, default=save_path, help='checkpoint output path')
    train.add_argument('-r', '--restore', action='store_true', default=False, help='load previous checkpoint and train')
    train.add_argument('-n', '--nsteps', type=int, default=np.inf, help='training steps')
    train.add_argument('-l', '--learning-rate', type=float, default=0.1, help='learning rate')

    train = sparser.add_parser('test', help='train models')
    train.set_defaults(func=partial(_parse_command, 'test'))
    train.add_argument('-s', '--save-path', type=str, default=save_path, help='checkpoint output path')
    train.add_argument('-d', '--data-path', type=str, required=True, help='data path')

    args = parser.parse_args()
    if getattr(args, 'func', None) is None:
        parser.print_help()
        sys.exit(0)

    return args.func(args)



def main():
    def handle_signals(signum, frame):
        exit(1)

    signal.signal(signal.SIGINT, handle_signals)
    signal.signal(signal.SIGTERM, handle_signals)

    cmd, args = parse_args()

    if cmd == 'train':
        cmd_train(args)

    elif cmd == 'test':
        cmd_test(args)

    else: raise Exception('invalid command {}'.format(cmd))


if __name__ == '__main__':
    main()