import signal
import sys
import os
import argparse
from functools import partial

from bank.commands import cmd_test, cmd_train



def parse_args():
    def _parse_command(cmd, args): return (cmd, args)

    parser = argparse.ArgumentParser(prog='bank', description="homework of machine learning")
    sparser = parser.add_subparsers()

    train = sparser.add_parser('train', help='train models')
    train.set_defaults(func=partial(_parse_command, 'config'))
    train.add_argument('-o', '--output', type=str, default=os.path.join('./'), help='checkpoint output path')
    train.add_argument('-c', '--continue', action='store_true', default=False, help='load previous checkpoint')

    train = sparser.add_parser('test', help='train models')
    train.set_defaults(func=partial(_parse_command, 'config'))
    train.add_argument('-i', '--input', type=str, default=os.path.join('./'), help='checkpoint input path')
    train.add_argument('-d', '--data', type=str, help='data path')

    args = parser.parse_args()
    if getattr(args, 'func', None) is None:
        parser.print_help()
        sys.exit(0)




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