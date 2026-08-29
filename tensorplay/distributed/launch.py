#
import sys
import warnings
from argparse import ArgumentParser

from . import run as run_module


def parse_args(args):
    parser = ArgumentParser(
        description="DEPRECATED tensorplay.distributed.launch; use "
        "`tensorplay.distributed.run` instead."
    )
    parser.add_argument("--nnodes", action="store", default="1:1")
    parser.add_argument("--node-rank", "--node_rank", action="store",
                        default=0, type=int)
    parser.add_argument("--nproc-per-node", "--nproc_per_node", action="store")
    parser.add_argument("--master-addr", "--master_addr", action="store",
                        default="127.0.0.1")
    parser.add_argument("--master-port", "--master_port", action="store",
                        default="0")
    parser.add_argument("-m", "--module", action="store_true")
    parser.add_argument("--no-python", "--no_python", action="store_true")
    parser.add_argument("training_script", type=str)
    parser.add_argument("training_script_args", nargs="*")
    return parser.parse_args(args)


def launch(args):
    warnings.warn(
        "tensorplay.distributed.launch will be removed; "
        "use tensorplay.distributed.run.",
        FutureWarning,
        stacklevel=2,
    )
    args = parse_args(args)
    if args.module or args.no_python:
        raise NotImplementedError(
            "--module/--no_python are not supported by the tp subprocess "
            "launcher yet."
        )
    run_module.main([
        "--nproc_per_node", str(args.nproc_per_node),
        "--master_addr", str(args.master_addr),
        "--master_port", str(args.master_port),
        args.training_script, *args.training_script_args,
    ])


def main(args=None):
    launch(sys.argv[1:] if args is None else args)
