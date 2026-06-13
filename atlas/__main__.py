import argparse
import multiprocessing

from . import build, demo, retile, run, server


def main() -> None:
    # Windows/macOS spawn re-imports this module in each pool worker; this guards
    # against the children re-running the CLI (and is required if ever frozen).
    multiprocessing.freeze_support()
    p = argparse.ArgumentParser(prog="atlas", description="Image Atlas: explore large image collections locally")
    sub = p.add_subparsers(dest="cmd", required=True)
    run.add_parser(sub)
    build.add_parser(sub)
    server.add_parser(sub)
    retile.add_parser(sub)
    demo.add_parser(sub)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
