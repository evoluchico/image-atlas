import argparse

from . import build, demo, retile, run, server


def main() -> None:
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
