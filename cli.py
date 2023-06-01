import sys, os, re, argparse
from sat.analysis.calculators import SPC, VPC
from sat.data import Source, Data
from sat.jsonio import data2json

def parameters(C):
    def gethelp(s: str):
        docstart = getattr(C, s).__doc__.split("\n")[0]
        stripped = re.sub('\s{2,}', " ", docstart)
        return stripped.lower()
    
    maxlen = max(map(len, C.params.values()))
    strings = []
    for fname, dname in C.params.items():
        strings.append(dname + " "*(maxlen - len(dname) + 4) + gethelp(fname))
    return "\n    " + "\n    ".join(strings)

def parse(command_line=None):
    parser = argparse.ArgumentParser(
        prog="python " + os.path.basename(sys.argv[0]),
        description="Surface Analysis Tool\n"
                    "Calculate surface and vertex parameters on triangulated meshes via nominal to actual comparison.",
        epilog="avalable surface parameters:" + parameters(SPC) + "\n\n" + \
            "available vertex parameters:" + parameters(VPC),
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "out", 
        help="name of the output file\nNote that *_vertex.txt may also be created")
    parser.add_argument(
        "act", 
        help="path to the actual surface mesh")
    parser.add_argument(
        "nom", 
        help="path to the nominal surface mesh (optional)\n"
            "approximate from actual surface if omitted", 
        nargs="?",
        default="")
    parser.add_argument(
        "-a", "--align",
        help="align the input meshes",
        action="store_true")
    parser.add_argument(
        "-sp", "--surface",
        action="store_true",
        help="calculate surface parameters"
    )
    parser.add_argument(
        "-vp", "--vertex",
        action="store_true",
        help="calculate vertex parameters\n"
            "will create an additional output file *_vertex.txt"
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Show debugging information")

    args = parser.parse_args(command_line)
    if args.debug:
        print("Parsed arguments:")
        print(args)
        exit()
    return args

def calc(args):
    data = Data()

    data.act.source = Source(args.act)
    if args.nom == "": # approximate from act
        raise NotImplementedError("Approximation of the nominal surface is not yet implemented")
    else:
        data.nom.source = Source(args.nom)

    if args.align:
        data.align()

    if args.surface:
        for fname in SPC.params.keys():
            getattr(data.SPC, fname)

    if args.vertex:
        for fname in VPC.params.keys():
            getattr(data.VPC, fname)
    
    data2json(args.out, data)
    print("Finished")

def main():
    args = parse()
    calc(args)

if __name__ == "__main__":
    main()