import argparse

# NOTE: DO NOT CHANGE ANY OF THE ARGPARSE ARGUMENTS - CHANGE NOTHING IN THIS FUNCTION
def process_commandline():
    parser = argparse.ArgumentParser(
        description='A basic text-to-speech app that synthesises speech using diphone concatenation.')

    # basic synthesis arguments
    parser.add_argument('--diphones', default="./diphones",
                        help="Folder containing diphone wavs")
    parser.add_argument('--play', '-p', action="store_true", default=False,
                        help="Play the output audio")
    parser.add_argument('--outfile', '-o', action="store", dest="outfile",
                        help="Save the output audio to a file", default=None)
    parser.add_argument('phrase', nargs='?',
                        help="The phrase to be synthesised")

    # Arguments for extensions
    parser.add_argument('--volume', '-v', default=None, type=int,
                        help="An int between 0 and 100 representing the desired volume")
    parser.add_argument('--reverse', '-r', action="store", default=None, choices=['words', 'phones', 'signal'],
                        help="Speak backwards in a mode specified by string argument: 'words', 'phones' or 'signal'")
    parser.add_argument('--addpron', '-a', action="store", default=None,
                        help="File of additional word pronunciations.  Use to add new ones or override built-in ones.")
    parser.add_argument('--fromfile', '-f', action="store", default=None,
                        help="Open file with given name and synthesise all text, which can be multiple sentences.")
    parser.add_argument('--crossfade', '-c', action="store_true", default=False,
                        help="Enable slightly smoother concatenation by cross-fading between diphone units")

    args = parser.parse_args()

    if (args.fromfile and args.phrase) or (not args.fromfile and not args.phrase):
        parser.error('Must supply either a phrase or "--fromfile" to synthesise (but not both)')

    return args
