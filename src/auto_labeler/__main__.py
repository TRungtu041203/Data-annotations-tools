import sys

from . import ingest  # noqa: F401  (import registers module)

# If the first CLI arg is "ingest", delegate:
if len(sys.argv) > 1 and sys.argv[1] == "ingest":
    # Shift args left so ingest's argparse sees them correctly
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    import auto_labeler.ingest  # noqa: F401

else:
    print("Auto-labeler package.  Use `python -m auto_labeler ingest --help`.")
