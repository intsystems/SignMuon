"""DEPRECATED: superseded by ``federated.tune``.

    python3 -m federated.tune --stage lr --device cuda:0 --methods ef21muonsign

This module used to launch a learning-rate sweep and rank the configurations by
the accuracy printed at ``--eval-round``. That accuracy was **test** accuracy:
there was no validation split in the federated setting, so every learning rate in
Table 5 was chosen by looking at the test set. With eleven methods and several rates
each that is dozens of peeks, and it is the single most common thing a reviewer
checks.

``federated.tune`` replaces it and fixes three things at once:

* selection reads validation accuracy on a 45k/5k split (the same 5k the
  centralized path holds out), and a tuning run never *loads* the test set;
* the grid is the 1-2-5 lattice ``centralized.tune`` uses, anchored per method by
  transporting the published rate to the chosen per-layer rule, so every method
  searches the same grid and a tuned value is quotable;
* an optimum landing on a grid endpoint widens the grid instead of being
  reported.

This file is kept so that old commands fail loudly with a pointer rather than
silently running the old protocol. Nothing here runs.
"""

from __future__ import annotations

import sys

MESSAGE = __doc__


def main() -> None:
    print(MESSAGE, file=sys.stderr)
    print("federated.grid has been removed. Use:\n\n"
          "    python3 -m federated.tune --stage lr --device cuda:0\n\n"
          "or, for the whole protocol in one command:\n\n"
          "    python3 -m federated.overnight --device cuda:0 --budget-hours 12\n",
          file=sys.stderr)
    sys.exit(2)


if __name__ == "__main__":
    main()
