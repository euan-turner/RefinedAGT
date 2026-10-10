"""
Makes the upstream poisoning-paper modules (scripts/poisoning_paper) importable by the tasks that
build on them: ``robust_regularization`` (the robust pre-training regulariser) and ``octmnist_train``
(the OCT-MNIST class splits). Importing this module appends that directory to ``sys.path``; it is
appended, not prepended, so it can never shadow a module of this directory.

Key external dependencies: none.
"""

import os
import sys

PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "poisoning_paper")
if PATH not in sys.path:
    sys.path.append(PATH)
