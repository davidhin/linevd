"""
Directly predicting dataflow outputs

The approach of "coding dataflow with DNN" is not working so well because of some incompatibilities
and complexity that arises when coding it up.
I want to try predicting dataflow output based on the source code.
This is akin to an encoder-decoder model which translates source code to the abstract state.

source code -> encoder -> decoder -> abstract state (dataflow output)
                                     - reaching definition
                                     - live variables
                                     - interval bounds

We will pretrain an encoder to extract information critical to computing the abstract state.
The supervision for this task is cheap: compute the abstract state for many source code programs from Github.

Then, we will detect bugs using the pretrained encoder, plus additional features.

source code -> encoder          |
            -> word embedding   |-> bug detection
            -> metadata         |
               - AST node type
               - complexity metrics
               - connectivity
"""

"""
Tasks:
1. Gather supervision (compute abstract state from source code)
  - Automatic method
  - Parse source code with Joern
  - Run Joern dataflow solver and store outputs
2. Train encoder-decoder model
  - Model architecture agnostic of feature representation
  - Sequence based
  - Graph based
"""

# %%

"""
1. Gather supervision
"""

import sastvd.helpers.datasets as svdds
import sastvd.helpers.joern_session as svdjs
import sastvd.helpers.joern as svdj
import os

test = True
df = svdds.bigvul(sample=test)
df = svdds.bigvul_filter(df, check_file=True, check_valid=True)

sess = svdjs.JoernSession()
try:
    problem = "reachingdefinition"
    for fpath in df["id"].map(svdds.itempath):
        if not os.path.exists(f"{fpath}.dataflow.json") or test:
            svdj.run_joern_dataflow(sess, fpath, problem, verbose=args.verbose)
finally:
    sess.close()
