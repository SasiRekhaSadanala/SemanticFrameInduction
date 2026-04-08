import nltk
import sys
nltk.download('framenet_v17', quiet=True)
from nltk.corpus import framenet as fn

lu = fn.lus('run.v')[0]
if len(lu.exemplars) > 0:
    ex = lu.exemplars[0]
    print("Text:", ex.text)
    print("Target:", ex.Target)
    print("FE:", ex.FE[0])
else:
    print("No exemplars found")
