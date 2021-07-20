from importlib import reload

import sastvd.helpers.datasets as svdd
import sastvd.helpers.git as svdg

reload(svdg)

df = svdd.bigvul()
svdg.mp_code2diff(df)

# TESTING
gitlines = svdg.get_codediff("bigvul", "177764")
print(gitlines["diff"])
