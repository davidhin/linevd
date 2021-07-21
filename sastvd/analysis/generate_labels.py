import sastvd.helpers.datasets as svdd
import sastvd.helpers.git as svdg

df = svdd.bigvul()
svdg.mp_code2diff(df)
