"""
Since the pose_dump.txt could come with different poses, and after some modification, outputing the timestamp,
the evo could not directly work with it.
Namely, we will need to post-process the files to output matched timestamps for benchmark.
"""

import fire
import pandas as pd
import numpy as np

"""
python evo_matcher.py match \
--fn1 assets/2011_09_26_0093_baseline_stamped.txt \
--fn2 assets/2011_09_26_0093_licas3_id002_stamped.txt 
"""


def match(fn1, fn2):
    assert 'txt' in fn1, f"{fn1} is not a txt file"
    assert 'txt' in fn2, f"{fn2} is not a txt file"
    f1 = pd.read_csv(fn1, header=None, delimiter=r"\s+")
    f2 = pd.read_csv(fn2, header=None, delimiter=r"\s+")
    f1_ts = set(f1[0])
    f2_ts = set(f2[0])
    matched_ts = list(set(f1_ts) & set(f2_ts))

    f1_matched = f1[f1[0].isin(matched_ts)]
    f2_matched = f2[f2[0].isin(matched_ts)]
    # Next, we de-select the timestamp
    f12write = f1_matched.drop([0], axis=1)
    f22write = f2_matched.drop([0], axis=1)
    np.savetxt(fn1 + '.matched.txt', f12write.values, fmt='%f')
    np.savetxt(fn2 + '.matched.txt', f22write.values, fmt='%f')


if __name__ == '__main__':
    fire.Fire()
