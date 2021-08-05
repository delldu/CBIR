# -*- coding: utf-8 -*-

from __future__ import print_function

from evaluate import infer
from DB import Database

from color import Color
from daisy import Daisy
from edge import Edge
from gabor import Gabor
from HOG import HOG
from vggnet import VGGNetFeat
from resnet import ResNetFeat

depth = 5
d_type = "d1"
query_idx = 0


import pdb

if __name__ == "__main__":
    db = Database()

    # retrieve by color
    # method = Color()
    # samples = method.make_samples(db)
    # query = samples[query_idx]
    # _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
    # print(result)

    # # retrieve by daisy
    # method = Daisy()
    # samples = method.make_samples(db)
    # query = samples[query_idx]
    # _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
    # print(result)

    # # retrieve by edge
    # method = Edge()
    # samples = method.make_samples(db)
    # query = samples[query_idx]
    # _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
    # print(result)

    # # retrieve by gabor
    # method = Gabor()
    # samples = method.make_samples(db)
    # query = samples[query_idx]
    # _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
    # print(result)

    # # retrieve by HOG
    # method = HOG()
    # samples = method.make_samples(db)
    # query = samples[query_idx]
    # _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
    # print(result)

    # # retrieve by VGG
    # method = VGGNetFeat()
    # samples = method.make_samples(db)
    # query = samples[query_idx]
    # _, result = infer(query, samples=samples, depth=depth, d_type=d_type)
    # print(result)

    # retrieve by resnet
    method = ResNetFeat()
    samples = method.make_samples(db)
    query = samples[query_idx]

    # len(samples) -- 1200

    # (Pdb) pp query
    # {'cls': 'n01558993',
    #  'hist': array([1.3798475e-03, 5.5409706e-04, 4.6418867e-05, ..., 1.2377275e-03,
    #        5.9160178e-05, 2.1058379e-03], dtype=float32),
    #  'img': 'database/n01558993/n0155899300001151.jpg'}

    ap, result = infer(query, samples=samples, depth=depth, d_type=d_type)
    print("AP:", ap, "Query Result:", result)
