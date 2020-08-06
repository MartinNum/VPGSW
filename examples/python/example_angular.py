# -*- coding: utf-8 -*-
from n2 import HnswIndex
import random
import numpy as np
import time


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')

if __name__ == "__main__":

    # datas = fvecs_read("/home/martin/Project/experiment/data_set/siftsmall/siftsmall_base.fvecs")
    # querys = fvecs_read("/home/martin/Project/experiment/data_set/siftsmall/siftsmall_query.fvecs")
    # labels = ivecs_read("/home/martin/Project/experiment/data_set/siftsmall/siftsmall_groundtruth.ivecs")

    # datas = fvecs_read("/home/martin/Project/experiment/data_set/sift1M_128/sift_base.fvecs")
    querys = fvecs_read("/home/martin/Project/experiment/data_set/sift1M_128/sift_query.fvecs")

    # datas = fvecs_read("/home/martin/Project/experiment/data_set/glove_100/glove-100_base.fvecs")
    # querys = fvecs_read("/home/martin/Project/experiment/data_set/glove_100/glove-100_query.fvecs")
    # labels = ivecs_read("/home/martin/Project/experiment/data_set/glove_100/glove-100_groundtruth.ivecs")

    # datas = fvecs_read("/home/martin/Project/experiment/data_set/gist_960/gist_base.fvecs")
    # querys = fvecs_read("/home/martin/Project/experiment/data_set/gist_960/gist_query.fvecs")
    # labels = ivecs_read("/home/martin/Project/experiment/data_set/gist_960/gist_groundtruth.ivecs")

    # sum = 0
    # for data in datas:
    #     sum++
    # print(sum)

    f = 128
    # t = HnswIndex(f, "L2")  # HnswIndex(f, "L2 or angular")
    # time_start = time.time()
    # for data in datas:
    #     t.add_data(data)
    
    # t.build(m=28, n_threads=4)
    # t.save('VPGSW.n10h_SIFT128d_28-12-4n_1M_L2')
    # time_end = time.time()


    u = HnswIndex(f, "L2")
    # u = HnswIndex(f)
    u.load('VPGSW.n10h_SIFT128d_28-12-4n_1M_L2')   
    # u.load('VPGSW.n10h_Gist960d_35-15-4n_1M_L2')   
    print("索引加载完成")


    # search_id = 6
    # k = 10
    # neighbor_ids = u.search_by_id(search_id, k, 100)
    # print(
    #     "[search_by_id]: Nearest neighborhoods of id {}: {}".format(
    #         search_id,
    #         neighbor_ids))

    # example_vector_query = [random.gauss(0, 1) for z in range(f)]
    # nns = u.search_by_vector(example_vector_query, k, 1000, 2, include_distances=True)# 2-SearchById_true() 默认-SearchById_()
    # print(
    #     "[search_by_vector]: Nearest neighborhoods of vector {}: {}".format(
    #         example_vector_query,
    #         nns))

    # nnsv = u.search_by_vector_violence(example_vector_query, k, include_distances=True)   
    # print(
    #     "[search_by_vector_violence]: Nearest neighborhoods of vector {}: {}".format(
    #         example_vector_query,
    #         nnsv))

    # hg = 0
    # hh = 1
    # hd = hh
    # while hd > 0:
    #     hd -= 1
    #     example_vector_query = [random.gauss(0, 1) for z in range(f)]
    #     nns = u.search_by_vector(example_vector_query, k, 20, 2, include_distances=True)

    #     # print(
    #     # "[search_by_vector]: Nearest neighborhoods of vector {}: {}".format(
    #     #     example_vector_query,
    #     #     nns))

    #     nnsv = u.search_by_vector_violence(example_vector_query, k, include_distances=True)
    #     for (key,value) in nns:
    #         for (keyv,valuev) in nnsv:
    #             if key == keyv:
    #                 hg +=1
    #                 break

    k = 10
    hg = 0
    hd = 100
    ef = 10
    while ef < 200:
        for i in range(100):
            nns = u.search_by_vector(querys[i], k, ef, 2, include_distances=True)# 2-SearchById_true() 默认-SearchById_()

            # nnsv = labels[i]
            # print(nnsv)


            nnsv = u.search_by_vector_violence(querys[i], k, include_distances=True)
            # print(nnsv)
            for (key,value) in nns:
                for (keyv,valuev) in nnsv:
                    if key == keyv:
                        hg +=1
                        break
        comprison_times = u.get_comparison_times()

        print("ef: {}".format(ef))
        print("比较次数： {}".format(comprison_times*1.0/hd))
        print("精度为: {}%".format((hg*1.0/(k*hd))*100))
        print("-------")
        hg = 0
        ef += 5
        # build_time = time_end - time_start
        # print(build_time)
    print("运行结束")

    # build_time = time_end - time_start
    # print(build_time)