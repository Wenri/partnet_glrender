import math
from collections import defaultdict
from operator import itemgetter


def pointcloud_log_gen():
    cur_im_id = None
    cur_rot_score = None
    cur_sim_score = None
    with open('../pointcloud.log', mode='r') as f:
        for line in f:
            part = line.split(':')
            scores = part[1].split()
            if part[0].startswith('Rot Sim Score'):
                cur_im_id = scores[0]
                cur_rot_score = scores[1]
            elif part[0].startswith('Reg Sim Score'):
                assert cur_im_id == scores[0]
                cur_sim_score = scores[1]
            elif part[0].startswith('Score'):
                yield cur_im_id, cur_rot_score, cur_sim_score, scores[0], scores[1]


all_results = [(im_id, rot_score, sim_score, cls, score) for
               im_id, rot_score, sim_score, cls, score in pointcloud_log_gen()
               if float(sim_score) < 0.005]
all_results.sort(key=itemgetter(4), reverse=True)

for scores in all_results:
    print('\t'.join(scores))

cls_score = defaultdict(float)
cls_count = defaultdict(int)

for im_id, rot_score, sim_score, cls, score in all_results:
    score = float(score)
    if not math.isinf(score):
        cls_score[cls] += float(score)
        cls_count[cls] += 1

cls_score = [(k, v / cls_count[k]) for k, v in cls_score.items()]
cls_score.sort(key=itemgetter(1), reverse=True)

for cls, score in cls_score:
    print(cls, score)
