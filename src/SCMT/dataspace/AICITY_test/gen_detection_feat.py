import pickle
import numpy as np
import gc

"""
(以下注释都为猜测还未验证)这个文件应该是读取ReID（行人重识别）的结果，将其整理并保存到Numpy数组中，然后将数组保存为Numpy文件
"""

# 定义一个字典，用于存储不同摄像头（cam）的检测结果。
cam_det = {'c041': [],
           'c042': [],
           'c043': [],
           'c044': [],
           'c045': [],
           'c046': []}
# 定义一个空字典，用于存储检测框的特征
box_feat = {}
print('* reading ReID results, it may cost a lot of time')
# 定义一个包含要处理的ReID结果文件名的列表
feat_list = ['aic22_1_test_infer_v2_HR48_eps.pkl', 'aic22_1_test_infer_v2_Convnext.pkl', 'aic22_1_test_infer_v2_Res2Net200.pkl', 'aic22_1_test_infer_v2_R50.pkl', 'aic22_1_test_infer_v2_ResNext101.pkl']

# 遍历feat_list中的文件名，这里只处理第一个文件（feat_list[:1]）
for res_file in feat_list[:1]:
    res_file = '/mnt/bk_data/track1/' + res_file
    # 使用二进制读取模式打开ReID结果文件
    with open(res_file, 'rb') as f:
        # 使用pickle模块加载文件中的对象，encoding='latin1' 用于支持Python 2.x 中的pickle格式
        obj = pickle.load(f, encoding='latin1')
        _count = 0
        for k, v in obj.items():
            # 检查键k是否包含字符串,如果是则跳过此项
            if 'ipynb_checkpoints' in k:
                continue
            # 从键 k 中提取摄像头的名称（cam）
            cam = k.split('/')[-1].split('_')[0]
            if cam != 'c042': continue
            if k not in box_feat.keys():
                box_feat[k] = []
            box_feat[k].append(v)
            _count += 1
            # if _count > 10: break

gc.collect()

# 将box_feat中的数据整理为一个包含检测框及其特征的列表cam_det，并对每个检测框的相关信息进行提取和组合。
for k, vs in box_feat.items():
    file_name = k.split('/')[-1].replace('.png', '')
    cam_name = file_name.split('_')[0]
    fid = int(file_name.split('_')[1])
    x = int(file_name.split('_')[2])
    y = int(file_name.split('_')[3])
    w = int(file_name.split('_')[4])
    h = int(file_name.split('_')[5])
    conf = float(file_name.split('_')[6])
    feat = []
    for v in vs:
        feat += v.tolist()
    line = [fid, -1, x, y, w, h, conf, -1, -1, -1] + feat
    cam_det[cam_name].append(line)

del box_feat
gc.collect()

for cam_name in cam_det.keys():
    print('* start to save', cam_name)
    det_feat_npy = np.array(sorted(cam_det[cam_name]))
    #print(det_feat_npy.shape)
    np.save('{}.npy'.format(cam_name), det_feat_npy)

del cam_det
gc.collect()
