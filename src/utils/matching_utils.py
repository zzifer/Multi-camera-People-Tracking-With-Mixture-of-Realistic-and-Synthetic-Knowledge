from shapely.geometry import Polygon
import os
import json


"""
计算两个多边形的交并比（IoU）
bbox是边界框的四个顶点坐标  pts是多边形的顶点坐标
get_iou将这些坐标转化为Shapely的多边形对象（polygon1和polygon2），
然后计算它们的交集区域，并使用.area属性获取交集区域的面积。
最后，返回交集区域的面积除以第一个多边形的面积，即交并比值
"""
def get_iou(bbox, pts):
    polygon1 = Polygon(bbox)
    polygon2 = Polygon(pts)
    intersect = polygon1.intersection(polygon2).area
    return intersect / polygon1.area


"""
检查边界框是否在指定区域内
bboxes是边界框的列表  outzone是区域的顶点坐标列表
首先创建了一个列表 points，其中包含了区域的顶点坐标。
然后，通过迭代bboxes中的每个边界框，以及迭代points中的每个区域顶点，使用 get_iou 函数计算边界框与区域的交并比 _iou。
如果 _iou 大于0.9（90%），则将计数 _count 增加1，并且终止对当前边界框的其他区域检查（因为已经确定在指定区域内）。
最后，返回在区域内的边界框数量占总边界框数量的比例
"""
def check_inside(bboxes, outzone):
    # Outzone 
    points = outzone
    _count = 0
    for box in bboxes:
        x, y, w, h = box
        for point in points:
            _iou = get_iou([(x,y), (x+w,y), (x+w, y+h), (x,y+h)], point)
            if _iou > 0.9: 
                _count += 1
                break
    return _count / len(bboxes)

def calculate_err(x, y):
    return (abs(x-y)/(y+1e-8)) * 100



"""
用于加载ROI（Region of Interest）的边界框和区域数
它首先创建了两个空字典 out_bbox 和 out_zone 用于存储边界框和区域数据。
然后，通过遍历指定目录（"datasets/ROI"）中的文件，根据文件名的前缀（"out_bbox" 或 "out_zone_tracklet"）和相应的摄像机标识，
将加载的JSON文件数据存储到相应的字典中。最后，函数返回这两个字典，其中 out_bbox 包含了边界框数据，out_zone 包含了区域数据。
"""
def load_roi_mask():
    out_bbox = {}
    out_zone = {}
    for file in os.listdir("datasets/ROI"):
        if "out_bbox" in file:
            camera = file.replace("out_bbox_", "").replace(".json", "")
            out_bbox[camera] = json.load(open(os.path.join("datasets/ROI", file)))
        if "out_zone_tracklet" in file:
            camera = file.replace("out_zone_tracklet_", "").replace(".json", "")
            out_zone[camera] = json.load(open(os.path.join("datasets/ROI", file)))
    return out_bbox, out_zone
