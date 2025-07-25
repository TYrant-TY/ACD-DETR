import argparse
import os
import sys
import json
from contextlib import redirect_stdout
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tidecv import TIDE, datasets
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def parse_opt():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--anno_json', type=str, default='/root/ty/RTDETR-main/dataset/VisDrone/data.json', help='annotation json path')
    # /root/ty/RTDETR-main/dataset/DOTA/DOTAv1-split/dota.json
    parser.add_argument('--anno_json', type=str, default='/root/ty/RTDETR-main/dataset/VisDrone/data.json', help='annotation json path')
    parser.add_argument('--pred_json', type=str, default='/root/ty/RTDETR-main/runs/val/rtdetr-r18/predictions.json', help='prediction json path')
    parser.add_argument('--output_dir', type=str, default='/root/ty/RTDETR-main/runs/val/rtdetr-18', help='output directory for results')
    parser.add_argument('--output_file', type=str, default='coco_data.txt', help='output file name')
    
    return parser.parse_known_args()[0]

def filter_predictions(pred_json, valid_img_ids, output_dir):
    """过滤预测结果，只保留有效的图像ID"""
    with open(pred_json, 'r') as f:
        predictions = json.load(f)
    
    # 过滤预测结果
    filtered_predictions = []
    removed_count = 0
    
    for pred in predictions:
        if pred['image_id'] in valid_img_ids:
            filtered_predictions.append(pred)
        else:
            removed_count += 1
    
    # 保存过滤后的预测结果
    filtered_pred_path = os.path.join(output_dir, 'filtered_predictions.json')
    with open(filtered_pred_path, 'w') as f:
        json.dump(filtered_predictions, f)
    
    print(f"原始预测数量: {len(predictions)}")
    print(f"过滤后预测数量: {len(filtered_predictions)}")
    print(f"移除的预测数量: {removed_count}")
    print(f"过滤后的预测文件保存到: {filtered_pred_path}")
    
    return filtered_pred_path

if __name__ == '__main__':
    opt = parse_opt()
    anno_json = opt.anno_json
    pred_json = opt.pred_json
    output_dir = opt.output_dir
    output_file = opt.output_file
    
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)
    
    # 完整的输出文件路径
    output_path = os.path.join(output_dir, output_file)
    
    try:
        # 首先检查图像ID匹配情况
        print("正在检查图像ID匹配情况...")
        anno = COCO(anno_json)
        valid_img_ids = set(anno.getImgIds())
        
        # 读取预测结果
        with open(pred_json, 'r') as f:
            predictions = json.load(f)
        
        pred_img_ids = set([pred['image_id'] for pred in predictions])
        
        print(f"标注文件中的图像数量: {len(valid_img_ids)}")
        print(f"预测文件中的图像数量: {len(pred_img_ids)}")
        print(f"匹配的图像数量: {len(valid_img_ids & pred_img_ids)}")
        print(f"不匹配的预测图像数量: {len(pred_img_ids - valid_img_ids)}")
        
        # 如果有不匹配的图像，过滤预测结果
        if pred_img_ids - valid_img_ids:
            print("发现不匹配的图像ID，正在过滤预测结果...")
            filtered_pred_json = filter_predictions(pred_json, valid_img_ids, output_dir)
            pred_json = filtered_pred_json
        
        # 打开文件用于写入结果
        with open(output_path, 'w', encoding='utf-8') as f:
            # 重定向stdout到文件
            with redirect_stdout(f):
                print("=" * 80)
                print("COCO EVALUATION RESULTS")
                print("=" * 80)
                print(f"Annotation file: {anno_json}")
                print(f"Prediction file: {pred_json}")
                print("-" * 80)
                
                # COCO评估
                anno = COCO(anno_json)  # init annotations api
                pred = anno.loadRes(pred_json)  # init predictions api
                eval = COCOeval(anno, pred, 'bbox')
                eval.evaluate()
                eval.accumulate()
                
                print("\nCOCO Detection Metrics:")
                print("-" * 40)
                eval.summarize()
                
                print("\n" + "=" * 80)
                print("TIDE EVALUATION RESULTS")
                print("=" * 80)
                
                # TIDE评估
                tide = TIDE()
                tide.evaluate_range(datasets.COCO(anno_json), datasets.COCOResult(pred_json), mode=TIDE.BOX)
                tide.summarize()
        
        # TIDE图表保存到指定目录
        tide_plot_dir = os.path.join(output_dir, 'tide_plots')
        os.makedirs(tide_plot_dir, exist_ok=True)
        tide.plot(out_dir=tide_plot_dir)
        
        print(f"\n结果已保存到: {output_path}")
        print(f"TIDE图表已保存到: {tide_plot_dir}")
        
    except Exception as e:
        print(f"评估过程中出现错误: {str(e)}")
        print("请检查标注文件和预测文件的格式是否正确")
        
        # 将错误信息也写入输出文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(f"评估失败: {str(e)}\n")
            f.write(f"标注文件: {anno_json}\n")
            f.write(f"预测文件: {pred_json}\n")