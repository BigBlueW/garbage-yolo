import os
import argparse
from ultralytics import YOLO

MODEL_PATH = "best.pt"

def main():
    parser = argparse.ArgumentParser(description="評估 YOLO 模型在 test 或 val 上的指標")
    parser.add_argument("--split", default="test", choices=["val", "test"],
                        help="要評估的資料分群 (預設: test)")
    parser.add_argument("--model", default=MODEL_PATH, help=f"模型路徑 (預設: {MODEL_PATH})")
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型檔案: {model_path}，請先確認是否已經完成！")
        return

    print(f"🚀 載入模型進行評估: {model_path}")
    model = YOLO(model_path)

    # 選擇資料集配置檔
    data_yaml = "dataset.yaml" if os.path.exists("dataset.yaml") else "custom_dataset.yaml"
    print(f"🔍 開始在 ({data_yaml} -> {args.split}) 上進行驗證...")
    
    metrics = model.val(data=data_yaml, split=args.split)

    print("\n" + "="*55)
    print(f"📊 {args.split.upper()} 評估結果 ({args.split} Metrics)")
    print("="*55)

    # 判斷是 OBB 還是 HBB（改用 model.task，較可靠）
    if getattr(model, "task", None) == "obb" or "obb" in str(getattr(model, "model", "")).lower():
        metric_obj = metrics.obb if hasattr(metrics, 'obb') and metrics.obb is not None else metrics.box
        task_type = "OBB (Oriented Bounding Box)"
    else:
        metric_obj = metrics.box
        task_type = "HBB (Horizontal Box)"
    
    print(f"任務類型: {task_type}")
    print("-" * 55)

    # 取得整體指標
    map50 = metric_obj.map50
    map95 = metric_obj.map
    precision = metric_obj.mp
    recall = metric_obj.mr

    print(f"綜合 Precision (精準率): {precision:.4f}  (預測有垃圾時的準確率)")
    print(f"綜合 Recall    (召回率): {recall:.4f}  (真實垃圾中被成功偵測的比例)")
    print(f"綜合 mAP@50            : {map50:.4f}  (主流 IoU 0.50 指標)")
    print(f"綜合 mAP@50-95         : {map95:.4f}  (嚴格綜合指標)")
    print("="*55)
    
    # 針對各類別的詳細指標
    print(f"{'類別名稱 (Class)':<18} | {'Precision':<10} | {'Recall':<10} | {'mAP@50':<10}")
    print("-" * 55)
    
    names = model.names
    for i, class_idx in enumerate(metric_obj.ap_class_index):
        c_name = names[class_idx]
        c_p = metric_obj.p[i]
        c_r = metric_obj.r[i]
        c_map50 = metric_obj.ap50[i]
        print(f"{c_name:<18} | {c_p:<10.4f} | {c_r:<10.4f} | {c_map50:<10.4f}")
        
    print("="*55)
    print(f"評估報表與圖表已儲存至: {metrics.save_dir}")

if __name__ == '__main__':
    main()
