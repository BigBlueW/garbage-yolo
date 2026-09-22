import os
import shutil
import argparse
import torch
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv11-OBB 訓練（垃圾分類）")
    parser.add_argument("--model", default="yolo11x-obb.pt", help="基礎預訓練權重")
    parser.add_argument("--data", default="dataset.yaml", help="資料集設定檔")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=1024, help="輸入解析度（小物體/角度精度吃此值）")
    parser.add_argument("--batch", type=int, default=16, help="可設 -1 讓 YOLO 自動找最大 batch")
    parser.add_argument("--lr0", type=float, default=0.002, help="初始學習率 (推薦搭配 cos_lr 使用 0.002)")
    parser.add_argument("--seed", type=int, default=42, help="固定隨機種子以重現實驗")
    parser.add_argument("--degrees", type=float, default=90.0, help="OBB 旋轉增強角度")
    parser.add_argument("--mixup", type=float, default=0.3, help="MixUp 合成資料機率")
    parser.add_argument("--copy_paste", type=float, default=0.5, help="Copy-Paste 合成資料機率")
    parser.add_argument("--freeze", type=int, default=None, help="凍結前 N 層（如 10，先只練 head）")
    parser.add_argument("--name", default="yolo11x_obb_model")
    parser.add_argument("--device", default=None, help="如 '0' 或 'cpu'，預設自動偵測")
    parser.add_argument("--no-copy-best", action="store_true",
                        help="實驗用：不把 best.pt 複製到根目錄，避免覆蓋你的正式權重")
    return parser.parse_args()


def main():
    args = parse_args()

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    print(f"🚀 準備啟動 YOLOv11-OBB 端到端訓練流程...")
    print(f"📌 使用裝置: {device} | 基礎模型: {args.model}")
    print(f"🔧 參數: imgsz={args.imgsz} batch={args.batch} lr0={args.lr0} seed={args.seed} "
          f"degrees={args.degrees} mixup={args.mixup} copy_paste={args.copy_paste} "
          f"freeze={args.freeze}")

    # 載入官方 OBB 預訓練模型
    model = YOLO(args.model)

    # 開始端到端 OBB 訓練（4 類：plastic / metal / paper / general_waste）
    model.train(
        data=args.data,
        epochs=args.epochs,
        patience=args.patience,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=8,
        cache=True,
        lr0=args.lr0,
        cos_lr=True,
        seed=args.seed,
        # 資料增強：旋轉對 OBB 任務特別有效（等於免費更多方向資料）
        degrees=args.degrees,
        translate=0.1,
        scale=0.5,
        flipud=0.5,
        fliplr=0.5,
        # 合成資料
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        copy_paste_mode="flip",
        close_mosaic=15,
        # 可選：兩階段微調（先凍結 backbone 只練 head）
        freeze=args.freeze,
        project="garbage_classification_runs",
        name=args.name,
        device=device,
        save=True
    )

    # 從 trainer 直接取得最佳權重路徑並複製到根目錄（避免路徑硬編碼錯誤）
    best_weight = getattr(getattr(model, "trainer", None), "best", None)
    if best_weight and os.path.exists(best_weight):
        print(f"✅ 本場最佳權重: {best_weight}")
        if not args.no_copy_best:
            shutil.copy(best_weight, "best.pt")
            print(f"🎉 訓練完成！最佳權重已複製至: best.pt")
        else:
            print("🎉 訓練完成！（實驗模式：未覆寫根目錄 best.pt）")
    else:
        print("🎉 訓練結束！")


if __name__ == '__main__':
    main()
