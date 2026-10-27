# 本地新增功能與交接說明

更新日期：2026-10-25
狀態：僅存在本機，尚未 commit／push 到 GitHub。

## 1. 這次工作的核心目標

專案原本只處理垃圾 OBB 偵測與分類。這次新增的主要能力，是從目前照片判斷「現在最安全應先夾哪一個物品」。實際執行採閉環方式：

```text
拍照 → 選出當下第 1 個 → 夾走 → 重新拍照 → 再選下一個
```

因此正式決策以每一輪的 Top-1 安全夾取為主。畫面上的完整 `1、2、3...` 是當前照片的參考計畫，不應取代每夾一次後重新拍照與重算。

## 2. 新增功能

### 2.1 OBB 標註工具修正

- `label_tool.py`
  - 修正 Windows 不分大小寫的 glob 會讓同一張圖片被列出兩次的問題。
  - 圖片清單與自動改名之前先去重。

### 2.2 夾取先後關係標註

- `grasp_order.py`：順序標註格式、物件簽章、關係圖與層級計算。
- `order_label_tool.py`：順序標註 GUI。
- `launch_order_label_tool.bat`：雙擊啟動順序標註 GUI。
- `check_order_labels.py`：檢查缺漏、過期物件簽章、循環關係及審核進度。
- `GRASP_ORDER_LABELING.md`：標註格式與操作說明。

箭頭定義：

```text
上層、需要先夾的物品 → 下層、需要後夾的物品
```

只標直接、確定的遮擋關係。僅靠近、僅接觸、OBB 相交但實際沒有壓住時，不應畫箭頭。

### 2.3 兩兩遮擋關係模型

- `order_pair_data.py`：從 OBB 與關係 JSON 建立兩兩物件訓練資料。
- `order_pair_model.py`：Tiny CNN／MobileNetV3 關係模型及雙向一致性判斷。
- `train_order_model.py`：訓練本地關係模型，支援對稱性損失、負樣本比例、預訓練初始化。
- `evaluate_order_model.py`：評估方向分類、有效箭頭及 closed-loop first-pick safety。
- `render_scene_pairs.py`：輸出場景物件配對預覽。

關係模型分成三類：

```text
A_before_B
B_before_A
no_relation
```

### 2.4 完整順序與閉環推論

- `grasp_sequence.py`：建立無循環關係圖、拓撲層級、風險排序及 `1..N` 顯示。
- `predict_grasp_sequence.py`：對單張圖片執行 OBB 偵測、關係判斷與順序輸出。
- `closed_loop_grasp.py`：相機閉環流程原型。
- `GRASP_SEQUENCE.md`：完整設計、指令及實驗紀錄。

重要語意：

- 黃色箭頭是正式的「先夾 → 後夾」關係。
- 灰色虛線只是模型不確定的診斷資訊，預設隱藏。
- 同一拓撲層內沒有強制先後，數字只是安全性 tie-break。
- 真正執行時只採信當前畫面的第 1 名，夾完必須重新拍照。

### 2.5 圖片測試 GUI

- `image_test_gui.py`
- `launch_image_test_gui.bat`

功能：

- 任意選擇本地圖片。
- 上一張／下一張連續測試。
- 選擇 OBB 與順序模型。
- 調整偵測、順序與風險門檻。
- 在 GUI 顯示 OBB、箭頭、完整數字順序與結果表格。
- 結果另存到 `order_runs/gui_tests/`，不修改原圖。
- 空桌面是正常狀態：顯示「未偵測到物品／桌面已清空」，不再跳出 Python traceback。

啟動方式：

```bat
python image_test_gui.py
```

或雙擊 `launch_image_test_gui.bat`。

目前 closed-loop 安全優先的建議門檻：

```text
偵測門檻：0.55
順序門檻：0.80
風險門檻：0.45
```

若目的是觀察更多候選箭頭，可把順序門檻降至 `0.70`，但錯誤箭頭風險會提高。

### 2.6 Intel RealSense D435 拍照 GUI

- `realsense_capture_gui.py`
- `launch_realsense_capture.bat`

功能：

- 固定使用外接 Intel RealSense D435（曾確認序號 `941322070906`）。
- RGB 預覽為 1280×720、30 FPS。
- `Space` 拍照，`Q`／`Esc` 離開。
- 預設照片存到 `capture_inbox/`，並附 `.capture.json` metadata。

啟動方式：

```bat
python realsense_capture_gui.py
```

若希望新照片直接進入 test：

```bat
python realsense_capture_gui.py --output-dir robotic-garbage-obb\test\images
```

注意：`pyrealsense2==2.59.1.11053` 已安裝在本機 `.venv`，但目前尚未寫入 `requirements.txt`。其他 Windows 電腦需要額外安裝。

### 2.7 外部資料、資料切分與洩漏檢查

- `convert_regrad.py`、`convert_instaorder.py`：外部資料轉換器。
- `external_order_data.py`：外部關係資料讀取。
- `pretrain_external_order.py`：外部資料預訓練。
- `split_external_manifest.py`：外部資料 manifest 切分。
- `scene_split_audit.py`：用視覺 embedding 找相似圖片。
- `build_grouped_dataset.py`：同一場景群組不跨 train／val／test 的衍生資料集建立器。
- `tests/`：關係格式、完整順序與外部轉換器測試。

REGRAD Relation Part 曾下載 super-mini 子集作流程驗證；相關大型檔案在 `external_data/`，不建議直接放入一般 Git commit。

## 3. 目前正式候選模型

### 3.1 OBB 偵測器

```text
best_incremental_v2.pt
```

這是從原本 `best.pt` 使用新增照片低學習率微調後的版本。沒有覆蓋舊 `best.pt`。

在保留原 detector 切分的乾淨 test 上：

| 指標 | 舊 best.pt | best_incremental_v2.pt |
|---|---:|---:|
| Precision | 0.915 | 0.930 |
| Recall | 0.891 | 0.937 |
| mAP@50 | 0.911 | 0.952 |
| mAP@50–95 | 0.799 | 0.839 |

### 3.2 夾取順序模型

```text
order_runs/order_model_grouped_v3_best.pt
```

同一個乾淨 test split 與舊模型比較：

| 指標 | 舊 grouped symmetric | grouped v3 |
|---|---:|---:|
| 三分類 Accuracy | 0.654 | 0.793 |
| Macro-F1 | 0.395 | 0.700 |
| 有效箭頭 Precision | 0.333 | 0.737 |
| 有效箭頭 Recall | 0.056 | 0.259 |
| Closed-loop first-pick safety | 0.988 | 0.988 |
| 完整順序違規數 | 27 | 14 |

目前主要瓶頸仍是有效箭頭 recall 偏低，所以完整箭頭容易漏；但本專案採每夾一次重新拍照，核心評估應以 Top-1 是否安全為主。

## 4. 資料集版本與用途

### `robotic-garbage-obb/`

Hugging Face dataset submodule，也是人工標註的主要來源。資料集的 Git 變更應在 Hugging Face submodule 內獨立處理，再由主專案更新 submodule commit；不要把整包資料直接提交到主 GitHub repository。

### `robotic-garbage-obb-detector-v2/`

OBB 增量訓練／評估用衍生副本。保留原本 detector 的歷史 train／val／test，僅把新照片按場景加入。當時數量為：

```text
train 361 / val 78 / test 78
```

### `robotic-garbage-obb-grouped-v3/`

順序模型的正式增量切分。保留舊 grouped 資料的歷史切分，再加入新照片。當時數量為：

```text
train 360 / val 72 / test 85
```

### `robotic-garbage-obb-grouped-v2/`

實驗用，不應作為正式比較依據。它曾重新分配全部舊圖片，而舊模型可能已看過其中部分圖片，會造成歷史模型與新版 test 之間的資料洩漏。保留它只是為了追蹤實驗，不建議推送或繼續使用。

### 2026-10-25 最新本地狀態

- `test_072.jpg`～`test_077.jpg` 已加入並已有對應 OBB／順序檔。
- `robotic-garbage-obb/test/images/` 另外還有兩張以 `realsense_...jpg` 命名的新照片，尚未經 `label_tool.py` 標準化命名與完整標註。

## 5. 標註與測試流程

先標 OBB：

```bat
python label_tool.py
```

再標直接遮擋關係：

```bat
python order_label_tool.py --split test
```

檢查全部順序標註：

```bat
python check_order_labels.py robotic-garbage-obb --strict
```

測試單張圖片：

```bat
python predict_grasp_sequence.py --image "圖片路徑.jpg" --detector "best_incremental_v2.pt" --order-model "order_runs\order_model_grouped_v3_best.pt" --det-conf 0.55 --relation-conf 0.80 --device cuda
```

## 6. 已知問題與下一步

目前觀察到四個主要錯誤：

1. 真正的遮擋箭頭可能漏掉。
2. 偶爾出現不該存在的箭頭。
3. 被遮擋的下層物件可能漏框。
4. 寶特瓶等長形物件可能被偵測成兩個局部框。

門檻無法同時解決以上問題：降低偵測門檻會增加被遮物召回，但也會增加碎片框；降低順序門檻會增加箭頭 recall，但也會增加錯誤箭頭。

建議後續依序進行：

1. 在低偵測門檻下產生候選，加入保守的「同類、方向接近、中心共線」碎片框合併。
2. 在圖片測試 GUI 增加「漏框、重複框、漏箭頭、錯箭頭、完全正確」回報，建立 hard-case 集合。
3. 用 hard positive／hard negative 重新微調 detector 與關係模型。
4. 將目前正式 Tiny CNN 關係模型升級為 ImageNet 預訓練 MobileNetV3，並重新校正順序門檻。
5. RealSense 拍攝時同步保存對齊 RGB 的 depth，讓深度資訊協助確認當前 Top-1 是否真的在上層。
6. 若模型不確定，應輸出「需要確認／不夾取」，不要強制產生錯誤決策。

完全被遮住、RGB 畫面中沒有任何可見像素的物品，無法從單張照片可靠辨識。這也是必須採「夾第 1 個後重新拍照」閉環策略的原因。

## 7. 推送 GitHub 前的建議

### 建議提交的程式碼與文件

- 上述新增的 `.py`、`.md`、`.bat`。
- `label_tool.py` 的 Windows 圖片去重修正。
- `tests/`。
- 必要的小型設定檔。

### 不建議直接提交到一般 Git commit

- `.venv/`
- `capture_inbox/`
- `runs/`
- `weights/`
- `external_data/` 的資料壓縮檔與影像。
- `robotic-garbage-obb-grouped*/`
- `robotic-garbage-obb-detector-v2/`
- `order_runs/` 裡的模型、逐圖 JSON 與預覽圖。
- `best_incremental_v2.pt`，除非使用 Git LFS、GitHub Release 或另外的模型儲存空間。

資料集應繼續放 Hugging Face；模型權重建議使用 Git LFS、GitHub Release 或 Hugging Face Model repository。主 GitHub repository 只保留程式碼、文件、小型測試資源與下載說明。

`.gitignore` 已涵蓋上述大型輸出目錄，避免誤把資料集、訓練輸出或權重推上 GitHub。

## 8. 分支與資料位置

- 程式碼使用 GitHub 分支：`feature/grasp-order-closed-loop`
- 照片、OBB 標註與夾取順序使用 Hugging Face Dataset 分支：`grasp-order-annotations-v1`
- 兩個分支皆由原本的 `main` 衍生，不會改寫原始 `main`。
- 兩張尚未完整標註的 RealSense 相片放在資料分支的 `unlabeled/`，不會混入正式 `test` 評估集。
