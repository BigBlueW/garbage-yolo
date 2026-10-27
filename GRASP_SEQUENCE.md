# 夾取順序：本地流程

這套流程將現有 OBB 與「先夾 → 後夾」箭頭轉成三類兩兩關係模型：

- `A_before_B`
- `B_before_A`
- `no_relation`

模型的關係會先經過循環移除，再做拓撲排序。輸出包含：

1. `NEXT GRASP`：閉環系統真正該執行的下一夾。
2. `COMPLETE CURRENT-FRAME PLAN`：目前照片的完整 1～N 順序。

同一 layer 代表沒有標註出誰必須先夾，程式才用偵測信心、離邊界距離、與其他物件的距離決定顯示次序。這個數字只是平手排序，不是假造遮擋關係。

推論使用可見 OBB 多邊形的混合候選規則：相交者正常判斷；未相交但間距不超過影像對角線 `0.7%` 者保留，以支援「實際被遮擋、但只框可見部分」的標法，並要求更高的 `0.90` 信心；更遠者不建立關係。每組候選还必須通過 A/B 與 B/A 雙向一致性，才會畫箭頭。

無法通過雙向一致性或信心不足者會寫入 JSON 的 `uncertain_pairs`，並在預覽圖以灰色虛線與 `?` 表示；它不會成為限制夾取順序的箭頭。

## 1. 先驗證現有人工標註能否輸出完整順序

在專案目錄且 `.venv` 已啟用時：

```bat
python predict_grasp_sequence.py --annotated test_001.png --split test
```

會輸出終端順序，並在 `order_runs/predictions/` 產生編號預覽圖。這一步不使用 AI 模型，用來檢查排序與編號邏輯；預覽圖不會混入資料集。

若要保留人工 OBB、但真正測試順序模型（不讀人工箭頭），請使用：

```bat
python predict_grasp_sequence.py --labeled-image train_239.png --split test --dataset robotic-garbage-obb-grouped --order-model order_runs\order_model_grouped_symmetric_best.pt --device cuda
```

請勿把 `--annotated` 的結果當成模型準確率；該模式顯示的是人工答案。

## 2. 訓練兩兩遮擋關係模型

```bat
python train_order_model.py
```

CPU 可以執行但較慢；MobileNetV3 最佳權重預設儲存在：

```text
order_runs/order_model_mobilenet_best.pt
```

本次本地實驗的 MobileNetV3 驗證集 `macro-F1` 約為 `0.869`，但獨立 test 的
`macro-F1` 只有約 `0.503`，而且兩個先後類別召回率偏低。因此日常推論目前仍預設使用
較安全的 `order_runs/order_model_best.pt`。MobileNetV3 保留下來作為外部資料預訓練與
後續微調架構，不因單一驗證集成績較高就直接升級。

## 3. 在 test 評估

```bat
python evaluate_order_model.py --split test
```

優先看：

- `macro_f1`：三類關係是否平均學好。
- `per_class`：尤其 `A_before_B`、`B_before_A` 的 recall。
- `closed_loop_first_pick_safety`：第一夾是否為標註中沒有被其他物件壓住的物件。這比只看 accuracy 更符合機器手臂目標。

## 4. 對新照片輸出完整順序

```bat
python predict_grasp_sequence.py --image C:\path\to\photo.jpg --detector best.pt --order-model order_runs\order_model_best.pt --json result.json
```

只輸出下一夾：

```bat
python predict_grasp_sequence.py --image C:\path\to\new_photo.jpg --next-only
```

機器人每夾一次後重新拍照，再執行一次 `--next-only`。完整 1～N 是當下畫面的規劃；實際控制應永遠以重拍後新的 `NEXT GRASP` 為準。

## 5. 相機閉環模式

```bat
python closed_loop_grasp.py --camera 0
```

- `Space`：拍攝目前畫面並重新輸出 `NEXT` 與完整 1～N。
- 完成一夾後，再按一次 `Space`，不沿用舊畫面的剩餘順序。
- `Q` 或 `Esc`：離開。
- 最新結果同步寫到 `order_runs/live/latest.jpg` 與 `latest.json`，之後可讓機器手臂控制程式讀取 JSON。

## 資料量限制

目前真正含上下關係的影像仍少。沒有關係的圖片能教 `no_relation`，卻不能替代遮擋正負例。若驗證集的前後類別 recall 不穩定，下一步應優先增加「確實重疊、方向清楚」的場景，且以整組場景分割 train/val/test，避免相近照片洩漏。

## 外部資料預訓練

程式已支援 REGRAD Relation Part V2 與 InstaOrder。外部資料不會改寫你的資料集，轉換器只建立 JSONL 索引。

REGRAD 解壓後，分別轉換官方 train 與 seen_valid：

```bat
python convert_regrad.py --source "external_data\REGRAD\Relation Part\train" --output external_data\regrad_train.jsonl
python convert_regrad.py --source "external_data\REGRAD\Relation Part\seen_valid" --output external_data\regrad_val.jsonl
python pretrain_external_order.py --train-manifest external_data\regrad_train.jsonl --val-manifest external_data\regrad_val.jsonl --device cuda
```

再用你的人工標註微調（不沿用外部資料的 optimizer）：

```bat
python train_order_model.py --architecture mobilenet_v3_small --init-checkpoint order_runs\order_model_regrad_pretrained.pt --output order_runs\order_model_regrad_finetuned.pt --device cuda
python evaluate_order_model.py --model order_runs\order_model_regrad_finetuned.pt --split test --device cuda
```

只有當 `A_before_B`、`B_before_A` 的 recall、`macro_f1` 與
`closed_loop_first_pick_safety` 都不低於目前模型時，才把新模型傳給 `--order-model`。

InstaOrder 需要官方 InstaOrder JSON、COCO instances JSON 與對應 COCO 圖片：

```bat
python convert_instaorder.py --instaorder-json external_data\COCO\annotations\InstaOrder_train2017.json --coco-json external_data\COCO\annotations\instances_train2017.json --images external_data\COCO\train2017 --output external_data\instaorder_train.jsonl
```

InstaOrder 標的是遮擋順序，不完全等同機器手臂的安全夾取順序；它只能當輔助預訓練或教師來源，最後一定要再用本地人工關係微調與 test 驗證。互相遮擋的 `A & B` 標籤會被轉換器捨棄，不強迫模型學出循環。

若要串接兩個外部來源，先用 InstaOrder 得到 checkpoint，再把它傳給 REGRAD 預訓練：

```bat
python pretrain_external_order.py --train-manifest external_data\regrad_train.jsonl --val-manifest external_data\regrad_val.jsonl --init-checkpoint order_runs\order_model_instaorder_pretrained.pt --output order_runs\order_model_regrad_pretrained.pt --device cuda
```

### 2026-10-20 REGRAD supermini 實驗

- 官方 `train_supermini`：50 個場景、450 張影像。
- 依場景切分：40 train / 10 val，場景重疊為 0。
- 轉換結果：5,072 個雙向 pair。
- REGRAD 預訓練最佳 val Macro-F1：約 `0.525`。
- 本地人工資料微調最佳 val Macro-F1：約 `0.886`。
- 獨立本地 test：accuracy `0.750`、Macro-F1 `0.586`、第一夾安全率 `1.0`。
- `A_before_B` recall `0.571`，`B_before_A` recall `0.214`。

結論：REGRAD 預訓練比只用本地資料的 MobileNet（test Macro-F1 約 `0.503`）好，
但仍不及目前正式小型模型（test Macro-F1 約 `0.698`，兩個方向 recall 約
`0.714 / 0.643`）。因此保留
`order_runs/order_model_regrad_supermini_finetuned.pt` 作研究權重，不升級為預設模型。

### 2026-10-20 場景分組與方向對稱實驗

原始 pair 數量其實已經平衡（`A_before_B=71`、`B_before_A=71`）；問題主要是同一對
物件交換 A/B 後，模型預測不一定一致。因此訓練器現在會同時學習 AB 與 BA，並用
`--symmetry-weight` 約束兩者機率一致。預設值為 `0.2`。

場景稽核與建立不覆寫原資料的新切分：

```bat
python scene_split_audit.py --dataset robotic-garbage-obb --output order_runs\scene_split_audit_grouping.json --threshold 0.965 --all-pairs
python render_scene_pairs.py --report order_runs\scene_split_audit_grouping.json --output order_runs\scene_split_pairs.jpg
python build_grouped_dataset.py --source robotic-garbage-obb --destination robotic-garbage-obb-grouped --report order_runs\scene_split_audit_grouping.json --threshold 0.965
python check_order_labels.py --dataset robotic-garbage-obb-grouped
```

`robotic-garbage-obb-grouped` 共 472 張：train 329、val 65、test 78；相似群組不跨
split，472 張都已審核、共 89 條關係、0 個標註錯誤。原始
`robotic-garbage-obb` 完全保留。

公平比較（兩個模型都只使用新的 grouped train 訓練）：

- 無對稱損失：test Macro-F1 `0.836`。
- 對稱損失 `0.2`：test Macro-F1 `0.847`；A/B recall 都是 `0.70`。
- 對稱模型第一夾安全率 `0.987`（76/77）；唯一不安全圖片是 `train_239.png`。

對稱模型存為 `order_runs/order_model_grouped_symmetric_best.pt`，無對稱對照組存為
`order_runs/order_model_grouped_control_best.pt`。目前仍是實驗權重，沒有取代正式
`order_model_best.pt`。把關係門檻降到 `0.50` 雖可令這個 test 的第一夾安全率變成
`1.0`，但錯誤關係邊會由 6 條增加到 11 條，因此不可只靠降門檻升級。

可輸出逐圖診斷與關係圖 precision/recall/F1：

```bat
python evaluate_order_model.py --dataset robotic-garbage-obb-grouped --model order_runs\order_model_grouped_symmetric_best.pt --split test --device cuda --details-json order_runs\grouped_symmetric_test_details.json
```

### 2026-10-20 不確定關係風險排序

低於正式關係門檻、但 AB／BA 方向一致且信心至少 `0.45` 的判斷，不會變成正式箭頭；
它只會在同一拓撲 layer 中降低疑似下層物件的排序優先度。方向不一致或低於 `0.45`
者不參與風險排序。正式關係、錯誤箭頭數與模型權重完全不變。

以 grouped validation 選定 `0.45` 後，grouped test 結果：

- 第一夾安全率：`0.987` → `1.000`（76/77 → 77/77）。
- 完整順序合規率：`0.961` → `0.987`（74/77 → 76/77）。
- 完整順序違規：5 條 → 3 條。
- 正式關係邊的 TP／FP／FN 不變，沒有靠增加箭頭取得改善。

可用 `--no-uncertain-risk` 關閉此功能作對照，或用
`--uncertain-risk-floor` 明確指定下限。剩餘 3 條違規都集中在 grouped test 的
`train_024.png`，是下一批應優先補充的困難場景。

灰色虛線只是診斷用的不確定配對，不是順序箭頭。正常預覽現在預設隱藏它們，資訊仍
完整寫入 JSON；除錯時才加 `--show-uncertain` 顯示。正式黃色箭頭始終代表模型接受的
關係。

針對 `train_024.png` 另測了 `negative-ratio=4`、全部 1,504 個 hard negatives、以及
全負例池的每輪平衡抽樣。它們雖減少錯誤線，卻使正關係 recall 降到 `0.30/0.50`，
正式圖關係 recall 最低只剩 `0.10`；因此全部保留為研究權重，不取代
`order_model_grouped_symmetric_best.pt`。這證明目前瓶頸是密集疊放正例不足，不能只靠
提高負例數解決。
