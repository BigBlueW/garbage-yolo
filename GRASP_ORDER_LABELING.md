# 夾取順序標註方法

## 設計原則

既有 YOLO OBB 標籤只負責物件種類、位置和角度，不改變其四個類別。夾取順序另外儲存在：

```text
robotic-garbage-obb/<split>/grasp_order/<image_stem>.json
```

箭頭 `#A → #B` 的唯一意義是：**A 必須先夾走，才能安全夾 B**。只有確定存在壓住、遮擋或取物阻礙時才畫箭頭；只是相鄰、框重疊但實物沒有互相壓住時不要標。

程式會把箭頭轉成可並行的夾取層：

```text
L1: 現在可以先夾
L2: L1 移除後可以夾
L3: L1、L2 移除後可以夾
```

同一層的物件沒有強制先後順序。完全被遮住、照片中看不見的物件無法從單張照片標註，應在上層物件移走並重新拍照後處理。

## 啟動

在專案目錄雙擊 `launch_order_label_tool.bat`，或執行：

```powershell
.\.venv\Scripts\python.exe .\order_label_tool.py
```

## GUI 操作

1. 確認所有彩色 OBB 與編號正確；本工具不會修改 OBB。
2. 左鍵點一下「先夾」的上層物件，再點一下「後夾」的下層物件。
3. 黃色箭頭出現後，檢查方向是否正確。
4. 該圖所有必要箭頭都完成後，按 `M`，畫面會顯示 `REVIEWED`。
5. 按 `D`、空白鍵或 Enter 到下一張；按 `N` 可直接跳到下一張未審核圖片。

快捷鍵：

- `R`：切換新增／移除模式。在移除模式同樣依序點箭頭起點、終點。
- `Z`：復原上一個關係編輯。
- `M`：確認本張已完整審核；即使沒有任何堆疊，也必須按一次。
- `A`／`D`：上一張／下一張。
- `N`：下一張未審核。
- `T`：切換 train、val、test。
- `S`：立即儲存。
- `Q`／Esc：儲存並離開。
- `K`：只有 OBB 已變更而使舊關係失效時，用來清空該張舊關係並重新審核。

重疊框難以選取時，在同一位置連續點擊，工具會在該位置涵蓋的物件間輪換。

## 標註規則

- 看得見的下層物件仍必須有 OBB，不能故意漏標。
- 只標物理上的先後限制，不要為互不相關物件硬排第 1、2、3 名。
- 不確定上下關係時先不要按 `M`，保留為 `UNREVIEWED`，稍後確認。
- 不需畫反向箭頭；`A → B` 已同時代表 B 位於 A 之後。
- 工具會拒絕形成循環的箭頭，例如 `A → B → C → A`。

## 檢查進度

```powershell
.\.venv\Scripts\python.exe .\check_order_labels.py
```

完成全部資料後可使用嚴格模式：

```powershell
.\.venv\Scripts\python.exe .\check_order_labels.py --strict
```

後續訓練應以 `before / after / no_relation` 的物件配對分類建立關係圖，再用拓撲排序產生 L1、L2、L3。主要評估指標應是 Pairwise Macro-F1、Top-1 Pick Success Rate、Unsafe Pick Rate 與 Cycle Rate，不應只看 Accuracy。
