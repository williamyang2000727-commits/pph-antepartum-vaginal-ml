# 陰道分娩產後出血預測：投稿實驗程式

對應論文：**An interpretable algorithmic risk prediction model for postpartum hemorrhage prior to delivery using antepartum features in vaginal delivery**

作者：Yung-Cheng Yang、Lan-Ying Huang、Yen-Wei Chu。PLOS Digital Health 投稿研究，2026。
資料來自臺中榮民總醫院，IRB CE25412A。

## 目前投稿配置

XGBoost（100 棵樹）＋KNN k=7＋StandardScaler＋SMOTEENN＋Top30。
`submission_config.json` 保存設定、35 個特徵的參考選中次數及投稿指標；參考值來自現行實驗CSV，僅作核對，不參與模型擬合。

測試集154人、37個事件：TP29、FP39、FN8、TN78；sensitivity 0.7838、specificity 0.6667、AUC 0.7404、MCC 0.3876。
最終配置是以全部2,646個候選的測試集分數選出，該測試集亦用於報告效能，因此存在選擇樂觀偏誤。這不是完全未參與模型選擇的外部驗證。
同配置的CV AUC為0.6266。這兩個數值不構成真實外部效能的保證上下界。

## 安裝

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

原研究使用Python 3.13.5。數值重現也受SciPy與BLAS建置影響，版本相同不保證所有最近鄰平手處理一致。`--verify`會在結果不符時失敗，不會更改參考值來通過。

## 重現目前最終模型

取得經授權的`all_features.csv`後：

```bash
python 04_train_final_model.py --data-dir /absolute/path/to/02_experiment_csv --output-dir /absolute/path/to/private-results --verify
```

這會重新執行615/154的分層切分、訓練集單變量篩選、100次bootstrap乘4個C值的L1特徵選擇、KNN7填補、標準化、SMOTEENN與XGBoost，核對全部35個選擇次數及最終指標。`05_final_model_evaluation.py`使用同一實作，不會重新跑舊RandomForest基準。

`--verify`只適用於原研究資料。使用其他符合格式的資料時省略此選項，報告會明確標示未核對投稿值。
输出包含`reproduction_report.json`、`bootstrap_selection_frequency_clean.csv`與原生`final_model_xgboost_native.json`；不輸出病人識別碼、逐人預測或包含訓練資料的KNN填補器。

選擇頻率並列時，`--verify`先核對所有35個選中次數，再採已發表的並列次序，使30特徵與已發布模型一致。不符合參考頻率時直接失敗。

## 各入口用途

| 檔案 | 用途 |
|---|---|
| `01_create_labels.py` | 從醫院原始表建立陰道分娩族群與PPH標籤 |
| `02_feature_engineering.py` | 46項測量乘6統計量，加年齡與27二元特徵，建立304候選特徵 |
| `03_feature_selection.py` | 與目前最終模型相同的train-only兩階段選擇，不再使用舊60%門檻版本 |
| `04_train_final_model.py` | 目前XGBoost＋KNN7完整訓練與指標驗證 |
| `05_final_model_evaluation.py` | 同一最終模型的重訓評估入口 |
| `submission_pipeline.py` | 03、04、05共用實作 |
| `submission_config.json` | 已發表配置與彙總參考數值 |
| `06_joint_search.py` | 7填補×9演算法×7不平衡×6特徵子集，10-fold CV |
| `07_full_test_eval.py` | 在完整train重新訓練全部2,646候選，再評估test |
| `08_test_eval_top10.py` | CV前10候選的補充比較，不負責選出投稿最終模型 |

早期03、RandomForest版04與05保留於Git歷史。現行入口不再執行那些版本。

## 原始資料至搜尋

01、02仍依原始研究的檔名從目前工作目錄讀寫資料；這兩個腳本中的原始CSV檔名是資料介面，請先查看檔案開頭並提供相同結構。所有醫院資料應放在私人位置。
02產生`all_features.csv`後，放入資料根目錄的`02_experiment_csv/`。

搜尋06、07、08仍從腳本位置向上尋找含`02_experiment_csv/`的資料根目錄。若在本倉庫重跑，請在本倉庫建立該資料目錄，再先執行：

```bash
python 03_feature_selection.py --data-dir ./02_experiment_csv --output-dir ./02_experiment_csv --verify
python 06_joint_search.py
python 07_full_test_eval.py
```

03輸出的檔名正是06、07所讀的`bootstrap_selection_frequency_clean.csv`。輸出目錄請使用新目錄，或先備份舊產物。

06的CV排序是三個指標百分位排名的平均；07的test排序是`0.4*MCC + 0.3*AUC + 0.3*sensitivity`。以後者公式重算CV時，最終配置排名522；06的百分位排序是另一個數值，不能混用。
CV內填補、標準化與重採樣逐折擬合；兩階段特徵選擇則在完整train做一次，沒有在各折重做。這是既有實驗設計，應保留其限制。

## 模型JSON與資料保護

`final_model_xgboost_top30.json`是已發布的自訂樹格式，不是XGBoost原生模型格式。它包括樹、特徵顺序與標準化常數，推論使用`sigmoid(base_margin + sum(leaves))`，節點比較需float32。
其中`imputation_values`是訓練集的中位數替代值，**不是完整KNN7填補器**。要重現論文指標必須用經授權的訓練資料擬合KNN7；不能拿中位數替代推論的結果聲稱等於論文模型。

本倉庫不提供四份醫院原始CSV、`all_features.csv`、病人列資料或識別碼。資料須向通訊作者申請並經機構審查核准。不要將私人資料或訓練填補器推送至公開倉庫。
目前共用入口重現特徵選擇及最終模型的分類、AUC、Brier與average precision；完整圖表、消融與校準延伸分析仍位於作者分析專案，不宣稱本入口已涵蓋全部論文分析。

## 授權與聯絡

程式採MIT授權，見LICENSE。通訊作者：Yen-Wei Chu（ywchu@nchu.edu.tw）。研究生：Yung-Cheng Yang（william.yang2000727@gmail.com）。
