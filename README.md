# 自動調參工具
常見的自動調參工具有 Grid Search、Random Search 和 Bayesian Optimization，以下會分別介紹:

1. GridSearch
GridSearch是一個暴力搜尋法(窮舉搜尋法)，他會嘗試所有超參數的候選項，從中找出表現最好的一組超參數當作最終的結果。

	比如我們有二個超參數，超參數A有三種可能，B則有四種可能，如下所示:

		A : { 30, 60, 90 }
		B : { 0.01, 0.04, 0.06, 0.08 }

	二個超參數經過排列組合後，總共會產生12種組合，把所有可能列出來，可以表示成一個3*4的表格，其中每個cell就是一個網格，搜尋的過程就像是跑過所有網格，所以才叫GridSearch。

	![](https://i.imgur.com/ZDpaG0A.png)
	
	由於GridSearch會嘗試所有的可能，所以比較容易找出全域最優解(最好的超參數)，但相對的這也需要耗時大量的計算時間。如果有k個超參數，每個超參數有m個候選項，那我們就有k^m個組合要搜尋，所以說雖然效果不錯，但是時間的代價是非常大的!
	

---

  在sklearn已經有提供GridSearch的相關功能，在這裡我們會介紹GridSearchCV各個參數的意義，並進行實作。

  * 常用參數介紹
    * estimator：所使用的分類器。
    * param_grid：需要進行優化的超參數。
    * scoring :準確度評分標準，默認None，根據所選模型不同，可設置不同的評分標準。
    * cv :交叉驗證參數，默認使用三折交叉驗證。
    * refit :默認為True,程序將會以交叉驗證訓練集得到的最佳參數，重新對所有可用的訓練集進行訓練，作為最終用於性能評估的最佳模型參數。
    * iid:默認True,為True時，默認為各個樣本fold概率分布一致，誤差估計為所有樣本之和，而非各個fold的平均。
    * verbose：日誌冗長度，型態為整數。0：不輸出訓練過程，1：偶爾輸出，數字越大越常輸出。
    * n_jobs: 平行運算數量，型態為整數。個數，-1：跟CPU核數一致, 默認值為1(一核)。
    * pre_dispatch：指定總共分發的並行任務數。當n_jobs大於1時，數據將在每個運行點進行複制，這可能導致OOM，而設置pre_dispatch參數，則可以預先劃分總共的job數量，使數據最多被複制pre_dispatch次。

  * 常用方法
    * grid.fit()：運行GridSearch。
    * grid_scores_：給出每一組參數的評估結果。
    * best_params_：描述了已取得最佳結果的超參數的組合。
    * best_score_：優化過程期間得到的最好的評分。

  * scoring參數選擇

    参考網址：http://scikit-learn.org/stable/modules/model_evaluation.html

  * GridSearch其他參數&功能

    參考網址:https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html

---
  * Python實作
  
    * Dataset : iris
    
    * Model : LightGBM
    
    * 調參工具 : GridSearchCV
    
    * 需要優化的超參數和其候選項:
    
      ![](https://i.imgur.com/AC74nlH.png)
    
    * 最佳超參數組合:
    
      ![](https://i.imgur.com/unLaNdP.png)
    
    * 優化過程中最佳評分 : 
    
      使用 accuracy 當作評分標準(通常越大代表越好)。
    
      ![](https://i.imgur.com/Rnj0XNW.png)
    
    * 使用最佳超參數組合，重新訓練、測試模型，結果如下:
    
      ![](https://i.imgur.com/nctvpuh.png)
      
---

2. RandomSearch
    當數據量過大時，我們可以考慮使用 RandomSearch 來尋找最好的超參數。
    
    to be continued...
