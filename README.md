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

  * GridSearchCV其他參數&功能

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
	當數據量過大時，GridSearch會呈指數成長，為了解決這個問題，Bengio 等人在《Random Search for Hyper-Parameter Optimization》論文中提出了RandomSearch的方法。
	
	他們指出大部分參數空間存在 "低有效維度 (low effective dimensionality)" 的特點，即有些參數對目標函數影響較大，另一些則幾乎沒有影響。而且在不同的數據集中通常有效參數也不一樣。 在這種情況下 Random Search 通常效果較好。
	

	* 	RandomSearch為什麼有效?
		下圖是一個例子，其中只有兩個超參數，綠色的超參數影響較大，而黃色的超參數則影響很小：。
		
	![](https://i.imgur.com/cin3B03.png)
	
	GridSearch 會評估每個可能的超參數組合，所以對於影響較大的綠色超參數，GridSearch 只探索了3個值，同時浪費了很多計算在影響小的黃色超參數上； 相比之下 RandomSearch 則探索了9個不同的綠色超參數值，因而效率更高，在相同的時間範圍內 RandomSearch 通常能找到更好的超參數 (當然這並不絕對)。

		（a）目標函數為 f(x,y)=g(x)+h(y)，其中綠色為g(x),黃色為h(y)，目標是求f的最大值。

		（b）由於g(x)數值上要明顯大於h(y)，因此有f(x,y)=g(x)+h(y)≈g(x)，也就是說在整體求解f(x,y)最大值的過程中，g(x)的影響明顯大於h(y)。

		（c）二個圖都進行9次實驗(搜尋)，可以看到左圖實際探索了各三個點(在橫軸和縱軸上的投影均為3個)，而右圖探索了9個不同的點(橫軸縱軸均是，不過實際上橫軸影響更大)。

		（d）右圖更可能找到目標函數的最大值。

		 因此使用隨機搜尋在某些情況下可以提高尋優效率。

	另外，RandomSearch可以在連續的空間搜索，而 GridSearch 則只能在離散空間搜索，而對於像神經網絡中的 learning rate，這樣的連續型參數適合使用連續分布。

	在實際的應用中，GridSearch 只需為每個參數事先指定一個參數列表就可以了，而 RandomSearch 則通常需要為每個參數制定一個Probability distributions，進而從這些分布中進行抽樣


* 	RandomSearch搜尋策略如下：

		* 對於搜尋範圍是distribution的超參數，根據給定的distribution隨機採樣。

		* 對於搜尋範圍是list的超參數，在給定的list中等概率採樣。

		* 對以上二步中得到的n組採樣結果，進行測試。



  	在sklearn已經有提供RandomizedSearchCV的相關功能了，接下來我們會介紹RandomizedSearchCV各個參數的意義，並進行實作。

* 常用參數介紹:
	* estimator：所使用的分類器。
	* param_distributions ：需要進行優化的超參數。
	* scoring :準確度評分標準，默認None，根據所選模型不同，可設置不同的評分標準。
	* cv :交叉驗證參數，默認使用三折交叉驗證。
	* refit :默認為True,程序將會以交叉驗證訓練集得到的最佳參數，重新對所有可用的訓練集進行訓練，作為最終用於性能評估的最佳模型參數。
	* iid:默認True,為True時，默認為各個樣本fold概率分布一致，誤差估計為所有樣本之和，而非各個fold的平均。
	* verbose：日誌冗長度，型態為整數。0：不輸出訓練過程，1：偶爾輸出，數字越大越常輸出。
	* n_jobs: 平行運算數量，型態為整數。個數，-1：跟CPU核數一致, 默認值為1(一核)。
	* pre_dispatch：指定總共分發的並行任務數。當n_jobs大於1時，數據將在每個運行點進行複制，這可能導致OOM，而設置pre_dispatch參數，則可以預先劃分總共的job數量，使數據最多被複制pre_dispatch次。

* 常用方法

	* fit()：運行RandomizedSearchCV。
	* best_params_：描述了已取得最佳結果的超參數的組合。
	* best_score_：優化過程期間得到的最好的評分。

* RandomizedSearchCV 其他參數 & 功能

  參考網址:https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV


---

3. Bayesian Optimization
    
    to be continued...
