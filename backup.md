# 数据集

数据集选用来自<a href= 'https://www.themoviedb.org/'> The Movie Database(TMDB)</a>的7000+电影信息，每一条电影信息包括电影id、预算、票房、观众喜爱程度、时长、评分等信息，相关的信息可在<a href='https://www.kaggle.com/c/tmdb-box-office-prediction/data'>Kaggle TMDB Box Office Prediction</a> 中查看。
```scla
scala> movieRDD.take(1).foreach{println}
budget,genres,homepage,id,keywords,original_language,original_title,overview,popularity,production_companies,production_countries,release_date,revenue,runtime,spoken_languages,status,tagline,title,vote_average,vote_count
```

# 分析目的

通过分析电影不同特征之间的关系了解受欢迎的电影之间共有的特征，比如输出电影的预算、时长、评分等特征和电影最终票房的关系，以及进一步可进行电影的票房预测等任务。

# 解决思路

将数据读取，绘制相关图


# 实验完成情况

* 调用spark相关接口读入电影数据并进行分析
* 使用matplotlib将分析到的数据可视化
* 使用基于XGBOOST、CatBoost、lightGBM的算法对票房进行预测，将结果提交到kaggle任务，score为3.79。

# 数据概况

数据格式为CSV，首先调用*textFile*将CSV文件读入，再通过正则表达式将不同特征分离，这里以逗号为分隔符，并且忽略引号内的逗号。
```scala
val movieRDD = sc.textFile("tmdb_5000_movies.csv")


// lamda 匹配的是， 忽略”“内的，
val budgetData = movieRDD.map(_.split(",(?=([^\\\"]*\\\"[^\\\"]*\\\")*[^\\\"]*$)",-1)).map(x => (x(0), x(6)))
val languageData = movieRDD.map(_.split(",(?=([^\\\"]*\\\"[^\\\"]*\\\")*[^\\\"]*$)",-1)).map(x => (x(5), x(6)))
val popularityData = movieRDD.map(_.split(",(?=([^\\\"]*\\\"[^\\\"]*\\\")*[^\\\"]*$)",-1)).map(x => (x(8), x(6)))
val revenueData = movieRDD.map(_.split(",(?=([^\\\"]*\\\"[^\\\"]*\\\")*[^\\\"]*$)",-1)).map(x => (x(12), x(6)))
val vote_averageData = movieRDD.map(_.split(",(?=([^\\\"]*\\\"[^\\\"]*\\\")*[^\\\"]*$)",-1)).map(x => (x(18), x(6)))
```
这样的带的数据格式就为(id,feature)，可以后续对这些特征进行组合分析。

# 可视化分析

## 最流行&&最盈利

通过将对应的属性进行简单排序即可得到。
```scala
println("The top popularity 10 movies")
popularityData.sortByKey(false).take(15).foreach(x => println(x._2 + ": " + x._1))

// println("The top votes_average 10 movies")
// vote_averageData.sortByKey(false).take(10).foreach(x => println(x._2 + ": " + x._1))

// println("The top revenue 10 movies")
// revenueData.sortByKey(false).take(10).foreach(x => println(x._2 + ": " + x._1))

/*
The top popularity 10 movies
The Twilight Saga: Breaking Dawn - Part 2: 99.687084
Ice Age: 99.561972
Thor: The Dark World: 99.499595
Man of Steel: 99.398009
Harry Potter and the Half-Blood Prince: 98.885637
Fifty Shades of Grey: 98.755657
12 Years a Slave: 95.9229
"I, Robot": 95.914473
Gladiator: 95.301296
Ex Machina: 95.130041
The Wolf of Wall Street: 95.007934
*/

println("The top budgets 10 movies")
budgetData.sortByKey(false).take(10).foreach(x => println(x._2 + ": " + x._1))

/*
The top budgets 10 movies
original_title: budget                                                          
The Peanuts Movie: 99000000
The Mummy Returns: 98000000
Cutthroat Island: 98000000
Astérix aux Jeux Olympiques: 97250400
Four Lions: 967686
The Lost City: 9600000
The Road to El Dorado: 95000000
Ice Age: Continental Drift: 95000000
Cinderella: 95000000
*/

```
## 票房与预算、欢迎程度等特征关系

直觉上来看，一般投入预算较大的电影往往能带来较大的收益，受欢迎的电影也会拥有很大的票房，所以首先对这几个特征进行可视化分析进行验证。
![image](uploads/4f2f6120ba2af864ffce4e9dc05fa72b/image.png)

可以看到票房和投入有比较强的正向关系。
![image](uploads/bd0d1e8602e557139cb8894dcea8bd10/image.png)

可以看到票房和流行度有比较强的正向关系。
![image](uploads/12b2f58757f1f8c1d0e92520839ba4d4/image.png)

可以看到票房和戏剧性程度有比较强的正向关系。

![image](uploads/50e182d7a22649a2fc4c4daa8abc296e/image.png)

具体看这几个特征之间的关系可以发现，这几个特征相对于其他一些无关紧要的特征确实有更强的联系。

## 一些玄学可视化

电影常常会有大小年，有些年度可能少有佳片，但是有的年份可能好片频出，比如1998年，所以我还研究了不同年份的电影预算的分布。
![image](uploads/786e8e10de0c3a417fac84b599692d42/image.png)

可以看到近年来电影的预算分布更为广，高预算和低预算电影并存。
![image](uploads/427b3826378ff5ed48f62199c4d3af5c/image.png)
* 1998年仅从预算上来看，也可以看出是电影井喷的一年。

以及不同语种的电影预算分布。
![image](uploads/2c0ac26c2b162ab88c98c50bb18109cd/image.png)

# 电影票房预测

预测算法采用经典的XGBoost、lightGBM、CatBoost进行集成预测，只进行了简单的调试训练。
* XGBoost
```
params = {'objective': 'reg:linear',
              'eta': 0.01,
              'max_depth': 6,
              'subsample': 0.6,
              'colsample_bytree': 0.7,
              'eval_metric': 'rmse',
              'seed': random_seed,
              'silent': True,
              }
```
* lgb
```
params = {'objective': 'regression',
              'num_leaves': 30,
              'min_data_in_leaf': 20,
              'max_depth': 9,
              'learning_rate': 0.004,
              # 'min_child_samples':100,
              'feature_fraction': 0.9,
              "bagging_freq": 1,
              "bagging_fraction": 0.9,
              'lambda_l1': 0.2,
              "bagging_seed": random_seed,
              "metric": 'rmse',
              # 'subsample':.8,
              # 'colsample_bytree':.9,
              "random_state": random_seed,
              "verbosity": -1}
```
* cat
```
model = CatBoostRegressor(iterations=100000,
                              learning_rate=0.004,
                              depth=5,
                              eval_metric='RMSE',
                              colsample_bylevel=0.8,
                              random_seed=random_seed,
                              bagging_temperature=0.2,
                              metric_period=None,
                              early_stopping_rounds=200)
```
对于每个model的预测结果加上相应的权重（0.2，0.4，0.4），得到最终的预测结果提交到kaagle。

![image](uploads/eee6ecaabc7f7e628ede67228e073459/image.png)




