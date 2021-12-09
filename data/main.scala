
val movieRDD = sc.textFile("tmdb_5000_movies.csv")

val budgetData = movieRDD.map(_.split(',')).map(x => (x(0), x(17)))

println(budgetData)

# true 是 升序
budgetData.sortByKey(false).take(10).foreach(x => println(x._2 + ": " + x._1))