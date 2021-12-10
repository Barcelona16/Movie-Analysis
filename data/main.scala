
val movieRDD = sc.textFile("tmdb_5000_movies.csv")



val budgetData = movieRDD.map(_.split(",(?=([^\\\"]*\\\"[^\\\"]*\\\")*[^\\\"]*$)",-1)).map(x => (x(0), x(6)))
val languageData = movieRDD.map(_.split(",(?=([^\\\"]*\\\"[^\\\"]*\\\")*[^\\\"]*$)",-1)).map(x => (x(5), x(6)))
val popularityData = movieRDD.map(_.split(",(?=([^\\\"]*\\\"[^\\\"]*\\\")*[^\\\"]*$)",-1)).map(x => (x(8), x(6)))
val revenueData = movieRDD.map(_.split(",(?=([^\\\"]*\\\"[^\\\"]*\\\")*[^\\\"]*$)",-1)).map(x => (x(12), x(6)))
val vote_averageData = movieRDD.map(_.split(",(?=([^\\\"]*\\\"[^\\\"]*\\\")*[^\\\"]*$)",-1)).map(x => (x(18), x(6)))


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

// println("The top revenue 10 movies")
// revenueData.sortByKey(false).take(10).foreach(x => println(x._2 + ": " + x._1))

println("The top popularity 10 movies")
popularityData.sortByKey(false).take(15).foreach(x => println(x._2 + ": " + x._1))

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
// println("The top votes_average 10 movies")
// vote_averageData.sortByKey(false).take(10).foreach(x => println(x._2 + ": " + x._1))





