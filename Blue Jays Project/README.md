Problem: Baseball is a sport that is full of statistics on player performances, and as a sport, you would want a team that wins. Teams make trades regularly throughout the year to get the best player or players that could give their team that edge over other teams. Pitchers being one of the most important positions, are highly desired based on how well they can get batters out. My goal is to build a predictive model using pitching data (includes data on velocity, spin rate, position, etc) to predict the likelihood of a batter getting out based on how the ball is pitched.

Client: The client in this scenario are general managers and coaches of baseball teams who are in charge of putting together a roster of pitchers. They will be interested in either drafting, trading, or pulling pitchers from their reserves based on how well they can get batters out. With a model that can quantify their skill, they will be able to obtain hidden talent at a cheap cost.

Approach: I will be using supervised learning classification algorithms that will predict the result of a pitch. The response variable will be the event of a pitch, and the predictors will be everything else from velocity in the x,y,z direction to the spin rate of the ball. I will be using 70% of the provided dataset as my training data, and the rest for my test data.

Deliverables: The deliverables will be an interpretable model that will allow the client to identify successful traits when looking to improve their pitching roster.

Dataset:
https://baseballsavant.mlb.com/statcast_search?hfPT=&hfAB=&hfBBT=&hfPR=&hfZ=&stadium=&hfBBL=&hfNewZones=&hfGT=R%7C&hfC=&hfSea=2018%7C&hfSit=&player_type=pitcher&hfOuts=&opponent=&pitcher_throws=&batter_stands=&hfSA=&game_date_gt=&game_date_lt=&hfInfield=&team=TOR&position=&hfOutfield=&hfRO=&home_road=&hfFlag=&hfPull=&metric_1=&hfInn=&min_pitches=0&min_results=0&group_by=team&sort_col=pitches&player_event_sort=h_launch_speed&sort_order=desc&min_pas=0#results
