USE baseball;
#sets baseball as the current database


#This calculates the batting average for every player by averaging each of their batting averages
#calculated for each game.

DROP TABLE IF EXISTS `batting_avg`; #drops table so I can run the query repeatedly without manually deleting table
CREATE TABLE batting_avg
  AS (SELECT batter, ROUND(AVG(Hit/atBat),3) ba #ba is batting average
    FROM batter_counts
    WHERE atBat > 0 #makes sure that the player has at least 1 atbat to fix divide by 0 error
    GROUP BY batter #groups each batter ID together to get an average for each player
    );

#This is very similar to the last query but I joined the game table to get the dates so I
#can group by the year as well as the batter. This makes it an annual batting average.
DROP TABLE IF EXISTS `batting_avg_annual`;
CREATE TABLE batting_avg_annual
  AS (SELECT batter, ROUND(AVG(Hit/atBat),3) ba, YEAR(local_date) year
    FROM batter_counts b JOIN game g on (b.game_id = g.game_id) # makes the join on matching the game_ids
    WHERE atBat > 0
    GROUP BY batter, year
    ORDER BY batter, year
    );


#This query was a challenge since I needed to make a table first that combines the batter counts and game to get all
#the important information.
DROP TABLE IF EXISTS `rolling_1`;
CREATE TABLE rolling_1
    SELECT b.game_id , Hit, atBat, local_date, batter
    FROM batter_counts b JOIN game g on (b.game_id = g.game_id) # makes the join on matching the game_ids
    WHERE atBat > 0;

#This query takes the table made in the previous query and joins it on itself so I can get two dates to compare to each
#other so I can calculate the rolling average. 
DROP TABLE IF EXISTS `batting_avg_100_day`;
CREATE TABLE batting_avg_100_day AS
    SELECT r1.batter, r1.game_id, r1.local_date, AVG(r2.Hit/r2.atBat)
    FROM rolling_1 r1, rolling_1 r2
    WHERE r2.local_date between date_add(r1.local_date, INTERVAL -100 DAY) and r1.local_date
    GROUP BY r1.batter;

