USE baseball;
#sets baseball as the working database

/*
Ideas:
o career strikeouts / career walks (table pitcher_stat)
o season strikeouts / season walks (table pitcher_stat)
o career strikeouts / career AB - career walks / career AB  (table pitcher_stat)
o (wins/losses) * home_line (table1: game, table2: pregame odds)
o season avg / career avg (table: pitcher_stat)
o pitchesThrown / daysSinceLastPitch (table: pitcher_count)
o walks + hits / innings pitched (table:pitcher_count)
- rolling walks + hits / innings_pitched
o team avg rolling sum players rolling avg / num players
o team batting average annual


*/
#UPDATE pitcher_stat career_bb = career_bb + 1

DROP TABLE IF EXISTS `alt_pitch_stat`; #drops table so I can run the query repeatedly without manually deleting table
CREATE TABLE alt_pitch_stat AS
    SELECT
        ps.game_id,
        ps.player_id,
        ps.team_id,
        ps.season_avg,
        ps.career_avg,
        ps.season_avg / ps.career_avg AS season_over_career_era,
        CASE
            WHEN ps.career_bb = 0
            THEN 0
            ELSE ps.career_so / ps.career_bb
        END AS career_so_over_bb,
        CASE
            WHEN ps.season_bb = 0
            THEN 0
            ELSE ps.season_so / ps.season_bb
        END AS season_so_over_bb,
        (ps.career_so / ps.career_ab) - (ps.career_bb / ps.career_ab) AS diff_so_vs_bb

    FROM pitcher_stat ps
    WHERE ps.career_ab > 0 AND ps.career_avg > 0;


DROP TABLE IF EXISTS `alt_pitch_count`;
CREATE TABLE alt_pitch_count AS
    SELECT
        game_id,
        updatedDate,
        pitcher,
        Hit,
        Walk,
        pitchesThrown,
        DaysSinceLastPitch,
        endingInning + 1 - startingInning AS innings_pitched,
        CASE
            WHEN DaysSinceLastPitch IS NULL THEN pitchesThrown
            WHEN DaysSinceLastPitch = 0 THEN pitchesThrown
            ELSE pitchesThrown / DaysSinceLastPitch
        END AS pitches_over_DSLP,
        (Hit + Walk) / (endingInning + 1 - startingInning) AS whip

    FROM pitcher_counts
    WHERE startingPitcher = 1;

DROP TABLE IF EXISTS `pitch_count_and_stat`;
CREATE TABLE pitch_count_and_stat AS
    SELECT
        apc.game_id,
        apc.pitcher,
        apc.pitchesThrown,
        apc.innings_pitched,
        apc.pitches_over_DSLP,
        apc.whip,
        aps.season_over_career_era,
        aps.career_so_over_bb,
        aps.season_so_over_bb,
        aps.diff_so_vs_bb

    FROM alt_pitch_count apc
        JOIN alt_pitch_stat aps on apc.pitcher = aps.player_id;




DROP TABLE IF EXISTS `rolling_whip`;
CREATE TABLE rolling_whip AS
    SELECT
        a.game_id,
        a.pitcher,
        AVG(a.whip),
        b.whip whip_b
    FROM alt_pitch_count a JOIN alt_pitch_count b ON b.updatedDate BETWEEN date_add(a.updatedDate, INTERVAL -100 DAY) AND a.updatedDate AND a.pitcher = b.pitcher
    GROUP BY a.game_id, a.pitcher;




DROP TABLE IF EXISTS `game_with_outcome`;
CREATE TABLE game_with_outcome AS
    SELECT
        g.game_id,
        g.home_team_id,
        g.home_pitcher,
        g.local_date,
        g.home_w,
        g.home_l,
        b.winner_home_or_away,
        CASE b.winner_home_or_away
            WHEN 'H' THEN 1
            ELSE 0
        END AS home_team_wins

    FROM game g
        JOIN boxscore b ON g.game_id = b.game_id;


DROP TABLE IF EXISTS `game_with_odds`;
CREATE TABLE game_with_odds AS
    SELECT
        g.game_id,
        g.local_date,
        g.home_team_id,
        g.home_pitcher,
        g.home_w,
        g.home_l,
        po.home_line,
        g.home_team_wins,
        CASE
            WHEN g.home_l = 0 THEN 0
            ELSE g.home_w / g.home_l * po.home_line
            END AS wins_loss_odds

    FROM game_with_outcome g
        JOIN pregame_odds po on g.game_id = po.game_id
    GROUP BY g.game_id;


DROP TABLE IF EXISTS `rolling_1`;
CREATE TABLE rolling_1
    SELECT b.game_id
         , Hit
         , atBat
         , local_date
         , batter
    FROM batter_counts b
        JOIN game g on (b.game_id = g.game_id) # makes the join on matching the game_ids
    WHERE atBat > 0;


#This query takes the table made in the previous query and joins it on itself so I can get two dates to compare to each
#other so I can calculate the rolling average.
CREATE INDEX batter_idx ON rolling_1(batter);
CREATE INDEX date_idx ON rolling_1(local_date)


DROP TABLE IF EXISTS `batting_avg_100_day`;
CREATE TABLE batting_avg_100_day AS
    SELECT a.batter
         , a.game_id
         , a.local_date
         , ROUND(SUM(b.Hit)/SUM(b.atBat),3) batting_avg
    FROM rolling_1 a JOIN rolling_1 b ON
    b.local_date BETWEEN date_add(a.local_date, INTERVAL -100 DAY) AND a.local_date AND a.batter = b.batter
    GROUP BY a.batter, a.local_date

DROP TABLE IF EXISTS `add_team_roll`;
CREATE TABLE add_team_roll AS
    SELECT
    bg.batter,
    bg.game_id,
    bg.team_id,
    AVG(ba.batting_avg) team_roll_ba
    FROM batting_avg_100_day ba
        JOIN battersInGame bg ON bg.batter = ba.batter AND bg.game_id = ba.game_id
    GROUP BY game_id, bg.team_id;

#This is very similar to the last query but I joined the game table to get the dates so I
#can group by the year as well as the batter. This makes it an annual batting average.
DROP TABLE IF EXISTS `batting_avg_annual`;
CREATE TABLE batting_avg_annual AS
    SELECT
           b.game_id
           , batter
           , ROUND(SUM(Hit)/SUM(atBat),3) ba_annual
           , YEAR(local_date) year
    FROM batter_counts b
        JOIN game g on (b.game_id = g.game_id) # makes the join on matching the game_ids
    WHERE atBat > 0
    GROUP BY batter, year
    ORDER BY batter, year;

DROP TABLE IF EXISTS `add_team_annual`;
CREATE TABLE add_team_annual AS
    SELECT
    ba.batter,
    ba.game_id,
    bg.team_id,
    AVG(ba.ba_annual) team_annual_ba
    FROM batting_avg_annual ba
        JOIN battersInGame bg ON ba.batter = bg.batter AND ba.game_id = bg.game_id
    GROUP BY game_id, team_id;


DROP TABLE IF EXISTS `join_team_ba_game`;
CREATE TABLE join_team_ba_game AS
    SELECT
        g.game_id,
        #g.local_date,
        year(g.local_date) AS year,
        g.home_team_id,
        g.home_pitcher,
        g.home_w,
        g.home_l,
        g.home_line,
        g.wins_loss_odds,
        ata.team_annual_ba,
        atr.team_roll_ba,
        p.pitchesThrown,
        p.innings_pitched,
        p.pitches_over_DSLP,
        p.whip,
        p.season_over_career_era,
        p.career_so_over_bb,
        p.season_so_over_bb,
        p.diff_so_vs_bb,
        g.home_team_wins

    FROM game_with_odds g
        JOIN add_team_roll atr on g.game_id = atr.game_id AND g.home_team_id = atr.team_id
        JOIN add_team_annual ata on g.game_id = ata.game_id AND g.home_team_id = ata.team_id
        JOIN pitch_count_and_stat p on g.game_id = p.game_id AND g.home_pitcher = p.pitcher;
