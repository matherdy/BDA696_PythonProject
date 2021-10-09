import sys

from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.sql.functions import round
from transform import CalcBaTransform


def main():
    user = input("Enter Username: ")
    pw = input("Enter Password: ")
    spark = SparkSession.builder.master("local[*]").getOrCreate()

    batter_counts_df = (
        spark.read.format("jdbc")
        .option("url", "jdbc:mysql://localhost:3306/baseball")
        .option("driver", "org.mariadb.jdbc.Driver")
        .option("dbtable", "batter_counts")
        .option("user", user)
        .option("password", pw)
        .load()
    )
    batter_counts_df.createOrReplaceTempView("batter_counts")
    batter_counts_df.persist(StorageLevel.DISK_ONLY)

    game_df = (
        spark.read.format("jdbc")
        .option("url", "jdbc:mysql://localhost:3306/baseball")
        .option("driver", "org.mariadb.jdbc.Driver")
        .option("dbtable", "game")
        .option("user", user)
        .option("password", pw)
        .load()
    )
    game_df.createOrReplaceTempView("game")
    game_df.persist(StorageLevel.DISK_ONLY)

    rolling_1_df = spark.sql(
        """
    SELECT
             b.game_id
             , Hit
             , atBat
             , local_date
             , batter
         FROM batter_counts b
         JOIN game g on (b.game_id = g.game_id)
         WHERE atBat > 0
         ORDER BY batter, local_date
    """
    )
    rolling_1_df.createOrReplaceTempView("rolling_1")
    rolling_1_df.persist(StorageLevel.DISK_ONLY)

    # rolling_1_df.show()

    ba_rolling_100_df = spark.sql(
        """
        SELECT
            a.batter
            , a.game_id
            , a.local_date
            , SUM(b.Hit) AS tot_hit
            , SUM(b.atBat) AS tot_atbat
        FROM rolling_1 a
        JOIN rolling_1 b ON a.batter = b.batter
        AND b.local_date BETWEEN DATE_SUB(a.local_date, 100) AND a.local_date
        GROUP BY a.batter, a.game_id, a.local_date;
        """
    )
    # ba_rolling_100_df.show()

    batting_avg = CalcBaTransform(inputCols=["tot_hit", "tot_atbat"], outputCol="ba")

    pipeline = Pipeline(stages=[batting_avg])
    model = pipeline.fit(ba_rolling_100_df)
    batting_avg_2_df = model.transform(ba_rolling_100_df)
    output = batting_avg_2_df.withColumn("Rounded_ba", round(batting_avg_2_df["ba"], 3))

    output.show()


if __name__ == "__main__":
    sys.exit(main())
