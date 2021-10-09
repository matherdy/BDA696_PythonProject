import sys

from pyspark.sql import SparkSession
from transform import SplitColumnTransform


def main():

    spark = SparkSession.builder.master("local[*]").getOrCreate()

    rolling_1_df = (
        spark.read.format("jdbc")
        .option("url", "jdbc:mysql://localhost:3306/baseball")
        .option("driver", "org.mariadb.jdbc.Driver")
        .option(
            "query",
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
                """,
        )
        .option("user", "root")
        .option("password", "BDAMaster")
        .load()
    )
    # rolling_1_df.show()

    rolling_1_df.createOrReplaceTempView("rolling_1")

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

    batting_avg = SplitColumnTransform(
        inputCols=["tot_hit", "tot_atbat"], outputCol="ba"
    )

    output = batting_avg.transform(ba_rolling_100_df)
    output.show()


if __name__ == "__main__":
    sys.exit(main())
