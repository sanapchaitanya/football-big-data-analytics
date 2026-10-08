import pandas as pd


# --------------------------------------------------
# COMMON SPORTS DATA SCHEMA
# --------------------------------------------------

COMMON_COLUMNS = [

    "event_id",
    "start_date",
    "sport_type",
    "country",
    "country_code",
    "tournament",
    "season",
    "match",
    "participant_1",
    "participant_2",
    "event_status",
    "result",
    "source"

]


# --------------------------------------------------
# VALIDATE DATAFRAME SCHEMA
# --------------------------------------------------

def validate_schema(df):

    print("=" * 60)
    print("SPORTS DATA SCHEMA VALIDATION")
    print("=" * 60)


    # --------------------------------------------------
    # CHECK COLUMNS
    # --------------------------------------------------

    missing_columns = [

        column

        for column in COMMON_COLUMNS

        if column not in df.columns

    ]


    extra_columns = [

        column

        for column in df.columns

        if column not in COMMON_COLUMNS

    ]


    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    print("\nExpected columns:")
    print(COMMON_COLUMNS)


    print("\nActual columns:")
    print(df.columns.tolist())


    print("\nMissing columns:")
    print(missing_columns)


    print("\nExtra columns:")
    print(extra_columns)


    # --------------------------------------------------
    # FINAL RESULT
    # --------------------------------------------------

    if not missing_columns:

        print("\nSchema validation: PASSED")

    else:

        print("\nSchema validation: FAILED")


    return missing_columns, extra_columns


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    # Create a small test DataFrame
    test_data = {

        "event_id": ["123"],

        "start_date": [
            pd.Timestamp("2026-01-01", tz="UTC")
        ],

        "sport_type": ["Football"],

        "country": ["England"],

        "country_code": ["ENG"],

        "tournament": ["Premier League"],

        "season": ["2026-27"],

        "match": ["Arsenal at Chelsea"],

        "participant_1": ["Chelsea"],

        "participant_2": ["Arsenal"],

        "event_status": ["Scheduled"],

        "result": [None],

        "source": ["ESPN"]

    }


    test_df = pd.DataFrame(test_data)


    validate_schema(test_df)