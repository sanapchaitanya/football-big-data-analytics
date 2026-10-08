import requests


# --------------------------------------------------
# ESPN CLIENT
# --------------------------------------------------

class ESPNClient:

    BASE_URL = "https://site.api.espn.com/apis/site/v2"


    # --------------------------------------------------
    # INITIALIZE
    # --------------------------------------------------

    def __init__(self):

        self.session = requests.Session()


    # --------------------------------------------------
    # GET SCOREBOARD
    # --------------------------------------------------

    def get_scoreboard(
        self,
        sport,
        league,
        dates
    ):

        url = (
            f"{self.BASE_URL}"
            f"/sports/{sport}/{league}/scoreboard"
        )


        params = {
            "dates": dates
        }


        response = self.session.get(
            url,
            params=params,
            timeout=30
        )


        response.raise_for_status()


        return response.json()

    # --------------------------------------------------
    # GET SEASON CALENDAR
    # --------------------------------------------------

    def get_calendar(
        self,
        league,
        season
    ):

        url = (
            "https://sports.core.api.espn.com/v2/"
            f"sports/soccer/leagues/{league}/"
            f"seasons/{season}/types/1/calendar/ondays"
        )

        response = self.session.get(
            url,
            timeout=30
        )

        response.raise_for_status()

        return response.json()

# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    client = ESPNClient()


    data = client.get_scoreboard(
        sport="soccer",
        league="eng.1",
        dates="20260821"
    )


    print("=" * 60)
    print("ESPN CLIENT TEST")
    print("=" * 60)


    print("\nRequest successful!")

    print("\nNumber of events:")

    print(
        len(data.get("events", []))
    )


    if data.get("events"):

        event = data["events"][0]

        print("\nFirst event:")

        print(
            event["name"]
        )