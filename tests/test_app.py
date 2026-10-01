"""Exercise the dashboard pages and meaningful selector combinations."""

from streamlit.testing.v1 import AppTest

from src.data import ROOT


def check(app):
    assert not app.exception, [str(item.value) for item in app.exception]
    assert not app.error


def test_all_dashboard_pages_and_selectors():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    check(app)
    for page in app.sidebar.radio[0].options:
        app.sidebar.radio[0].set_value(page).run()
        check(app)
        if page == "Team explorer":
            for season in [2014, 2019, 2023]:
                app.selectbox[0].set_value(season).run()
                for code in ["ATL", "OKC", "GSW"]:
                    app.selectbox[1].set_value(code).run()
                    check(app)
            for metric in [
                "ortg",
                "drtg",
                "win_pct",
                "efg",
                "point_diff",
                "net_rating",
            ]:
                app.selectbox[2].set_value(metric).run()
                check(app)
        elif page == "Roster explorer":
            for season, code in [(2014, "OKC"), (2019, "LAL"), (2023, "BOS")]:
                app.selectbox[0].set_value(season).run()
                app.selectbox[1].set_value(code).run()
                check(app)
                app.selectbox[2].select_index(0).run()
                check(app)
        elif page == "Historical game predictor":
            for code in ["OKC", "BOS", "All teams"]:
                app.selectbox[0].set_value(code).run()
                app.selectbox[1].select_index(-1).run()
                check(app)
