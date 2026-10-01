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
        if page == "Basketball questions":
            for feature in ["win_pct_diff", "weighted_absence_diff", "oreb_rate_diff"]:
                app.selectbox[0].set_value(feature).run()
                check(app)
        elif page == "Feature workshop":
            for metric in ["weighted_ppa", "dreb_rate", "top5_share"]:
                app.selectbox[0].set_value(metric).run()
                check(app)
        elif page == "Model comparison":
            for name in [
                "Home win rate",
                "Forest combined10 leaf=60",
                "Boosting combined10 leaves=3",
                "L1 team_players C=0.1",
            ]:
                app.selectbox[0].set_value(name).run()
                check(app)
        elif page == "Biggest misses":
            for label in app.selectbox[0].options:
                app.selectbox[0].set_value(label).run()
                check(app)
        elif page == "Team explorer":
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
            app.multiselect[0].set_value([5, 10, 20]).run()
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


def test_home_chapter_navigation():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.button(key="home_Basketball questions").click().run()
    check(app)
    assert app.sidebar.radio[0].value == "Basketball questions"
    app.button(key="chapter_next").click().run()
    check(app)
    assert app.sidebar.radio[0].value == "Feature workshop"
