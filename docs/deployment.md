# Streamlit deployment

Status: tested locally; **not deployed to Streamlit Community Cloud**. There is no verified live app URL. GitHub authentication does not supply a Streamlit Cloud login.

The deployment fields are:

| Setting | Value |
|---|---|
| Repository | `kevinrouse/nba-game-predictor` |
| Branch | `main` |
| Main file | `app.py` |
| Python | `3.11` |
| Secrets | None |

Sign in at [Streamlit Community Cloud](https://share.streamlit.io/), choose **Create app**, enter those fields and deploy. Authorize repository access if Streamlit asks. These steps follow the [official deployment instructions](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy).

Once deployment succeeds, open its assigned URL and check the eleven navigation sections, the Home chapter buttons, a feature selection, a model configuration, a roster date, and a historical game. Check the record comparison, probability-contribution chart, a direct `?game=22301181` URL and the forecast download. The downloaded result must match the selected game. Only after verifying the URL should it be added to the README and GitHub homepage.

The app loads repository-local derived tables and uses cached reads. It does not fit a model or read the full source player file at startup. Frozen models, source data and all required derived artifacts are included. No path outside the repository or environment secret is needed.

Local launch:

```bash
pip install -r requirements.txt
streamlit run app.py
```

Full reproduction uses `requirements-dev.txt` and `python -m scripts.reproduce`. Notebook and plotting dependencies are separated from the runtime requirements.
