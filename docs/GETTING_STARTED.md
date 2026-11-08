# First run

[Project overview](../README.md) · [Documentation index](README.md)

The repository is Overwatch, while the installed package and CLI are BigBoss.
Use Python 3.11+ and a dedicated environment. Node.js 20+ with npm is needed
for the dashboard. A GPU and local LLM are optional.

## Install from source

~~~bash
git clone https://github.com/Vedant-Sood/Overwatch.git
cd Overwatch
~~~

On Ubuntu/WSL2:

~~~bash
bash scripts/setup.sh
source .venv/bin/activate
~~~

On Windows PowerShell:

~~~powershell
./scripts/setup.ps1
./.venv/Scripts/Activate.ps1
~~~

The setup scripts install the Python dev, xgb and tune extras and build the
frontend when npm is available. If your shell blocks activation, use the
environment's Python and bigboss executables directly. Setup does not install
an optional PyTorch runtime or download an Ollama model.

## Explore without market-data credentials

After dependencies are installed, run:

~~~bash
bigboss demo-data
bigboss status --synthetic
bigboss train --synthetic --horizon 20
bigboss backtest --synthetic --horizon 20
bigboss scan --synthetic --horizon 20
bigboss serve
~~~

Open http://127.0.0.1:8000 and select Synthetic in the dashboard. Training and
backtesting can take time. Synthetic data has its own store; these commands
do not fetch live prices or place real orders.

The API has no login, so `bigboss serve` binds to 127.0.0.1 and only answers
requests addressed to `localhost` or `127.0.0.1`. If you deliberately serve it
on another address with `--host`, anyone who can reach that address can use the
API; set `BIGBOSS_ALLOWED_HOSTS` (comma-separated hostnames) to limit which
Host headers it accepts.

A model may fail a promotion gate or return insufficient evidence. Read the
reported gates rather than treating the existence of a model version as
approval to use it.

## Switch to observed data

Copy configs/bigboss.toml to configs/local.toml and set data.sec_user_agent
to a descriptive contact identity before SEC ingestion. Configure provider
keys through environment variables.

Choose an ingestion path from [DATA_SOURCES.md](DATA_SOURCES.md), then inspect
coverage with bigboss status. Run observed-data commands without --synthetic.
Check provider history windows, adjustments and universe coverage before
training. Data-provider availability and terms can change.

The separate Training-Data repository is not installed by these setup scripts.
Import only compatible local formats using the documented adapters.

## Development checks

~~~bash
python -m pytest -m "not network"
~~~

This selects the existing local suite. Network tests need separate provider
access. The detailed [README](../README.md) documents dashboard development,
optional models, paper trading and API commands. Recorded experiment outcomes
are in [RESULTS.md](RESULTS.md); no fresh model-performance run is claimed here.
