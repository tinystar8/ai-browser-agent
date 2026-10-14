# AI Browser Agent

A local browser agent for Ubuntu. You send a task over HTTP. A Qwen model running in Ollama decides the next browser action. [Browser Use](https://github.com/browser-use/browser-use) drives Chromium through Playwright. SQLite stores every task and action.

Nothing in the default setup calls the OpenAI cloud API. `OPENAI_BASE_URL` points at Ollama on this machine. The process refuses to start if that URL is `api.openai.com`.

## What it will not do

The browser agent blocks these before Chromium launches, and again before each proposed action runs:

- deleting files
- sending messages or email
- financial transactions
- changing passwords
- installing software

The optional shell tool may run `ls`, `pwd`, `git status`, and `docker ps` on its own. `rm`, `sudo`, `shutdown`, and `reboot` run only when you send `confirm: true` to `POST /shell`. The browser model cannot set that flag. Wiping `/` is refused even with confirmation.

The agent does not type passwords. Sign in yourself in the Chromium window it opens. The persistent profile keeps that session for the next task.

## Project layout

```
ai-browser-agent/
├── app/
│   ├── main.py              FastAPI routes
│   ├── cli.py               ./prompt command
│   ├── agent.py             Browser Use task runner
│   ├── llm.py               LangChain client aimed at Ollama
│   ├── skills.py            Loads skills/*.md
│   ├── browser.py           Chromium profile and headless flag
│   ├── memory.py            SQLite tasks, actions, chat, instructions
│   ├── static/index.html    Chat page for steering a running task
│   ├── models.py            Request and response models
│   ├── config.py            .env loading
│   ├── safety.py            Block rules
│   └── tools/
│       ├── filesystem.py    Read-only files; opens SQLite .db as text
│       ├── shell.py         Allowlisted commands
│       ├── solidity.py      Saves verified .sol files by their real names
│       └── functions.py     Read/write function list and address values
├── skills/                  Prompt skills you can add
├── contracts/               Saved Solidity, one folder per address
├── prompt                   Send a prompt in plain text
├── database/agent.db        Created on startup
├── logs/                    actions.jsonl and agent.log
├── screenshots/             PNG after each step
├── browser-profile/         Persistent Chromium profile
├── .env
├── requirements.txt
├── run.sh
└── README.md
```

## 1. Ubuntu packages

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip ca-certificates curl
python3 --version   # must be 3.12 or newer
```

On a desktop session, leave `HEADLESS=false` so you can watch Chromium and log in. On a server with no display, set `HEADLESS=true` in `.env`. A headless run cannot complete a site that needs you to sign in interactively.

## 2. Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

The installer starts a systemd service. Check it:

```bash
systemctl status ollama
```

If it is not running:

```bash
ollama serve
```

Leave that process running. The API is `http://localhost:11434`.

## 3. Install the Qwen model

```bash
ollama pull qwen3.6:35b
```

This image is large. Pull it on a machine with enough disk and RAM for a 35B model (roughly 24 GB of memory, more if the quant is not compressed).

Confirm the tag Ollama stored:

```bash
ollama list
```

If the tag differs from `qwen3.6:35b`, put the real name in `.env`:

```bash
OPENAI_MODEL=qwen3.6:35b
```

Quick check that the OpenAI-compatible endpoint is up:

```bash
curl http://localhost:11434/v1/models
```

## 4. Configure the agent

The repo ships with `.env`:

```bash
OPENAI_API_KEY=ollama
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_MODEL=qwen3.6:35b
HEADLESS=false
BROWSER_PROFILE=./browser-profile
```

`OPENAI_API_KEY=ollama` is a dummy value. Ollama does not need a real OpenAI key. The LangChain client still sends it because the OpenAI SDK requires some key.

`USE_VISION=false` keeps screenshots on disk without sending them to the model. Turn it on only if your Qwen build accepts images. Vision calls are much slower on a 35B model.

## 5. Python and Chromium

From this directory:

```bash
chmod +x run.sh
./run.sh
```

`run.sh` will:

1. Create `.venv` with Python 3.12+ if it is missing.
2. Install `requirements.txt`.
3. Install Playwright's Chromium build.
4. Start FastAPI on `http://127.0.0.1:8000`.

Stop the server with Ctrl+C while no task is running. During a task, Browser Use treats the first Ctrl+C as pause and the second as exit.

## Chat while it runs

Start `./run.sh`, then open [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

Type a sentence and press Send. If nothing is running, that starts a task. If a task is already running, the message is given to the agent before its next click, and the click it had already chosen is skipped.

When the task finishes, Chromium stays open. The chat asks you before closing it. Send the next instruction to continue in the same window, or send `close` to close it.

Each message sent during a run is saved with the task and the page it was about. A later task that shares those words includes the instruction, so the agent can make the same decision again. A different kind of task does not inherit it.

## Give a prompt

Start `./run.sh` in one terminal. In another:

```bash
./prompt Open Google and search for Ethereum price
```

Or run `./prompt` and type the sentence when it asks. The command waits, then prints the final result.

Use a skill the same way. The first word is the skill name, and the rest are its arguments:

```bash
./prompt etherscan-contract 0x006FaD88c35D973A87E451CF8D000c7e83Dad409
```

That skill saves the verified Solidity before the browser opens. Each source file keeps its original name under `contracts/<address>/`, for example `contracts/<address>/lib/openzeppelin-contracts-v5/contracts/interfaces/IERC1967.sol`. A proxy also saves the implementation sources under `contracts/<implementation>/`.

It also writes `contracts/<address>/functions.txt`. Each function is marked `read` or `write`. Address inputs are marked. Read functions that take no arguments include the value from Ethereum, and any returned address is listed at the bottom of that file. A proxy is read through the proxy, the same way Etherscan's Read as Proxy tab works.

`./prompt --list` shows the skills that are installed.

If the API is not running, `./prompt` starts the browser itself.

## Add a skill

Create `skills/your-skill.md`. A header, a blank line, then the prompt. Put `{argument}` where the command-line value should go.

```markdown
name: your-skill
description: What this skill does
args: query

Open Google and search for {query}.
Summarize the first result. Do not sign in.
```

Then run:

```bash
./prompt your-skill ethereum price
```

`args` is a comma-separated list when a skill needs more than one value. `validate: address=ethereum` checks that an argument is a 0x address. No other code change is required. Restart `./run.sh` after adding a file so the API reloads it. `./prompt` reads the folder on each run.

## API

### Health

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status": "ok", "ollama": "reachable", "model": "qwen3.6:35b"}
```

`status` is `ok` when the API process is up. `ollama` is `reachable` only when `GET /v1/models` on your base URL answers.

### Start a task

```bash
curl -X POST http://127.0.0.1:8000/task \
  -H 'Content-Type: application/json' \
  -d '{"task": "Open Google and search for Ethereum price"}'
```

```json
{"task_id": 1, "status": "running"}
```

A blocked task comes back immediately:

```json
{"task_id": 2, "status": "blocked", "reason": "Blocked automatically: financial transactions. ..."}
```

Only one browser task runs at a time. A second request while the first is running returns HTTP 409.

### Read a task

```bash
curl http://127.0.0.1:8000/task/1
```

```json
{"status": "completed", "result": "Ethereum is trading at ...", "actions": []}
```

`actions` lists what the model selected. Screenshots land in `screenshots/task-1/step-001.png`.

Recent history:

```bash
curl 'http://127.0.0.1:8000/tasks?limit=10'
```

The next browser task receives a short summary of earlier finished tasks in its prompt.

### Shell tool

```bash
curl -X POST http://127.0.0.1:8000/shell \
  -H 'Content-Type: application/json' \
  -d '{"command": "git status"}'
```

Confirmation required:

```bash
curl -X POST http://127.0.0.1:8000/shell \
  -H 'Content-Type: application/json' \
  -d '{"command": "rm ./tmp-notes.txt", "confirm": false}'
```

```json
{"status": "needs_confirmation", "output": "", "reason": "`rm` is not run automatically. ..."}
```

Send the same command again with `"confirm": true` only if you mean to run it.

## Example tasks

These match the intended use: look something up, or read a site you are already signed into.

```bash
curl -X POST http://127.0.0.1:8000/task \
  -H 'Content-Type: application/json' \
  -d '{"task": "Open Google and search for Ethereum price"}'

curl -X POST http://127.0.0.1:8000/task \
  -H 'Content-Type: application/json' \
  -d '{"task": "Open GitHub and check my latest failed workflow"}'

curl -X POST http://127.0.0.1:8000/task \
  -H 'Content-Type: application/json' \
  -d '{"task": "Open https://news.ycombinator.com and extract the titles of the top 5 posts"}'

curl -X POST http://127.0.0.1:8000/task \
  -H 'Content-Type: application/json' \
  -d '{"task": "Navigate to the AWS console and summarize the dashboard"}'
```

For GitHub and AWS, start once with `HEADLESS=false`, sign in in the Chromium window, and let the profile save the session. Later tasks reuse `./browser-profile`. The agent will not type a password for you.

A 35B model often needs one to three minutes per step. `MAX_STEPS=25` caps a run.

## Logs

Each action is appended to `logs/actions.jsonl`:

```json
{"time": "2026-10-13T10:00:00", "action": "click", "target": "Login button", "task_id": 1}
```

`logs/agent.log` is the same stream plus server messages. The `actions` table in SQLite stores `id`, `task_id`, `action`, `details`, and `timestamp`. The `tasks` table stores `id`, `task`, `status`, `created_at`, `finished_at`, and `result`.

`database/agent.db` is a binary SQLite file, so the editor will not show its rows. The file tool opens it read-only and prints each table as text. From the project directory:

```bash
./.venv/bin/python -c "from pathlib import Path; from app.tools.filesystem import inspect_path; print(inspect_path('database/agent.db', Path('.')))"
```

## Troubleshooting

**`ollama` is `unreachable`.** Start Ollama (`systemctl start ollama` or `ollama serve`) and check `curl http://localhost:11434/v1/models`.

**Model not found.** `ollama list` and set `OPENAI_MODEL` to that exact tag. Then restart `./run.sh`.

**Step times out.** Raise `LLM_TIMEOUT` and `STEP_TIMEOUT` in `.env`. The first token from a 35B model can take a while, especially while the model is loading into memory.

**Invalid JSON from the model.** The agent asks Qwen for one JSON object and strips `<think>` blocks and Markdown fences. If failures continue, the model build may not follow the action schema. Use a tool-capable Qwen tag and keep `USE_THINKING=false`.

**Chromium does not start.** From the virtualenv run `python -m playwright install chromium`. If another Chromium is already using `./browser-profile`, close it. Two processes cannot share that profile.

**`HEADLESS=false` on a server.** There is no display. Set `HEADLESS=true`, or use a machine with a desktop session.

**Port already in use.** Change `PORT` in `.env`.

**HTTP 409.** Wait until `GET /task/{id}` is `completed`, `failed`, or `blocked` before starting another browser task. If the server was stopped mid-task, the next startup marks that task `failed` so the queue is not stuck.

**Refusing the OpenAI cloud API.** `OPENAI_BASE_URL` must be your Ollama server, normally `http://localhost:11434/v1`.

## Development

The API imports the `app` package from the project root. `./run.sh` already does that. To run it by hand:

```bash
source .venv/bin/activate
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```
