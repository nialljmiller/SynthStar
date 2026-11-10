# Handoff: the live demo on huepil (real hdps, web front end over Tailscale Funnel)

A separate project, started 2026-10-27. It is not the portable two-file demo (the notebook plus its tarball for the
symposium hub; that one is finished and on `main` of argus-hdps/argus_night_live), and not the alerts chat (hdps runs,
BOOM alert samples, real/bogus training on newton and vega).

The user, 2026-10-27: "lets start a new chat for the real hdps that runs on huepil and does the front end funnel html
thing. this is a seperate project."

## Ask the user first

1. **"The real hdps that runs on huepil"**: huepil hosts an hdps deployment run by the `hdps` user (a Ray cluster,
   java, and a `ray::GPUWorker` holding about 13 GB of GPU memory). Is the goal to use that deployment (live
   processing on huepil's GPU), or to run the demo app (argus_night_live, which runs its own copy of hdps's code) on
   huepil?
2. **"The front end funnel html thing"**: a web page, published to the internet with Tailscale Funnel. Which page
   (the argus_night_live app, a new page, or something on the existing hdps front end below), and at what address?

## huepil

From the memory file `~/.claude/projects/-Users-njm/memory/huepil.md`, plus checks on 2026-10-27:

- Log in with `ssh nillmill@huepil` (key auth). Tailscale address 100.91.210.25. Never save the password. nillmill has
  no sudo.
- Ubuntu 24.04, Linux x86_64. Threadripper 9960X (24 cores, 48 threads), 125 GB RAM, NVIDIA RTX PRO 6000 Blackwell
  with 96 GB.
- Shared with the `hdps` user, whose jobs come first. Check `nvidia-smi` and `free -g` before starting anything.
  At 20:45 UTC: GPU 13 GB used of 98 GB at 0% utilisation; RAM 23 GB used of 125 GB.
- Python: the system python3.12 has no pip. Make venvs with uv (`~/.local/bin/uv`, version 0.13.6), e.g.
  `uv venv --python 3.14 .venv && uv pip install --python .venv/bin/python ...`.
- Tailscale 1.102.4. `tailscale serve status` on 2026-10-27:
  - `https://huepil.tailab0df0.ts.net` (Funnel on): `/` proxies to `http://127.0.0.1:3000`. That is nginx, which
    redirects `/` to `/explorer/`. This is the existing public page; its owner was not checked (probably the hdps
    team).
  - `https://huepil.tailab0df0.ts.net:8443` (tailnet only): proxies to `http://127.0.0.1:8090`, a page titled
    "ADDEM mosaic, β + ρ, 15 × 60 s per field, 2026-10-07".
  - Tailscale's `OperatorUser` is `hdps`. Only that user (or root) can change serve or Funnel settings; nillmill
    cannot. Funnel can only publish on ports 443, 8443 and 10000, and 443 is already taken by the page above.

## What was tried on 2026-10-27 (Voilà, never published, stopped)

- `~/argus_night_live_live` on huepil holds:
  - the no-Zemax archive `20261027T2010_6199055`, unpacked, with `current` linked to it;
  - a uv venv `.venv` with Python 3.14, the archive's `requirements.txt` and `voila`;
  - `tools/check.py`, which passed there (`check.log`).
- `night_live.ipynb` is Voilà's entry point:
  ```python
  %matplotlib widget
  import os, sys, warnings
  warnings.simplefilter("ignore")
  os.environ.setdefault("ARGUS_NIGHT_LIVE_WORKERS", "2")  # a shared machine: two worker processes per visitor
  sys.path.insert(0, os.path.expanduser("~/argus_night_live_live/current"))
  from argus_night_live.app import App
  App().show()
  ```
- `serve.sh` (in that folder):
  ```bash
  cd ~/argus_night_live_live
  exec .venv/bin/voila night_live.ipynb --no-browser --port=8866 --Voila.ip=127.0.0.1 --strip_sources=True \
    --MappingKernelManager.cull_idle_timeout=1200 --MappingKernelManager.cull_interval=120 \
    --MappingKernelManager.cull_connected=True \
    --Voila.tornado_settings='{"websocket_max_message_size": 209715200, "headers": {"Content-Security-Policy": "frame-ancestors *"}}'
  ```
  The intended publish command, which the operator user `hdps` would have to run, was
  `tailscale funnel --bg --https=10000 http://127.0.0.1:8866`. It was never published.
- Testing went through an ssh tunnel (`ssh -N -L 18866:127.0.0.1:8866 nillmill@huepil`) with headless Chrome
  (puppeteer). The controls appeared after 2.4 s, and screenshots were taken. ipympl figure updates under Voilà
  were looked into: its `send_binary`/`_data_url` path is in ipympl's `backend_nbagg.py`.
- The user stopped this to have the portable two-file version made first. Voilà is stopped and nothing listens on 8866.

## Constraints that carry over

- **Zemax NDA**: Zemax PSF files, ArgusSim's ZemaxPSF code, the EE50 profiles and Zemax-derived PSF stamps must
  never be in anything public, and a Funnel address is public on the internet. Use the no-Zemax build: the
  argus_night_live export ships the public argus_sim, and stars are drawn with the Gaussian PSF.
  - The old field packs (f294, f065, f098) were drawn with Zemax. The user called the images "less NDA" than the
    files and code; they are being replaced.
  - Gaussian packs so far are in `~/argus_repos/argus_night_live/data/interactive/packs_gaussian/`: f098 β, and
    M67 for 2026-02-17 and 2026-02-20. The alerts chat runs the rest on newton and vega and reports each run
    folder.
  - See memory `zemax-nda.md`.
- **Private code**: hdps (argus-hdps) and the demo repo are private. A public page must not serve source code or a
  Jupyter server (Voilà with `--strip_sources`; no file browser, no terminals).
- **Shared machine**: each visitor gets a kernel holding a night in memory, plus its worker processes. Limit the
  number of visitors and the memory each one may use.
- Commit and push only when the user asks. Never put AI attribution in commits.

## Where things are

- Demo repo: `~/argus_repos/argus_night_live` (branch `dev` = source, `main` = notebook and README),
  github.com/argus-hdps/argus_night_live (private).
  - The archive is built by `tools/export.py`; `--vendor <dir>` adds Python packages.
  - Latest archives (Gaussian nights only): `dist/linux/` (sent to Hank), `dist/mac/`, and `dist/noarch/`
    (`argus_night_live_20261028T0056_d43d148_noarch.tar.gz`, no Python packages: uses the kernel's own). The
    5cf6572 archive (Zemax-drawn packs) was deleted from the Mac on 2026-10-30; huepil keeps its own copy.
  - The app is `argus_night_live/app.py`. `App()` shows a live figure with ipympl, or a still image plus X/Y boxes
    without it.
  - See memory `demo-repo.md`.
- hdps: `~/argus_repos/hdps_v4`, branch `alert-v4-local` (local only, never pushed).
- The alerts chat's runs: memories `chat-split.md` and `newton-job-monitoring.md`.

## State 2026-10-27 23:30 UTC (huepil chat)

User's answers: run the demo app first, the hdps deployment later. Then: "dont make anything public apart form a quick
test and only do things on my account". So no Funnel through the `hdps` account (port 10000 is off the table), and
nothing runs outside `nillmill`.

On huepil, `~/argus_night_live_live`:
- `current` -> `20261027T2230_5cf6572` (the latest archive, `archives/`, unpacked by `deploy.sh` without its
  Zemax-drawn packs and without `vendor/`). The 2010 copy's `data/interactive/packs` (Zemax-drawn) was deleted.
- `packs_gaussian/`: f294 β (added 23:55), f098 β, M67 2026-02-17 β and 2026-02-20 β, the same files (sha256) as
  the demo repo's `packs/` and `packs_gaussian/`. `env.sh` points `ARGUS_NIGHT_LIVE_PACKS` there; the Field menu shows
  f294, f098 and M67. `cache/` pre-filled by `prewarm.sh`. To add a pack: rsync it into `packs_gaussian/`, run
  `prewarm.sh`; new visitors see it, no restart. Gaussian runs with no movie data (never finished, no pack possible):
  newton `interactive_f065_g.ended_20261027`, `interactive_m67_g0214.ended_20261027`; vega
  `interactive_m67_g0218.ended_20261027`.
- `tools/check.py` with these settings (`check_2230.log`): every part passes. Its one failure is the expected one:
  packs are read from outside the archive's folder.
- `env.sh` (the settings), `serve.sh` (Voilà, root `www/` holding only `night_live.ipynb`, `--strip_sources`,
  culls a kernel 180 s idle with no page connected, checked every 60 s), `www/night_live.ipynb` (4 visitor slots by lock files in `slots/`; beyond that the page says all
  sessions are in use; the visitor's kernel sets oom_score_adj 500), `argus-night-live.service` (12G cap, no swap,
  CPUWeight 20, OOMPolicy=continue; saved there, NOT installed).
- Running now as a transient unit, only while nillmill has a login (no linger):
  `systemd-run --user --unit=argus-night-live-test -p MemoryMax=12G -p MemorySwapMax=0 -p CPUWeight=20 -p OOMPolicy=continue $HOME/argus_night_live_live/serve.sh`.
  Stop: `ssh nillmill@huepil systemctl --user stop argus-night-live-test`. The Mac's `ssh -f -N -L 18866:...`
  tunnel is the login that keeps the unit alive. Stop it with `pkill -f "18866:127.0.0.1:8866"` on the Mac.

Tested through `ssh -L 18866:127.0.0.1:8866 nillmill@huepil` with headless Chrome:
- Visitors reach the app 4-6 s after loading. 5 at once: 4 get the app, the fifth the "in use" page.
- 4 visitors (2 workers each, "One of each type", Play) peaked at 5.3 GB in the service.
- Play shows the workers' JPEG frames: 238 exposures in about 13 s at 1×.
- A closed page's kernel was culled about 4 min after closing. An open page idle for 5.5 min kept its kernel.
- Voilà accepts only comm, kernel_info and shutdown messages from the browser. A probe's `execute_request` was
  refused ("not allowed. Ignoring.").

Open:
- Voilà's own shutdown when a page closes (`POST /voila/api/shutdown/<kernel>`) gets 403 (`'_xsrf' argument
  missing`). The idle cull of disconnected kernels (above) frees the slot 3-4 min later instead. Turning off the
  XSRF check (`"disable_check_xsrf": true` in tornado_settings) was blocked by the permission classifier. It is the
  user's call, and is not needed with the cull.
- `loginctl enable-linger nillmill` (to keep a user service running without a login) was blocked by the permission
  classifier, also the user's call.
- Quick public test, waiting on the user's choice:
  - (a) A Cloudflare quick tunnel (`cloudflared tunnel --url http://127.0.0.1:8866`) run as nillmill on huepil, with
    the binary in `~`. It needs no account, gives a random `https://<words>.trycloudflare.com` address that ends
    when the process stops, and goes through a third-party service.
  - (b) `tailscale funnel 18866` on the Mac, through the ssh tunnel, at `https://nialls-macbook-air.tailab0df0.ts.net`.
    The Mac's node (tailnet argus-hdps.org.github) has no Funnel permission: a tailnet admin must add the funnel
    attribute to it first. The page is up only while the Mac and the tunnel are.
  - Through either, run `visitor.mjs` against the public address first. The tunnel test leaves jupyter_server's
    websocket Origin/Host check untested; if `/api/kernels/<id>/channels` is refused, set
    `--ServerApp.allow_origin` (or Voilà's equivalent) to that exact public origin, never `*`.
- The 12G cap was sized from light visitors (one field, 7 stars). A heavy visitor (many stars, a night switch,
  zoom, play) is not measured. Above the cap, the kernel's OOM kills a visitor's kernel (oom_score_adj 500), not
  the server.
- Test scripts (headless Chrome, puppeteer on the Mac): `visitor.mjs` and `play.mjs` in this chat's scratchpad,
  `/private/tmp/claude-501/-Users-njm-MESA-ARGUS/4ef8d917-d07b-4705-9e16-32b72394ee29/scratchpad/huepil/`
  (temporary).

## Famous fields on vega (2026-10-28, user: "can vega be set on making more. specifically of famous fields"; chose M3 +
M45, then "do just m45")

- MESA_ARGUS (Mac, uncommitted) `examples/observations/interactive_m3.toml` and `interactive_m45.toml`: added
  `[argussim.telescope] psf_model = "gaussian"` (M67's line; both had none, so they would have used Zemax). M45's date
  2026-11-20 -> 2026-11-09 (new moon; on 11-20 the Moon is 77% lit, 60 deg away). Both copied to vega (vega's old ones
  kept as `*.bak_vega_20261027`); `stars/m3_field` and `stars/m45_field` copied to vega. Resolved on vega:
  psf_model gaussian, movie_window_px 0.
- vega `outputs/rb_queue/vega_v2.queue` (the alerts chat's runner, `tools/rb_queue.py`; backup
  `vega_v2.queue.bak_m45_20261027`): appended `interactive_m45_g@15G ... observe.py
  examples/observations/interactive_m45.toml --name interactive_m45_g --bands beta`. Cap 15G because 15.5 GB was free
  beside 11.7 GB of other use. Its 4 rb2_ lines had all finished (state ok): nothing was waiting.
- Deleted on vega (user approved): `outputs/pseudo_real/interactive_m67_g0218.ended_20261027` (4.0 GB, the ended
  Gaussian night, no movie data); vega then had 16 GB free (runner minimum 15 GB).
- Started 2026-10-27 23:27 (vega's clock), unit `q-interactive_m45_g.scope`. Next: copy the run to the Mac, build the
  pack (`argus_night_live.pack`, `--pack m45`, ARGUS_PIPELINE=~/MESA_ARGUS), rsync it into huepil's
  `packs_gaussian/`, run `prewarm.sh`. M3 not queued (user: just M45).

### M45 result (2026-10-28)

- vega run ended 01:31 (vega's clock) after 2.06 h, peak 15.0 GB plus 1.5 GB swap at the 15G cap. The queue's state
  line says `failed` (exit 1), but the night itself finished: 39038 hdps alert candidates from 210 frames (none
  failed), movie data saved (176 MB). The failure came after, in the light-curve step
  (`synthstar/pipeline/lightcurves.py` `reference_stars`): ATLASRefcat2 returned nothing at M45 on vega
  (`pandas.concat`: "No objects to concatenate"), so `lightcurves/` and `summary.md` are incomplete. The pack does not
  use that step. The runner counts `failed` as finished, so it will not rerun.
- Run copied to the Mac: `outputs/pseudo_real/interactive_m45_g/` (1.3 GB).
- Pack built on the Mac, outside the demo repo:
  `outputs/pseudo_real/interactive_m45_g/pack/m45_beta.joblib` (133 MB, sha256 a9a18cd5...0b04, build log beside it).
  - The build used ARGUS_PIPELINE=~/MESA_ARGUS and ARGUS_NIGHT_LIVE_PACKS=that folder.
  - It ran through a small wrapper, scratchpad `huepil/build_pack.py`, which sets `pack.REPO = /`, because `build()`
    prints the path relative to the repo.
  - Result: 210 frames, 0 followed variables (the field has none), 39038 candidates, cut-out zero points 26.096
    (rms 0.052), Gaussian PSF ("no Zemax file read"). 134 s.
  - The demo repo's `data/interactive/packs/` was not touched. Copying the pack there is the demo chat's call.
- huepil: the pack is in `packs_gaussian/`, its cache was pre-warmed by `prewarm.sh` (210 exposures, 7219 field
  stars), and its sha256 matches the Mac's. Without a restart, the Field menu shows "M45, the Pleiades (young open
  cluster)". Headless test through the tunnel: the field loads (2026-11-09, β), star A is detected in 210/210 frames,
  and Play reaches exposure 209 in about 10 s at 1×. No console errors.
- For the demo chat, found here:
  - **NDA:** every pack (M45 and the Gaussian packs already made, M67 included) keeps `settings["ARGUS_CONFIG"]`,
    the private ArgusSim config. It holds `psf_model "zemax"`, `psf_fits_path .../zemax_psf_25angles.fits` and the
    `ee50_contract_*` values.
    - It comes from `pack.module_settings` (night_movie's module settings).
    - The public argus_sim in the archive has none of these keys.
    - export.py's FORBIDDEN check looks only at file paths, so these values would ship in the tarball's packs.
    - On huepil they are not reachable: Voilà serves only the notebook, and visitors cannot run code.
  - Display: the M45 header reads `03h46m60.0s`, a rounding error that should read `03h47m00.0s`.

## Tailnet link (2026-10-28, user: "a link that i can give to anyone who has tailscale ... logged into our tailscale")

- **Link:** http://huepil.tailab0df0.ts.net:8866/
  - Voilà now binds huepil's tailnet address only: `ARGUS_NIGHT_LIVE_IP=100.91.210.25` in `env.sh`, used by
    `serve.sh`.
  - Never 0.0.0.0: huepil has other interfaces (nginx on 0.0.0.0:80).
  - The ssh tunnel is no longer needed.
- **Not used:** `tailscale serve`/HTTPS, because huepil's Tailscale operator is `hdps` and 443 on the tailnet address is
  the hdps team's. Plain HTTP inside the tailnet is still WireGuard-encrypted, but browsers show "Not secure".
- **Why it went down:** the transient unit died while the session was unattended, at the latest by 2026-10-28 13:59
  (huepil's clock). The Mac's tunnel, the only nillmill login, had closed, and with Linger=no the user manager stops
  with the last login.
- **Now:**
  - `argus-night-live.service` (12G, CPUWeight 20, OOMPolicy continue) is installed in `~/.config/systemd/user/` and
    started. It is not enabled.
  - It lives only while a nillmill login exists. The Mac holds one with
    `ssh -f -N -o ServerAliveInterval=30 nillmill@huepil` (pid 64789 on 10-28), which dies if the Mac sleeps.
- **Waiting on the user** (both were refused by the permission classifier as persistence): run
  `ssh nillmill@huepil 'loginctl enable-linger nillmill && systemctl --user enable argus-night-live.service'`.
  Then the holding ssh can go.
- **Tested from the Mac over the tailnet:**
  - Headless Chrome on the link: the page loads, the websocket passes Voilà's Origin check, M45 selects and plays to
    exposure 209.
  - Every Jupyter API path (`/api/contents`, `/api/terminals`, `/api/kernels`, `/tree`, `/lab`, ...) redirects into
    `/voila/files/`, whose allowlist refuses it (403). `POST /api/kernels` gets 403.
  - Kernel ZMQ ports bind 127.0.0.1 only.
- **Not tested:** whether other tailnet members can reach port 8866. huepil is a tagged device, so the tailnet ACL
  decides that. One other member should open the link, or a tailnet admin should check the ACL.
- **Capacity:** 4 visitors at once (`ARGUS_NIGHT_LIVE_SLOTS`). A closed tab frees its slot 3-4 min later.

### Switched from systemd to tmux (2026-10-28 14:20 UTC, user: "can we do this with a python scrip running in screen
instead of a systemd. i know its more jank but it is less messy")

This replaces the service setup in the section above. Linger and `systemctl --user enable` are no longer needed.

- **Removed:** `argus-night-live.service`, from `~/.config/systemd/user/` and from the folder (stopped, daemon-reload).
- **Now:** `~/argus_night_live_live/start.sh` starts the tmux session `argus-live`.
  - The session runs `while true; do nice -n 15 ./serve.sh; ...; sleep 10; done`, so Voilà restarts 10 s after it
    exits.
  - `screen` is not installed on huepil; tmux 3.4 is.
- **Look at it:** `tmux attach -t argus-live` (leave with Ctrl-b d).
- **Stop it:** `tmux kill-session -t argus-live`.
- **After a reboot:** run `./start.sh` again. Nothing starts it at boot.
- **Survives logout:** huepil's logind has KillUserProcesses=false, so processes stay when their login ends.
  - Tested: the tmux server and Voilà ran in `session-31637.scope`. That ssh login ended (State=closing), and the page
    kept answering.
  - The Mac's held ssh (pid 64789) is closed.
  - Over the tailnet afterwards: 200 on the link; headless Chrome loads M45 and plays it to exposure 209.
- **Lost against systemd:**
  - The 12G cap on everything together. What remains is the per-visitor limit (4 slots) and `oom_score_adj` 500 on
    each visitor kernel: under memory pressure, the kernel's OOM killer takes visitor kernels first.
  - CPUWeight 20, replaced by `nice -n 15`, which the kernels and workers inherit.
  - Restart at boot.
- **Output:** the tmux pane's scrollback only (2000 lines). The old `voila.log` in the folder is from the 10-27 runs.

### Hank cannot reach the link (2026-10-28 ~14:45 UTC)

- **What happened:** Hank (corbettht, laptop `joara` 100.116.237.128) tried the link and it did not load. He confirmed
  he is on the tailnet.
- **What huepil saw:**
  - `tailscale status` shows traffic from joara: "idle, tx 2676 rx 9572", where before it showed none. Part of this is
    my `tailscale ping`.
  - Not one request from his address reached Voilà, which logs every request.
  - From the Mac (nialljmiller) the same link works.
- **Likely cause:** the tailnet's access rules (ACL) let his account reach huepil (tags `tag:lab-servers`,
  `tag:public-html`) but not tcp:8866. The policy itself cannot be read from huepil as nillmill:
  `tailscale debug netmap` needs sudo, and the tailscaled journal is not readable.
- **Fix:** a tailnet admin allows the members on tcp:8866, either
  `{"action": "accept", "src": ["autogroup:member"], "dst": ["tag:public-html:8866"]}` (acls) or
  `{"src": ["autogroup:member"], "dst": ["tag:public-html"], "ip": ["tcp:8866"]}` (grants).
  - Not done here: the hdps team's 80/443 (nginx, tailscale serve), which is excluded by the user's rule.

### Under /night-live/, nginx block tested (2026-10-28 15:20 UTC; Hank: "go through the local nginx to relay")

- **Hank's route:** the hdps team's public Funnel serves https://huepil.tailab0df0.ts.net/ from
  127.0.0.1:3000, which is nginx, site `/etc/nginx/sites-available/dms-data-sample` (the data explorer: `/explorer/`,
  `/data/`, `/docs`, `/api/`). A `/night-live/` location there forwards to Voilà. I only read their config.
- **Voilà now:**
  - It runs with `--base_url=/night-live/` (`ARGUS_NIGHT_LIVE_BASE` in `env.sh`). The prefix is needed because the
    explorer already uses `/api/`.
  - No token or password (user, 2026-10-28: "this needs to be as simple as possible").
  - Tailnet link: `http://huepil.tailab0df0.ts.net:8866/night-live/`. Tested: 200, and headless Chrome plays M45 to 209.
- **The block for nginx:** `~/argus_night_live_live/night-live.nginx`, to go inside the `listen 3000` server, then
  `sudo nginx -t && sudo systemctl reload nginx`.
  - Tested in a private nginx 1.24 run as nillmill on 127.0.0.1:18300, with the same `absolute_redirect off`, reached
    through an ssh tunnel. It has since been stopped and removed.
  - With `Host $host` the websocket got 403 ("Blocking Cross Origin WebSocket Attempt. Origin:
    http://127.0.0.1:18300, Host: 127.0.0.1"), because `$host` drops the port.
  - With `Host $http_host` it got 101: M45 loads and plays to 209.
  - Behind Funnel (no port) both forms would match, but `$http_host` is right in every case.
- **For the demo chat:** every page load makes one request `GET /night-live/iVBORw0KGgo...`, a base64 PNG used as a
  relative URL (missing its `data:image/png;base64,` prefix?). The console shows "Error setting state: Failed to
  execute 'drawImage' ... 'broken' state".
  - Through nginx it is a 414.
  - Directly, Tornado resets the connection (the `ERR_CONNECTION_RESET` seen in every headless test).
  - The page works regardless. It is probably an ipympl canvas state (`_data_url`?) set by the app.
- **Next:** Hank or the hdps team adds the block and reloads nginx. Then I test
  `https://huepil.tailab0df0.ts.net/night-live/` from outside the tailnet and from it.

## Spooky stars: the Halloween field (2026-10-28)

The user's spec:
- Halloween night, vega.
- A jack-o-lantern made of short-period, high-amplitude pulsators or binaries.
- Field stars so the exposure is a boring star field.
- New Moon.
- Menu label and all naming "spooky stars".
- The light-curve panel shows about 6 of the pumpkin's stars.

What was made:
- **Generator:** `tools/spooky_stars.py` (MESA_ARGUS, uncommitted) writes `examples/observations/spooky_stars.toml`.
  Its settings from `[image]` on are `interactive_m45.toml`'s (Gaussian PSF, movie data for the pack).
- **Night:** `date = "2027-10-31"`, Halloween with a new Moon.
  - At the site (lat 30.67, lon -104.02) the Moon is 8% lit and below the horizon for all dark hours.
  - On 2026-10-31 it is 56% lit and up for 6.2 of 10.2 h.
  - The engine has no Moon setting, so the year changed.
  - 238 β exposures through 21 telescopes.
- **Field:** f359 (Cassiopeia, l 116.6, b -1.0), background `f359_539p61_214_field`, layout real. The block holds 645
  Gaia stars, 296 of them G < 19, which hide the pumpkin in the exposure.
- **Jack-o-lantern:** 147 `[[place]]` stars, 9" apart, each moved up to 2" at random:
  - body outline 63, stem 6, filled triangle eyes 15 + 15, nose 7, mouth with two teeth 41;
  - 56 W UMa (`libnm_contact_*`, amplitude ≥ 0.5 mag) and 91 RRc (`lib_rrc_*`, ≥ 0.4 mag), all P ≤ 0.42 d, no model
    used twice;
  - mean G 15-18, drawn uniformly. Each star's distance is solved so its G holds through the 3D dust at that
    distance: E(B-V) 0.2-0.65 on this sightline, so ignoring the dust, as `[inject]` does, would have made them
    0.5-1.7 mag fainter;
  - random phases.
- **Light curves:** a scene follows only its placed stars (`choose_followed`; the field's Gaia variables are not
  followed), so all 6 drawn are pumpkin stars.
- **Preview:** scratchpad `spooky_preview.py` draws `spooky_preview.png` (temporary).
- **vega:**
  - The 85 library stars vega lacked were copied there (1.5 GB, `stars/`), along with the TOML and the tool.
  - The scene builds on vega with the queue's own command (checked, then removed).
  - In `outputs/rb_queue/vega_v2.queue` (backup `vega_v2.queue.bak_spooky_20261028`), the line
    `spooky_stars@20G ... observe.py examples/observations/spooky_stars.toml --name spooky_stars --bands beta` sits
    right after the running `rb3_f023_882p69_165_20261017_rho`. The user chose "next in line": the alerts chat's 52
    waiting rb3 nights each move back about 2.5 h.
- **Next:**
  1. When it ends: copy `outputs/pseudo_real/spooky_stars` to the Mac.
  2. Build the pack `--pack spooky_stars` (as M45: `build_pack.py`, ARGUS_PIPELINE=~/MESA_ARGUS, packs folder in the
     run).
  3. rsync it to huepil `packs_gaussian/` and run `prewarm.sh`.
  4. Menu label "Spooky stars": the app shows `FIELD_NAMES.get(k, k)`, so `"spooky_stars": "Spooky stars"` must be
     added to huepil's deployed `current/argus_night_live/app.py`. The demo chat needs the same line in the repo, or
     the next archive loses it.

### Spooky stars: first run, the live pack, and the re-run (2026-10-28 evening)

- **First run:** vega 13:07-19:55 (6.80 h), peak 20.0 GB plus 2.0 GB swap at its 20G cap, so nearly killed.
  - 238/238 frames; 141,556 candidates (truth: constant 111,416, none 24,625, variable 5,515).
  - Copied to the Mac, `outputs/pseudo_real/spooky_stars/`, without `hdps/` (2.8 GB, which the pack does not read) or
    `lightcurves/`.
  - The pack build takes about 20 min, against M45's 2: the engine setup does ArgusSim photometry for the 84,000
    catalogue stars.
  - The alerts chat removed its rb3 nights from vega's queue at 16:45, so vega is free.
- **Light curves:**
  - The engine's `choose_shown` picks the 6 that change most. In this crowded field those were the worst
    cross-matched (hdps frame catalogue): `measured - true` around +39 mag (near-zero flux), and an axis of ±40 mag.
  - Of the 147 followed stars, 99 were measured at all. Their 99.75th-percentile |measured - true| is < 0.2 mag for 9,
    < 0.5 for 24 and < 1 for 38.
  - The pack is built with scratchpad `huepil/build_pack_spooky.py`. It patches `night_movie.choose_shown` in that
    process only, to take the 6 with the smallest `rel["resid"]` among those spanning >= 0.3 mag (here resid 0.10-0.17,
    span 0.7-0.9). The engine is unchanged.
  - That pack is live on huepil and the panel is clean.
- **Menu label:** huepil's deployed `current/argus_night_live/app.py` has `"spooky_stars": "Spooky stars"` added to
  FIELD_NAMES (original kept as `app.py.orig_5cf6572`). The demo chat should add the same line to the repo.
- **The pumpkin in the images:**
  - Over the night, the difference image's per-pixel 90th-10th percentile traces the jack-o-lantern exactly at the
    placed positions (median 174 e/px there, against 18 at random points, so it is not mirrored).
  - In a single frame it is faint: random phases and G 15-18, beside the bright stars' subtraction residuals on the
    symlog ±1000 scale.
  - The alert markers ("!", 23 per exposure) trace it in both panels while "Show alerts" is on.
- **Re-run, `spooky_stars_v2`:**
  - `tools/spooky_stars.py` now has GLOW = 0.14 d: every star is at maximum light at once, mid-night. An RRc's epoch is
    its maximum (RSPStar: phase 0 = max), and a W UMa's is its primary eclipse (sed.h5 kind "EB", so phase 0 = file
    start = eclipse), so epoch = GLOW + 0.75 P.
  - G is 14.5-17.5.
  - Queued on vega as `spooky_stars_v2@21G ... --name spooky_stars_v2` (backup `vega_v2.queue.bak_spooky2_20261028`).
    It started 20:41 vega time; expect about 7 h.
- **When it ends:**
  1. Copy it to the Mac (without hdps/ and lightcurves/).
  2. Build `--pack spooky_stars` with `build_pack_spooky.py` from the v2 run.
  3. rsync to huepil and run `prewarm.sh`.
  4. Check the difference panel near UT 04:50 (mid-night): the whole jack-o-lantern should be red there.

### Spooky stars v3: the bright constant stars left out (2026-10-29)

User: "can you remove the bright contstant stars, i like the idea of the diff really revealing what the image is".

- **Why G 14.5:** in v1's difference images (alerts within the block, by the matched star's G), the constant stars
  that alert are G 11-13 at ~13000-32000 e- per alert (6 stars) and G 13-14.5 at ~500-670 e- (9 stars), against
  the pumpkin's 800-2500 e-. Fainter constant stars are ~180-250 e-. The cut is the pumpkin's brightest, G_RANGE[0].
- **How:** `tools/spooky_stars.py` writes the star set `stars/spooky_stars_field/`. It is `f359_539p61_214_field`
  without its 1634 stars brighter than G 14.5 (23 in the block). Stars with no Gaia G (136) are kept. The other
  tables are copied as they are, and centre and radius_deg are unchanged. `[stars] background` names it. No engine
  change. The [[place]] list is identical to v2's: only the background line differs.
- **Calibration:** the run uses TrueCalibrationTask, so hdps's WCS and zero point do not depend on the bright stars.
- **Queued on vega behind v2** as `spooky_stars_v3@21G ... --name spooky_stars_v3` (backup
  `vega_v2.queue.bak_spooky3_20261029`). The star set, the TOML and the tool were copied to vega.
- **The exposure gives the pumpkin away.** A drawing of the block at the page's stretch (asinh, vmax = 99.95th
  percentile), checked against v1's real frame 120, shows the outline in v1's real exposure already, and a little more
  with the cut. About 600 constant decoy stars at G 14.5-17.5 over the block would hide it (300 do not), at the cost
  of crowding. The user chose to keep v3 as queued (no decoys).
- **When v2 ends:** deploy it as planned (pack `spooky_stars`). When v3 ends, the same steps from
  `outputs/pseudo_real/spooky_stars_v3`.
- **v2 live (2026-10-29 07:46 UTC):**
  - Run: `spooky_stars_v2` ok on vega, 6.38 h, peak 21.0 GB. 142635 alerts; accepted truth: variable 8701 (v1: 5515).
  - Copied to `outputs/pseudo_real/spooky_stars_v2` without hdps/ and lightcurves/; pack built with
    `build_pack_spooky.py`. 121 of 147 followed stars measured (v1: 99); the 6 shown have resid 0.055-0.14 mag.
  - Pack sha256 5c13a3b1a719a958…, swapped into huepil `packs_gaussian/spooky_stars_beta.joblib`. v1's cache entries
    were removed and prewarm run. v1's pack stays at `outputs/pseudo_real/spooky_stars/pack/`.
  - Checked on the page (headless Chrome, exposure 124, UT 04:27, alerts off): the whole jack-o-lantern is red in the
    Difference panel. The light curves are clean: P9, P21, P79, P97, P128 and star A. The bright stars' dipoles remain;
    v3 removes them.
- **v3:** the background watcher reports when `spooky_stars_v3` ends. Deploy it the same way from
  `outputs/pseudo_real/spooky_stars_v3`.

### Spooky stars v3 ran, but hdps made no alerts (2026-10-29, for the alerts chat too)

- `spooky_stars_v3` ended "ok" on vega at 06:29 (3.43 h against v2's 6.38 h). It reports `hdps: 0 alerts from 238
  frames (0 frames hdps could not process)`, and the run folder has no `alerts.csv`. hdps/beta has the 238
  `transients_*.parquet` but no `alerts_*.parquet` or `alerts_stamps_*.npz`.
- **The difference images are fine.** In the block at exposure 124 the pumpkin glows with no bright-star dipoles (MAD 6.7
  e-/px against v2's 7.1). Scratchpad `spooky_v2_v3.png`.
- **The alert step returned at once.** hdps's AlertVettingTask spans last 0.00 s on all 238 frames (v2: 16-37 s,
  mean 24). SFFTSubtractionTask averages 9.9 s (v2: 30.3). DiffDetectionTask finds 500-2600 detections per frame,
  median S/N 8.9 and flux 369 (v2: 50-170 per frame, median S/N 91 and flux 14281, the bright stars' residuals).
- **Only the star set differs:** the resolved observation.toml differs from v2's in the name and `[stars] background`.
  The code is the same: no file under MESA_Argus, hdps_v4 or ~/.local site-packages changed between the runs.
- **Why there is no message:** hdps's module loggers (`logger.info`, e.g. "AlertVetting: no accepted alerts (%d
  candidates scored)") are not captured anywhere. hdps_worker sets up no logging.
- **Not yet known** which of AlertVettingTask's early returns fires (`_read_tile` -> None: no D, no Var(D), or < 5% valid).
- **Test running on vega:** `~/argus_work/spooky_ab/` holds `run_ab.sh` and `sitecustomize.py`. The patch wraps
  AlertVettingTask in that process only and prints its per-tile inputs. The test runs the science pass over UT 04:20-04:32
  for v2 and v3 against each run's own reference coadds. hdps is unedited.
- **The page keeps v2** until this is resolved, since the page shows alert counts.
- **v3 movie for Slack (2026-10-29):** `outputs/pseudo_real/spooky_stars_v3/spooky_stars_v3_night_beta.mp4` (9.9 s,
  24 fps, 1600x1000, 7.7 MB) and `..._slow.mp4` (2.4x slower with ffmpeg, 23.8 s, 13.8 MB). It was drawn from v3's
  movie data with scratchpad `render_spooky_movie.py` (night_movie --render-only with the run's arguments). The light
  curves are the 6 best-measured stars, as on the page. Gaussian PSF. No alert markers (v3 has none).
- **Cause found (test on vega, `~/argus_work/spooky_ab/`, v3 at stride 20, 12 frames):** every v3 tile reaches
  AlertVettingTask with `difference_variance` None and no `output_inverse_variance` (v2: present on every tile, valid
  3-56%). hdps's `ImageReducer` (hdps_calibration `image_reducer.py`, "Attach UncertaintyData if any calibration was
  applied OR any pixel is saturated") gives a frame its UncertaintyData only then. The simulation's chain has no master
  bias, dark or flat (bias_path/dark_path/flat_path "none"), so a frame gets a noise model only when some pixel reaches
  saturation_adu (58000). v3 had no star brighter than G 14.5 anywhere, so nothing saturated. The reprojector then
  propagates no inverse variance, SFFT's `_store_difference_variance` returns early (no Var(D)), and AlertVettingTask's
  `_read_tile` returns None for every tile. DiffDetectionTask, without Var(D), finds the 10x more low-S/N detections.
  **For the alerts chat:** any simulated frame without a saturated pixel loses its alerts silently (not seen in the
  real fields so far: every window has stars brighter than G ~11).
- **Fix for spooky stars (v4):** `tools/spooky_stars.py` CLEAR = 300": the stars brighter than G 14.5 are left out
  only within 300" of the centre (the block's cut-out is +-130"). Farther out they stay, so every frame saturates
  somewhere. hdps is unchanged.

### Route through Hank's :8444 tailscale serve (2026-10-29)

User: "what can i tell hank to make it so my link is as accesile as his is here https://huepil.tailab0df0.ts.net:8444/argus/"
- **Hank's route:** huepil's serve has `https://huepil.tailab0df0.ts.net:8444 (tailnet only) / -> http://127.0.0.1:8003`.
  It is tailscale serve, tailnet-only, and Hank's laptop reaches that port.
- **What Hank runs (user: tell him, do not ask):**
  `tailscale serve --bg --https=8444 --set-path /night-live http://100.91.210.25:8866/night-live`. It adds a path only;
  `/` and `/argus/` are untouched. To remove it: `tailscale serve --https=8444 --set-path /night-live off`.
- **Our side, done:** serve.sh `--Voila.tornado_settings` gained `"allow_origin": "https://huepil.tailab0df0.ts.net:8444"`
  (backup `serve.sh.bak_20261029`), and Voilà was restarted at 15:06 UTC. Tested: a websocket handshake to a kernel's
  channels with Origin `https://huepil.tailab0df0.ts.net:8444` and a mismatched Host (`100.91.210.25:8866`) gives 101;
  Origin `https://example.com` gives 403. The page works whether or not tailscale serve passes the browser's Host on.
- **Link once added:** https://huepil.tailab0df0.ts.net:8444/night-live/ (with the trailing slash). The nginx block
  (night-live.nginx) is no longer needed.

### Spooky stars v4 live (2026-10-30)

- **Run:** `spooky_stars_v4` (stars brighter than G 14.5 left out within 300" only; star set 84027 stars) ok on
  vega, 6.23 h, peak 21.0 GB. 144462 alerts from 238 frames; accepted truth: variable 13709 (v2: 8701), constant
  106549. The noise model is back: hdps/beta has alerts_*.parquet and stamps for every frame.
- **Difference image:** block difference at exposure 124 has MAD 6.8 e-/px and 99.9% |d| 510 (v2: 7.1 and 2215). The
  bright-star dipoles are gone and the pumpkin is all red (scratchpad `spooky_v2_v4.png`).
- **Pack:** built with `build_pack_spooky.py`. 134 of 147 measured; the 6 shown have resid 0.040-0.127 mag. Pack sha256
  f89e221472a27bfc…, swapped into huepil `packs_gaussian/spooky_stars_beta.joblib`; v2's cache entries removed,
  prewarm run (84038 field stars).
- **Checked on the page** (headless Chrome, exposure 124, alerts off): pumpkin red and clean in the Difference panel;
  light curves P54, P79, P129, P139, P130 and star A; star A alerted in 127/238 frames; the field's alerts 57.9 per
  exposure.
- **Earlier packs** stay on the Mac: v1 `outputs/pseudo_real/spooky_stars/pack/`, v2 `outputs/pseudo_real/spooky_stars_v2/pack/`.
