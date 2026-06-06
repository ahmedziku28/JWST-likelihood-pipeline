#!/usr/bin/env python3
"""notifier.py — standalone email alert daemon for the exo_de MCMC campaign.

Polls runs/ directory, detects status transitions, sends pretty HTML emails
via Gmail SMTP. Runs independently of run_manager.py — communicates only
through files on disk.

Place this file next to run_manager.py (inside your runs/ working directory).

Workflow:
    python notifier.py --setup       # one-time email/SMTP config
    python notifier.py --test        # verify SMTP and email rendering
    python notifier.py --list        # list pending alerts
    python notifier.py               # send pending alerts and exit
    python notifier.py --watch       # poll every 5 min in background (use tmux)
    python notifier.py --since DATE  # resend alerts after YYYY-MM-DD

Alert types:
    CONVERGED   — cobaya wrote "The run has converged!" to .log
    DEAD        — chains haven't been written in >30 min and job not in squeue
    STUCK       — health.json verdict is STUCK (debounced 24h per run)
    PROGRESS_80 — cobaya's R-1 first crossed below 0.10 (fires once per run)

Snapshot: runs/.notifier_snapshot.json    (per-run last-known status)
Log:      runs/.alerts.jsonl              (append-only, kept forever)
Config:   ~/.run_manager_email.json       (SMTP creds, chmod 600)
"""
from __future__ import print_function

import argparse
import getpass
import json
import os
import re
import signal
import smtplib
import subprocess
import sys
import time
from collections import deque
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# ── Paths (relative to script directory) ────────────────────────────────────

SCRIPT_DIR        = os.path.dirname(os.path.abspath(__file__))
RUNS_ROOT         = os.path.join(SCRIPT_DIR, 'runs')
ALERTS_FILE       = os.path.join(RUNS_ROOT, '.alerts.jsonl')
SNAPSHOT_FILE     = os.path.join(RUNS_ROOT, '.notifier_snapshot.json')
EMAIL_CONFIG_FILE = os.path.expanduser('~/.run_manager_email.json')

# ── Tunables ────────────────────────────────────────────────────────────────

WATCH_POLL_SECONDS       = 2000
STUCK_DEBOUNCE_HOURS     = 24.0
PROGRESS_80_THRESHOLD    = 0.10
DEAD_CHAIN_STALE_MINUTES = 30.0
LOG_TAIL_LINES_FOR_EMAIL = 10
SMTP_TIMEOUT_SECONDS     = 30


# ── Daemon management ───────────────────────────────────────────────────────

PIDFILE       = os.path.join(RUNS_ROOT, '.notifier.pid')
WATCH_LOG     = os.path.join(RUNS_ROOT, '.notifier_watch.log')

# ── Daemon PID file helpers ─────────────────────────────────────────────────

def _read_pidfile():
    if not os.path.isfile(PIDFILE):
        return None
    try:
        with open(PIDFILE) as f:
            return int(f.read().strip())
    except (ValueError, IOError, OSError):
        return None


def _is_pid_alive(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _write_pidfile(pid):
    try:
        with open(PIDFILE, 'w') as f:
            f.write(str(pid))
    except (IOError, OSError):
        pass


def _remove_pidfile():
    try:
        os.remove(PIDFILE)
    except OSError:
        pass


def _running_daemon_pid():
    """Return PID if daemon is running, else None. Cleans stale PID files."""
    pid = _read_pidfile()
    if _is_pid_alive(pid):
        return pid
    if pid is not None:
        _remove_pidfile()
    return None


# ── ANSI for terminal output (no external dep) ──────────────────────────────

ANSI = {
    'reset':  '\033[0m',
    'bold':   '\033[1m',
    'gray':   '\033[90m',
    'red':    '\033[31m',
    'green':  '\033[32m',
    'yellow': '\033[33m',
    'blue':   '\033[34m',
    'cyan':   '\033[36m',
}

def c(text, color):
    return ANSI.get(color, '') + str(text) + ANSI['reset']

def vprint(*args, **kwargs):
    """Print with a leading timestamp to stdout, flushed."""
    ts = datetime.now().strftime('%H:%M:%S')
    print(c('[' + ts + ']', 'gray'), *args, **kwargs)
    sys.stdout.flush()

# ───────────────────────────────────────────────────────────────────────────
# Run discovery — walk the campaign directory structure
# ───────────────────────────────────────────────────────────────────────────

def discover_runs():
    """Walk RUNS_ROOT to find every run directory.

    Returns list of dicts: {name, folder, model, shmr, zcut, has_uvlf,
    has_bg, has_cmb}. Derives metadata from the path structure and run name
    rather than parsing YAML — no external deps required.
    """
    runs = []

    # Production runs: runs/{exotic,lcdm}/{full,restr}/{fixed,vbeta,vshmr}/<name>/
    for model_dir in ('exotic', 'lcdm'):
        model = 'exo' if model_dir == 'exotic' else 'lcdm'
        for zcut_dir in ('full', 'restr'):
            for shmr_dir in ('fixed', 'vbeta', 'vshmr'):
                base = os.path.join(RUNS_ROOT, model_dir, zcut_dir, shmr_dir)
                if not os.path.isdir(base):
                    continue
                for entry in os.listdir(base):
                    folder = os.path.join(base, entry)
                    if not os.path.isfile(os.path.join(folder, entry + '.yaml')):
                        continue
                    parts = entry.split('_')
                    runs.append({
                        'name':     entry,
                        'folder':   folder,
                        'model':    model,
                        'shmr':     shmr_dir,
                        'zcut':     zcut_dir,
                        'has_uvlf': any(p in parts for p in ('uvlf', 'ceers', 'primer')),
                        'has_bg':   'bg'  in parts,
                        'has_cmb': 'cmb' in parts,
                    })

    # Non-UVLF runs: runs/non_uvlf/<name>/
    non_uvlf_root = os.path.join(RUNS_ROOT, 'non_uvlf')
    if os.path.isdir(non_uvlf_root):
        for entry in os.listdir(non_uvlf_root):
            folder = os.path.join(non_uvlf_root, entry)
            if not os.path.isfile(os.path.join(folder, entry + '.yaml')):
                continue
            parts = entry.split('_')
            runs.append({
                'name':     entry,
                'folder':   folder,
                'model':    'exo' if entry.startswith('exo') else 'lcdm',
                'shmr':     'fixed',
                'zcut':     'full',
                'has_uvlf': False,
                'has_bg':   'bg'  in parts,
                'has_cmb':  'cmb' in parts,
            })

    return runs

# ───────────────────────────────────────────────────────────────────────────
# Per-run status detection
# ───────────────────────────────────────────────────────────────────────────

def log_contains_converged(folder, run_name, n_lines=50):
    """Check last n_lines of .log for cobaya's convergence message."""
    log_path = os.path.join(folder, run_name + '.log')
    if not os.path.isfile(log_path):
        return False
    try:
        with open(log_path) as f:
            tail = deque(f, maxlen=n_lines)
    except (IOError, OSError):
        return False
    return any('The run has converged!' in line for line in tail)


def chain_mtimes(folder, run_name):
    """Return list of chain .txt mtimes (POSIX seconds), empty if none."""
    outputs = os.path.join(folder, 'outputs')
    if not os.path.isdir(outputs):
        return []
    mtimes = []
    for fname in os.listdir(outputs):
        if not (fname.startswith(run_name + '.') and fname.endswith('.txt')):
            continue
        try:
            mtimes.append(os.path.getmtime(os.path.join(outputs, fname)))
        except OSError:
            continue
    return mtimes


def chains_stale_minutes(folder, run_name):
    """Minutes since the most-recent chain .txt mtime; inf if none."""
    mtimes = chain_mtimes(folder, run_name)
    if not mtimes:
        return float('inf')
    age_sec = (datetime.now() - datetime.fromtimestamp(max(mtimes))).total_seconds()
    return age_sec / 60.0


def chains_exist(folder, run_name):
    return len(chain_mtimes(folder, run_name)) > 0


def squeue_running_names():
    """Set of run names currently in squeue for the current user."""
    user = os.environ.get('USER') or getpass.getuser()
    try:
        r = subprocess.run(
            ['squeue', '-h', '-u', user, '-o', '%j'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15,
        )
        if r.returncode != 0:
            return set()
        text = r.stdout.decode('utf-8', errors='ignore')
        return set(line.strip() for line in text.split('\n') if line.strip())
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return set()


def get_run_status(run, squeue_names):
    """Determine status string: CONVERGED, RUNNING, DEAD, PENDING."""
    if log_contains_converged(run['folder'], run['name']):
        return 'CONVERGED'
    if run['name'] in squeue_names:
        return 'RUNNING'
    if not chains_exist(run['folder'], run['name']):
        return 'PENDING'
    # Has chains, not in squeue, not converged.
    if chains_stale_minutes(run['folder'], run['name']) < DEAD_CHAIN_STALE_MINUTES:
        # squeue may lag relative to chain activity; treat as still running.
        return 'RUNNING'
    return 'DEAD'


def read_cobaya_rminus1(folder, run_name):
    """Last R-1 value (column 4) from cobaya's .progress file. None if missing."""
    progress_path = os.path.join(folder, 'outputs', run_name + '.progress')
    if not os.path.isfile(progress_path):
        return None
    last = None
    try:
        with open(progress_path) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    last = parts
    except (IOError, OSError):
        return None
    if last is None:
        return None
    try:
        return float(last[3])
    except (ValueError, IndexError):
        return None


def read_cobaya_n_samples(folder, run_name):
    """Last sample count (column 1) from .progress, as int. None if missing."""
    progress_path = os.path.join(folder, 'outputs', run_name + '.progress')
    if not os.path.isfile(progress_path):
        return None
    last = None
    try:
        with open(progress_path) as f:
            for line in f:
                if line.startswith('#'):
                    continue
                parts = line.split()
                if len(parts) >= 1:
                    last = parts
    except (IOError, OSError):
        return None
    if last is None:
        return None
    try:
        return int(float(last[0]))
    except (ValueError, IndexError):
        return None


def read_log_tail(folder, run_name, n_lines=LOG_TAIL_LINES_FOR_EMAIL):
    log_path = os.path.join(folder, run_name + '.log')
    if not os.path.isfile(log_path):
        return ''
    try:
        with open(log_path) as f:
            tail = deque(f, maxlen=n_lines)
    except (IOError, OSError):
        return ''
    return ''.join(tail).rstrip()


def read_health(folder):
    health_path = os.path.join(folder, '.health.json')
    if not os.path.isfile(health_path):
        return None
    try:
        with open(health_path) as f:
            return json.load(f)
    except (IOError, OSError, ValueError):
        return None

# ───────────────────────────────────────────────────────────────────────────
# Alerts log (append-only JSONL)
# ───────────────────────────────────────────────────────────────────────────

def append_alert(event):
    """Append one alert record to ALERTS_FILE."""
    event = dict(event)
    event.setdefault('ts', datetime.now().isoformat())
    event.setdefault('sent', False)
    # Stable ID: kind + run + ts. Lets us mark-sent without ambiguity.
    event.setdefault('id', '{}_{}_{}'.format(
        event.get('kind', '?'), event.get('run', '?'), event['ts']))
    try:
        os.makedirs(os.path.dirname(ALERTS_FILE), exist_ok=True)
        with open(ALERTS_FILE, 'a') as f:
            f.write(json.dumps(event) + '\n')
    except (IOError, OSError) as e:
        vprint(c('WARNING: failed to write alert: {}'.format(e), 'red'))


def load_alerts(pending_only=False, since=None):
    """Return list of alert dicts. Filter optional."""
    if not os.path.isfile(ALERTS_FILE):
        return []
    out = []
    try:
        with open(ALERTS_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if pending_only and rec.get('sent'):
                    continue
                if since is not None:
                    try:
                        ts = datetime.fromisoformat(rec.get('ts', ''))
                    except ValueError:
                        continue
                    if ts < since:
                        continue
                out.append(rec)
    except (IOError, OSError):
        return []
    return out


def mark_alerts_sent(alert_ids):
    """Atomic rewrite of ALERTS_FILE with `sent=True` set on alert_ids."""
    if not alert_ids or not os.path.isfile(ALERTS_FILE):
        return
    id_set = set(alert_ids)
    try:
        with open(ALERTS_FILE) as f:
            lines = f.readlines()
    except (IOError, OSError):
        return
    rewritten = []
    for line in lines:
        try:
            rec = json.loads(line.strip())
        except (ValueError, AttributeError):
            rewritten.append(line)
            continue
        if rec.get('id') in id_set:
            rec['sent']    = True
            rec['sent_at'] = datetime.now().isoformat()
        rewritten.append(json.dumps(rec) + '\n')
    tmp = ALERTS_FILE + '.tmp'
    try:
        with open(tmp, 'w') as f:
            f.writelines(rewritten)
        os.replace(tmp, ALERTS_FILE)
    except (IOError, OSError) as e:
        vprint(c('WARNING: failed to mark alerts sent: {}'.format(e), 'red'))


def last_stuck_alert_age_hours(run_name):
    """Hours since most-recent STUCK alert for this run, or None."""
    alerts = load_alerts()
    most_recent = None
    for a in alerts:
        if a.get('kind') == 'STUCK' and a.get('run') == run_name:
            try:
                ts = datetime.fromisoformat(a.get('ts', ''))
            except ValueError:
                continue
            if most_recent is None or ts > most_recent:
                most_recent = ts
    if most_recent is None:
        return None
    return (datetime.now() - most_recent).total_seconds() / 3600.0


def progress_80_already_fired(run_name):
    for a in load_alerts():
        if a.get('kind') == 'PROGRESS_80' and a.get('run') == run_name:
            return True
    return False

# ───────────────────────────────────────────────────────────────────────────
# Snapshot (per-run last-known status for transition detection)
# ───────────────────────────────────────────────────────────────────────────

def load_snapshot():
    if not os.path.isfile(SNAPSHOT_FILE):
        return {}
    try:
        with open(SNAPSHOT_FILE) as f:
            data = json.load(f)
        if isinstance(data, dict) and 'runs' in data:
            return data.get('runs', {})
    except (IOError, OSError, ValueError):
        pass
    return {}


def save_snapshot(snapshot):
    payload = {
        'updated_at': datetime.now().isoformat(),
        'runs':       snapshot,
    }
    tmp = SNAPSHOT_FILE + '.tmp'
    try:
        os.makedirs(os.path.dirname(SNAPSHOT_FILE), exist_ok=True)
        with open(tmp, 'w') as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, SNAPSHOT_FILE)
    except (IOError, OSError) as e:
        vprint(c('WARNING: failed to save snapshot: {}'.format(e), 'red'))

# ───────────────────────────────────────────────────────────────────────────
# Alert builders
# ───────────────────────────────────────────────────────────────────────────

def build_converged_alert(run):
    return {
        'kind':     'CONVERGED',
        'run':      run['name'],
        'model':    run['model'],
        'shmr':     run['shmr'],
        'zcut':     run['zcut'],
        'has_uvlf': run['has_uvlf'],
        'has_bg':   run['has_bg'],
        'has_cmb':  run['has_cmb'],
        'folder':   run['folder'],
        'rminus1':    read_cobaya_rminus1(run['folder'], run['name']),
        'n_samples':  read_cobaya_n_samples(run['folder'], run['name']),
    }


def build_dead_alert(run):
    return {
        'kind':              'DEAD',
        'run':               run['name'],
        'model':             run['model'],
        'shmr':              run['shmr'],
        'zcut':              run['zcut'],
        'has_uvlf':          run['has_uvlf'],
        'has_bg':            run['has_bg'],
        'has_cmb':           run['has_cmb'],
        'folder':            run['folder'],
        'chains_stale_min':  round(chains_stale_minutes(run['folder'], run['name']), 1),
        'last_log_tail':     read_log_tail(run['folder'], run['name']),
    }


def build_stuck_alert(run, health):
    return {
        'kind':       'STUCK',
        'run':        run['name'],
        'model':      run['model'],
        'shmr':       run['shmr'],
        'zcut':       run['zcut'],
        'has_uvlf':   run['has_uvlf'],
        'has_bg':     run['has_bg'],
        'has_cmb':    run['has_cmb'],
        'folder':     run['folder'],
        'rminus1':    health.get('rminus1'),
        'rminus1_cl': health.get('rminus1_cl'),
        'acceptance': health.get('acceptance'),
        'bottleneck': health.get('bottleneck'),
        'n_samples':  health.get('n_samples'),
    }


def build_progress80_alert(run, cobaya_r1):
    return {
        'kind':       'PROGRESS_80',
        'run':        run['name'],
        'model':      run['model'],
        'shmr':       run['shmr'],
        'zcut':       run['zcut'],
        'has_uvlf':   run['has_uvlf'],
        'has_bg':     run['has_bg'],
        'has_cmb':    run['has_cmb'],
        'folder':     run['folder'],
        'rminus1':    cobaya_r1,
        'n_samples':  read_cobaya_n_samples(run['folder'], run['name']),
    }

# ───────────────────────────────────────────────────────────────────────────
# Transition detection — the heart of the daemon
# ───────────────────────────────────────────────────────────────────────────

def detect_transitions(runs, prev_snapshot):
    """Compare current run statuses against prev_snapshot. Append + return
    new alerts. Returns (alerts, current_snapshot)."""
    squeue_names = squeue_running_names()
    new_alerts = []
    current = {}

    for run in runs:
        status = get_run_status(run, squeue_names)
        current[run['name']] = status
        prev = prev_snapshot.get(run['name'])

        # First-time observation — record only, no alerts (avoid spam on
        # initial daemon startup where every existing converged run would fire).
        if prev is None:
            continue

        # CONVERGED transition
        if status == 'CONVERGED' and prev != 'CONVERGED':
            ev = build_converged_alert(run)
            append_alert(ev)
            new_alerts.append(ev)
            continue

        # DEAD transition
        if status == 'DEAD' and prev != 'DEAD':
            ev = build_dead_alert(run)
            append_alert(ev)
            new_alerts.append(ev)
            continue

        # RUNNING — check STUCK and PROGRESS_80
        if status == 'RUNNING':
            # STUCK (debounced)
            health = read_health(run['folder'])
            if health and health.get('verdict') == 'STUCK':
                age = last_stuck_alert_age_hours(run['name'])
                if age is None or age > STUCK_DEBOUNCE_HOURS:
                    ev = build_stuck_alert(run, health)
                    append_alert(ev)
                    new_alerts.append(ev)

            # PROGRESS_80 — once ever
            if not progress_80_already_fired(run['name']):
                r1 = read_cobaya_rminus1(run['folder'], run['name'])
                if r1 is not None and r1 < PROGRESS_80_THRESHOLD:
                    ev = build_progress80_alert(run, r1)
                    append_alert(ev)
                    new_alerts.append(ev)

    return new_alerts, current

# ───────────────────────────────────────────────────────────────────────────
# HTML email rendering — light/dark via prefers-color-scheme
# ───────────────────────────────────────────────────────────────────────────

# Alert theming: (light_accent, dark_accent, label, glyph)
ALERT_THEME = {
    'CONVERGED':   ('#0d8a64', '#3fcaa3', 'Converged',                '◆'),
    'DEAD':        ('#b03434', '#ef6b6b', 'Dead',                     '◇'),
    'STUCK':       ('#a35100', '#f0b264', 'Stuck',                    '◑'),
    'PROGRESS_80': ('#1f5dd3', '#7aa5f5', 'Approaching threshold',    '◐'),
}


def html_escape(s):
    return (str(s)
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;'))


def format_email_html(event):
    """Render one alert as (subject, html_body)."""
    kind = event.get('kind', '?')
    run  = event.get('run', '?')
    light_accent, dark_accent, label, glyph = ALERT_THEME.get(
        kind, ('#666', '#999', kind, '·'))

    subject = '[{}] {} — {}'.format(label, run, datetime.now().strftime('%H:%M'))

    # Configuration descriptors
    cfg_parts = []
    if event.get('model'):
        cfg_parts.append(html_escape(event['model']))
    if event.get('shmr'):
        cfg_parts.append('SHMR&nbsp;' + html_escape(event['shmr']))
    if event.get('zcut'):
        cfg_parts.append('z-cut&nbsp;' + html_escape(event['zcut']))
    flags = []
    if event.get('has_uvlf'): flags.append('UVLF')
    if event.get('has_bg'):   flags.append('BG')
    if event.get('has_cmb'):  flags.append('CMB')
    if flags:
        cfg_parts.append(' + '.join(flags))
    cfg_line = ' · '.join(cfg_parts) if cfg_parts else ''

    # Stat rows
    stat_rows = []
    def add_stat(k, v):
        if v is not None and v != '':
            stat_rows.append((k, v))
    if event.get('rminus1') is not None:
        add_stat('R-1 (means)', '{:.4f}'.format(event['rminus1']))
    if event.get('rminus1_cl') is not None:
        add_stat('R-1 (CL)', '{:.4f}'.format(event['rminus1_cl']))
    if event.get('acceptance') is not None:
        add_stat('Acceptance', '{:.3f}'.format(event['acceptance']))
    if event.get('n_samples') is not None:
        add_stat('Samples', '{:,}'.format(event['n_samples']))
    if event.get('bottleneck'):
        add_stat('Bottleneck', event['bottleneck'])
    if event.get('chains_stale_min') is not None:
        add_stat('Chains stale for', '{:.1f} min'.format(event['chains_stale_min']))

    stats_html = ''
    for k, v in stat_rows:
        stats_html += (
            '<tr>'
            '<td class="k" style="padding:7px 18px 7px 0;color:#7a7a7a;font-size:12px;'
            'font-weight:500;letter-spacing:0.3px;text-transform:uppercase;'
            'font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',sans-serif;'
            'vertical-align:top;">{}</td>'
            '<td class="v" style="padding:7px 0;font-family:\'JetBrains Mono\','
            'ui-monospace,SFMono-Regular,Menlo,monospace;font-size:14px;'
            'color:#1a1a1a;">{}</td>'
            '</tr>'
        ).format(html_escape(k), html_escape(v))

    # Action / context block
    action_text = {
        'CONVERGED':   'Chains satisfied cobaya\'s stopping criterion. '
                       'The covmat now in <code>outputs/{run}.covmat</code> '
                       'can seed downstream runs via <code>apply_covmats.py</code>.',
        'DEAD':        'Chains stopped updating and the job exited squeue. '
                       'Likely walltime or crash. Inspect the log tail below, '
                       'then <code>resubmit</code> to resume from the checkpoint.',
        'STUCK':       'Health verdict registered STUCK — R-1 plateaued or '
                       'rising for multiple checkpoints. Consider '
                       '<code>reset</code> (warm restart, keep covmat) or '
                       '<code>restart</code> (cold, wipe covmat).',
        'PROGRESS_80': 'cobaya\'s internal R-1 first crossed below 0.10. '
                       'Endgame phase — no action needed. Convergence likely '
                       'within hours unless CL R-1 is tail-bottlenecked.',
    }.get(kind, '').replace('{run}', html_escape(run))

    # Log tail block (DEAD only)
    log_block = ''
    if event.get('last_log_tail'):
        log_block = (
            '<div class="log" style="margin-top:24px;padding:14px 18px;'
            'background:#f4f4f0;border-radius:4px;font-family:\'JetBrains Mono\','
            'ui-monospace,SFMono-Regular,Menlo,monospace;font-size:11.5px;'
            'line-height:1.6;white-space:pre-wrap;color:#3d3d3d;'
            'border-left:3px solid {accent};">'
            '<div style="color:#888;font-family:-apple-system,sans-serif;'
            'font-size:10.5px;margin-bottom:10px;text-transform:uppercase;'
            'letter-spacing:0.6px;font-weight:600;">Last {n} log lines</div>'
            '{tail}'
            '</div>'
        ).format(
            accent=light_accent,
            n=LOG_TAIL_LINES_FOR_EMAIL,
            tail=html_escape(event['last_log_tail']))

    folder_display = html_escape(event.get('folder', '').replace(SCRIPT_DIR + '/', ''))
    ts = datetime.now().strftime('%Y-%m-%d · %H:%M')

    # The full HTML. All CSS inline; one <style> block for prefers-color-scheme
    # since inline styles can't carry media queries. Most major clients honor it.
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="light dark">
<meta name="supported-color-schemes" content="light dark">
<title>{subject}</title>
<style>
  body {{ margin:0; padding:32px 16px; background:#f5f5f1;
          font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
          color:#1a1a1a; -webkit-font-smoothing:antialiased; }}
  .card {{ background:#ffffff; border:1px solid #e6e6e0; border-radius:6px;
           padding:36px 40px; }}
  .pill {{ display:inline-block; padding:5px 13px; border-radius:3px;
           background:{light_accent}; color:#ffffff; font-size:10.5px;
           font-weight:700; letter-spacing:1.2px; text-transform:uppercase;
           margin-bottom:22px; }}
  .runname {{ font-family:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace;
              font-size:19px; font-weight:600; margin-bottom:6px; word-break:break-all;
              color:#0a0a0a; line-height:1.3; }}
  .cfg {{ color:#7a7a7a; font-size:13px; margin-bottom:28px; letter-spacing:0.1px; }}
  .action {{ margin-top:26px; padding-top:20px; border-top:1px solid #ececec;
             font-size:13.5px; line-height:1.65; color:#3d3d3d; }}
  .action code {{ background:#f0f0eb; padding:1px 6px; border-radius:3px;
                  font-family:'JetBrains Mono',ui-monospace,monospace;
                  font-size:12px; color:#1a1a1a; }}
  .footer {{ margin-top:28px; padding-top:18px; border-top:1px solid #ececec;
             font-size:10.5px; color:#a0a0a0; letter-spacing:0.5px;
             text-transform:uppercase; font-weight:500;
             display:flex; justify-content:space-between; }}
  .folder {{ font-family:'JetBrains Mono',ui-monospace,monospace;
             text-transform:none; letter-spacing:0; color:#9a9a9a; font-size:11px; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background:#0e0e0c !important; color:#e8e8e3 !important; }}
    .card {{ background:#1a1a17 !important; border-color:#2a2a26 !important; }}
    .pill {{ background:{dark_accent} !important; color:#0e0e0c !important; }}
    .runname {{ color:#f5f5f0 !important; }}
    .cfg {{ color:#a8a8a0 !important; }}
    .k {{ color:#888880 !important; }}
    .v {{ color:#e8e8e3 !important; }}
    .action {{ color:#c8c8c0 !important; border-top-color:#2a2a26 !important; }}
    .action code {{ background:#2a2a26 !important; color:#e8e8e3 !important; }}
    .log {{ background:#0e0e0c !important; color:#c8c8c0 !important;
            border-left-color:{dark_accent} !important; }}
    .footer {{ border-top-color:#2a2a26 !important; color:#666660 !important; }}
    .folder {{ color:#666660 !important; }}
  }}
</style>
</head>
<body>
<table role="presentation" cellspacing="0" cellpadding="0" border="0" align="center"
       style="max-width:600px;width:100%;margin:0 auto;border-collapse:collapse;">
<tr><td>
<div class="card">
  <span class="pill">{glyph}&nbsp;&nbsp;{label}</span>
  <div class="runname">{run}</div>
  <div class="cfg">{cfg_line}</div>
  <table role="presentation" cellspacing="0" cellpadding="0" border="0"
         style="width:100%;border-collapse:collapse;">
    {stats_html}
  </table>
  <div class="action">{action_text}</div>
  {log_block}
  <div class="footer">
    <span>{ts}</span>
    <span class="folder">{folder_display}</span>
  </div>
</div>
</td></tr>
</table>
</body>
</html>""".format(
        subject=html_escape(subject),
        light_accent=light_accent, dark_accent=dark_accent,
        glyph=glyph, label=html_escape(label),
        run=html_escape(run), cfg_line=cfg_line,
        stats_html=stats_html, action_text=action_text,
        log_block=log_block, ts=ts, folder_display=folder_display,
    )

    return subject, html

# ───────────────────────────────────────────────────────────────────────────
# SMTP send + config
# ───────────────────────────────────────────────────────────────────────────

def load_email_config():
    if not os.path.isfile(EMAIL_CONFIG_FILE):
        return None
    try:
        with open(EMAIL_CONFIG_FILE) as f:
            return json.load(f)
    except (IOError, OSError, ValueError):
        return None


def save_email_config(cfg):
    try:
        with open(EMAIL_CONFIG_FILE, 'w') as f:
            json.dump(cfg, f, indent=2)
        os.chmod(EMAIL_CONFIG_FILE, 0o600)
        return True
    except (IOError, OSError) as e:
        vprint(c('failed to write config: {}'.format(e), 'red'))
        return False


def send_email(subject, html, config):
    """Returns (success_bool, message_str)."""
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From']    = config['from']
    msg['To']      = config['to']
    msg.attach(MIMEText(
        'This is an HTML email; please view in an HTML-capable client.', 'plain'))
    msg.attach(MIMEText(html, 'html'))

    try:
        with smtplib.SMTP(config['smtp_server'], int(config['smtp_port']),
                          timeout=SMTP_TIMEOUT_SECONDS) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.ehlo()
            smtp.login(config['from'], config['password'])
            smtp.sendmail(config['from'], [config['to']], msg.as_string())
        return True, 'ok'
    except smtplib.SMTPAuthenticationError as e:
        return False, 'auth failed (check app password): {}'.format(e)
    except smtplib.SMTPException as e:
        return False, 'SMTP error: {}'.format(e)
    except Exception as e:
        return False, '{}: {}'.format(type(e).__name__, e)
def send_admin_email(subject, html):
    # type: (str, str) -> Tuple[bool, str]
    """Synchronous one-shot email send for external callers (e.g. run_manager
    auto-daemon notifications). Loads the email config, sends, returns
    (success, message). Safe to call from any process — does not require the
    notifier daemon to be running."""
    config = load_email_config()
    if config is None:
        return False, 'no email config; run: python notifier.py --setup'
    return send_email(subject, html, config)


# ───────────────────────────────────────────────────────────────────────────
# Commands
# ───────────────────────────────────────────────────────────────────────────

def cmd_setup():
    print()
    print(c('  Email notifier setup', 'bold'))
    print('  ' + '─' * 60)
    print('  Gmail requires an "app password" — generate one at:')
    print(c('    https://myaccount.google.com/apppasswords', 'cyan'))
    print()

    existing = load_email_config() or {}

    def ask(prompt, default=None, secret=False):
        suffix = ' [{}]'.format('***hidden***' if secret and default else default) if default else ''
        if secret:
            v = getpass.getpass('  {}{}: '.format(prompt, suffix))
        else:
            v = input('  {}{}: '.format(prompt, suffix)).strip()
        return v or default

    cfg = {
        'from':        ask('Your Gmail address',  existing.get('from')),
        'to':          ask('Send alerts to',      existing.get('to') or existing.get('from')),
        'smtp_server': ask('SMTP server',         existing.get('smtp_server') or 'smtp.gmail.com'),
        'smtp_port':   ask('SMTP port',           existing.get('smtp_port') or '587'),
        'password':    ask('App password (16 chars, no spaces)',
                            existing.get('password'), secret=True),
    }

    missing = [k for k, v in cfg.items() if not v]
    if missing:
        print(c('  Missing fields: {}'.format(', '.join(missing)), 'red'))
        return

    if save_email_config(cfg):
        print()
        print(c('  ✓ Saved to {} (mode 600)'.format(EMAIL_CONFIG_FILE), 'green'))
        print('  Run: python notifier.py --test')


def cmd_test():
    config = load_email_config()
    if config is None:
        print(c('  No email config. Run: python notifier.py --setup', 'red'))
        return

    # One sample of each alert type, so you can see all four templates rendered.
    samples = [
        {
            'kind': 'CONVERGED', 'run': 'exo_uvlf_bg_cmb_fixed_full',
            'model': 'exo', 'shmr': 'fixed', 'zcut': 'full',
            'has_uvlf': True, 'has_bg': True, 'has_cmb': True,
            'rminus1': 0.0184, 'n_samples': 18540,
            'folder': '/path/to/runs/exotic/full/fixed/exo_uvlf_bg_cmb_fixed_full',
        },
    ]
    sent = 0
    for s in samples:
        subj, html = format_email_html(s)
        ok, msg = send_email(subj, html, config)
        if ok:
            sent += 1
            print(c('  ✓ Sent test ({}) to {}'.format(s['kind'], config['to']), 'green'))
        else:
            print(c('  ✗ {}: {}'.format(s['kind'], msg), 'red'))

    if sent > 0:
        print()
        print(c('  Check your inbox + try toggling system dark mode to see both themes.',
                'gray'))


def cmd_list():
    pending = load_alerts(pending_only=True)
    if not pending:
        print('  No pending alerts.')
        return
    print(c('  {} pending alert(s):'.format(len(pending)), 'bold'))
    for a in pending:
        ts = a.get('ts', '?')[:19].replace('T', ' ')
        print('    {}  {:<12s}  {}'.format(
            ts, a.get('kind', '?'), a.get('run', '?')))


def cmd_send_pending(since=None):
    """Send all unsent alerts (or, if since is set, re-send everything after that)."""
    config = load_email_config()
    if config is None:
        print(c('  No email config. Run: python notifier.py --setup', 'red'))
        return 0, 0

    if since is None:
        alerts = load_alerts(pending_only=True)
    else:
        # When --since is set we re-send regardless of sent flag.
        alerts = load_alerts(pending_only=False, since=since)

    if not alerts:
        return 0, 0

    sent_ok_ids = []
    failed = 0
    for a in alerts:
        subj, html = format_email_html(a)
        ok, msg = send_email(subj, html, config)
        if ok:
            sent_ok_ids.append(a['id'])
            vprint(c('sent: {} — {}'.format(a['kind'], a['run']), 'green'))
        else:
            failed += 1
            vprint(c('failed: {} — {} ({})'.format(a['kind'], a['run'], msg), 'red'))

    if since is None:
        mark_alerts_sent(sent_ok_ids)

    return len(sent_ok_ids), failed


def cmd_detect_once():
    """One detection pass — used both by --watch and by manual one-shot runs."""
    runs = discover_runs()
    if not runs:
        vprint(c('no runs found in {}'.format(RUNS_ROOT), 'yellow'))
        return 0
    prev = load_snapshot()
    new_alerts, current = detect_transitions(runs, prev)
    save_snapshot(current)
    if new_alerts:
        for a in new_alerts:
            vprint(c('alert: {} — {}'.format(a['kind'], a['run']), 'cyan'))
    return len(new_alerts)

def cmd_watch_daemon():
    """Internal — the actual polling loop. Invoked as a detached subprocess
    by cmd_watch. Writes all output to WATCH_LOG. Handles SIGTERM cleanly."""
    config = load_email_config()
    if config is None:
        print('No email config; daemon cannot start.')
        sys.exit(1)

    def _shutdown(signum, _frame):
        vprint('daemon received signal {}, shutting down cleanly'.format(signum))
        _remove_pidfile()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGHUP,  signal.SIG_IGN)   # survive terminal hangup

    vprint('daemon started, polling every {}s'.format(WATCH_POLL_SECONDS))

    try:
        while True:
            n_new = cmd_detect_once()
            if n_new > 0:
                sent, failed = cmd_send_pending()
                vprint('detection cycle: {} new alerts, {} sent, {} failed'.format(
                    n_new, sent, failed))
            else:
                sent, failed = cmd_send_pending()
                if sent > 0 or failed > 0:
                    vprint('retried pending: {} sent, {} failed'.format(sent, failed))
            time.sleep(WATCH_POLL_SECONDS)
    finally:
        _remove_pidfile()


def cmd_tail_watch_log():
    """Attach to the running daemon's log via tail -f. Ctrl-C exits the tail
    but leaves the daemon running."""
    if not os.path.exists(WATCH_LOG):
        print(c('  No watch log yet at {}'.format(WATCH_LOG), 'yellow'))
        print(c('  (The daemon may have just started — try again in 10s.)', 'gray'))
        return
    print(c('  Attaching to daemon log. Ctrl-C exits THIS VIEW only — daemon keeps running.',
            'gray'))
    print(c('  Log file: {}'.format(WATCH_LOG), 'gray'))
    print()
    try:
        subprocess.run(['tail', '-n', '20', '-f', WATCH_LOG])
    except KeyboardInterrupt:
        pass
    print()
    pid = _running_daemon_pid()
    if pid is not None:
        print(c('  Detached from log view. Daemon still running (PID {}).'.format(pid),
                'green'))
        print(c('  Stop daemon with: python notifier.py --stop', 'gray'))
    else:
        print(c('  Detached. (Daemon does not appear to be running.)', 'yellow'))


def cmd_watch():
    """If daemon not running, spawn it detached; then attach the log tail."""
    existing = _running_daemon_pid()
    if existing is not None:
        print(c('  Daemon already running (PID {})'.format(existing), 'green'))
        print()
        cmd_tail_watch_log()
        return

    if load_email_config() is None:
        print(c('  No email config. Run: python notifier.py --setup', 'red'))
        return

    print(c('  Starting detached daemon...', 'bold'))
    try:
        log_fd = open(WATCH_LOG, 'a')
        log_fd.write('\n--- daemon spawn at {} ---\n'.format(datetime.now().isoformat()))
        log_fd.flush()
        proc = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), '--watch-daemon'],
            stdout=log_fd,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,  # detach from controlling terminal
            close_fds=True,
        )
    except Exception as e:
        print(c('  Failed to spawn daemon: {}'.format(e), 'red'))
        return

    _write_pidfile(proc.pid)
    time.sleep(1.5)  # give it a moment to actually start

    if not _is_pid_alive(proc.pid):
        _remove_pidfile()
        print(c('  Daemon died immediately. Check log: {}'.format(WATCH_LOG), 'red'))
        return

    print(c('  ✓ Daemon started (PID {})'.format(proc.pid), 'green'))
    print(c('    PID file:  {}'.format(PIDFILE), 'gray'))
    print(c('    Log file:  {}'.format(WATCH_LOG), 'gray'))
    print()
    cmd_tail_watch_log()


def cmd_stop():
    pid = _running_daemon_pid()
    if pid is None:
        print(c('  No daemon running.', 'gray'))
        return
    print(c('  Stopping daemon (PID {})...'.format(pid), 'bold'))
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as e:
        print(c('  Could not signal daemon: {}'.format(e), 'red'))
        return
    for _ in range(10):
        time.sleep(0.5)
        if not _is_pid_alive(pid):
            _remove_pidfile()
            print(c('  ✓ Daemon stopped cleanly.', 'green'))
            return
    print(c('  Daemon did not exit in 5s. Sending SIGKILL.', 'yellow'))
    try:
        os.kill(pid, signal.SIGKILL)
    except OSError:
        pass
    _remove_pidfile()
    print(c('  ✓ Daemon killed.', 'green'))


def cmd_daemon_status():
    pid = _running_daemon_pid()
    if pid is None:
        print(c('  Daemon status: not running', 'yellow'))
        if os.path.exists(WATCH_LOG):
            print(c('  Last log:      {}'.format(WATCH_LOG), 'gray'))
        return
    print(c('  Daemon status: running (PID {})'.format(pid), 'green'))
    print(c('  PID file:      {}'.format(PIDFILE), 'gray'))
    print(c('  Log file:      {}'.format(WATCH_LOG), 'gray'))
    if os.path.exists(WATCH_LOG):
        try:
            with open(WATCH_LOG) as f:
                lines = deque(f, maxlen=8)
            if lines:
                print()
                print(c('  Recent log output:', 'bold'))
                for line in lines:
                    print('    ' + line.rstrip())
        except (IOError, OSError):
            pass

# ───────────────────────────────────────────────────────────────────────────
# Entry point
# ───────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Email alert daemon for the exo_de MCMC campaign.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--setup', action='store_true',
                        help='one-time email/SMTP setup (interactive)')
    parser.add_argument('--test', action='store_true',
                        help='send a sample email to verify SMTP + rendering')
    parser.add_argument('--list', action='store_true',
                        help='show pending alerts without sending')
    parser.add_argument('--watch', action='store_true',
                        help='start (or reattach to) the daemon, then tail its log')
    parser.add_argument('--watch-daemon', action='store_true',
                        help=argparse.SUPPRESS)  # internal — used by --watch
    parser.add_argument('--stop', action='store_true',
                        help='stop the running daemon')
    parser.add_argument('--status', action='store_true',
                        help='show daemon status')
    parser.add_argument('--since', metavar='YYYY-MM-DD',
                        help='re-send all alerts after this date')
    parser.add_argument('--detect-only', action='store_true',
                        help='run one detection pass; do not send')
    args = parser.parse_args()

    if not os.path.isdir(RUNS_ROOT):
        print(c('  runs/ directory not found at: {}'.format(RUNS_ROOT), 'red'))
        print(c('  Place notifier.py next to run_manager.py inside your project.', 'gray'))
        return 1

    if args.setup:
        cmd_setup()
        return 0
    if args.test:
        cmd_test()
        return 0
    if args.list:
        cmd_list()
        return 0
    if args.status:
        cmd_daemon_status()
        return 0
    if args.stop:
        cmd_stop()
        return 0
    if args.watch_daemon:
        cmd_watch_daemon()
        return 0
    if args.watch:
        cmd_watch()
        return 0
    if args.detect_only:
        cmd_detect_once()
        return 0

    since = None
    if args.since:
        try:
            since = datetime.fromisoformat(args.since)
        except ValueError:
            print(c('  --since must be YYYY-MM-DD format', 'red'))
            return 1

    # Default behavior: one detection pass + send pending, then exit.
    cmd_detect_once()
    sent, failed = cmd_send_pending(since=since)
    color = 'green' if failed == 0 else 'yellow'
    print(c('  Done. {} sent, {} failed.'.format(sent, failed), color))
    return 0


if __name__ == '__main__':
    sys.exit(main())
