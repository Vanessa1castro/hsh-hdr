"""HSH HDR Studio - private web app for merging bracketed RAWs into edited photos."""
import hmac
import os
import re
import secrets
import zipfile

from flask import (Flask, abort, jsonify, redirect, render_template, request, send_file,
                   session, url_for)
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename

from dropbox_client import Dropbox, DropboxError
from engine import raw_io
from jobs import STATUS_LABELS, JobStore, Worker

DATA_DIR = os.environ.get('DATA_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'))
os.makedirs(DATA_DIR, exist_ok=True)
APP_PASSWORD = os.environ.get('APP_PASSWORD', '')


def _secret_key():
    if os.environ.get('SECRET_KEY'):
        return os.environ['SECRET_KEY']
    p = os.path.join(DATA_DIR, 'secret.key')
    if not os.path.exists(p):
        with open(p, 'w') as fh:
            fh.write(secrets.token_hex(32))
    return open(p).read().strip()


app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
app.secret_key = _secret_key()
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Lax',
                  PERMANENT_SESSION_LIFETIME=60 * 60 * 24 * 30, MAX_CONTENT_LENGTH=None)

store = JobStore(DATA_DIR)
dbx = Dropbox(DATA_DIR, os.environ.get('DROPBOX_APP_KEY'), os.environ.get('DROPBOX_APP_SECRET'))
worker = Worker(store, dbx, retention_days=int(os.environ.get('KEEP_DAYS', '14')))
worker.start()


# ------------------------------------------------------------------ auth --

@app.before_request
def _require_login():
    if request.endpoint in ('login', 'static', 'health'):
        return None
    if not APP_PASSWORD:
        return ('Set the APP_PASSWORD variable in Railway to finish setup.', 503)
    if not session.get('ok'):
        if request.path.startswith('/api/'):
            return jsonify(error='Please sign in again.'), 401
        return redirect(url_for('login', next=request.path))
    return None


@app.route('/login', methods=['GET', 'POST'])
def login():
    error = ''
    if request.method == 'POST':
        if APP_PASSWORD and hmac.compare_digest(request.form.get('password', ''), APP_PASSWORD):
            session.permanent = True
            session['ok'] = True
            nxt = request.args.get('next', '/')
            return redirect(nxt if nxt.startswith('/') and not nxt.startswith('//') else '/')
        error = 'That password is not right.'
    return render_template('login.html', error=error)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.route('/health')
def health():
    return 'ok'


# ---------------------------------------------------------------- pages --

@app.route('/')
def index():
    return render_template('index.html')


# -------------------------------------------------------------- dropbox --

def _redirect_uri():
    return url_for('dropbox_callback', _external=True)


@app.route('/dropbox/connect')
def dropbox_connect():
    if not dbx.configured:
        return ('Add DROPBOX_APP_KEY and DROPBOX_APP_SECRET in Railway first.', 400)
    state = secrets.token_urlsafe(16)
    session['dbx_state'] = state
    return redirect(dbx.auth_url(_redirect_uri(), state))


@app.route('/dropbox/callback')
def dropbox_callback():
    if request.args.get('state') != session.pop('dbx_state', None):
        return ('Dropbox sign-in expired. Go back and click "Connect Dropbox" again.', 400)
    if request.args.get('error'):
        return redirect('/?dropbox=cancelled')
    try:
        dbx.finish_auth(request.args.get('code', ''), _redirect_uri())
    except DropboxError as e:
        return (str(e), 400)
    return redirect('/?dropbox=connected')


@app.route('/api/dropbox/disconnect', methods=['POST'])
def dropbox_disconnect():
    dbx.disconnect()
    return jsonify(ok=True)


# ------------------------------------------------------------------ api --

def _public(job, full=False):
    j = {k: job[k] for k in ('id', 'name', 'source', 'status', 'message', 'created', 'raw_count',
                             'sets_total', 'sets_done', 'dropbox_dest', 'saved_count', 'error',
                             'save_to_dropbox')}
    j['status_label'] = STATUS_LABELS.get(job['status'], job['status'])
    j['photo_count'] = len(job['photos'])
    j['failed'] = job['failed']
    if full:
        j['photos'] = job['photos']
    else:
        j['cover'] = job['photos'][0]['name'] if job['photos'] else ''
    return j


@app.route('/api/status')
def api_status():
    return jsonify(dropbox_configured=dbx.configured, dropbox_connected=dbx.connected,
                   dropbox_account=dbx.account_name, jobs=[_public(j) for j in store.all()[:60]])


_LINK_RE = re.compile(r'^https://(www\.)?dropbox\.com/', re.I)


@app.route('/api/jobs', methods=['POST'])
def api_create():
    data = request.get_json(force=True, silent=True) or {}
    source = data.get('source', 'dropbox')
    per_set = int(data.get('per_set') or 0)
    name = (data.get('name') or '').strip()[:120]
    if source == 'dropbox':
        link = (data.get('link') or '').strip()
        if not _LINK_RE.match(link):
            return jsonify(error='Paste a Dropbox folder link (it starts with https://www.dropbox.com/).'), 400
        if not dbx.connected:
            return jsonify(error='Connect Dropbox first (button at the top).'), 400
        if not name:
            try:
                name = dbx.link_metadata(link).get('name', '')
            except DropboxError as e:
                return jsonify(error=f'Dropbox could not open that link: {e}'), 400
        job = store.create(name or 'Dropbox folder', 'dropbox', link=link, per_set=per_set,
                           save_to_dropbox=data.get('save_to_dropbox', True))
        worker.wake.set()
    else:
        job = store.create(name or 'Uploaded RAWs', 'upload', per_set=per_set, save_to_dropbox=False)
    return jsonify(_public(job))


@app.route('/api/jobs/<jid>/upload', methods=['POST'])
def api_upload(jid):
    job = store.get(jid)
    if not job or job['status'] != 'receiving':
        abort(404)
    saved = 0
    for f in request.files.getlist('files'):
        fname = secure_filename(f.filename or '')
        if not fname or not raw_io.is_raw(fname):
            continue
        f.save(os.path.join(store.dir(jid), 'raw', fname))
        saved += 1
    return jsonify(saved=saved)


@app.route('/api/jobs/<jid>/start', methods=['POST'])
def api_start(jid):
    job = store.get(jid)
    if not job or job['status'] != 'receiving':
        abort(404)
    job['status'] = 'queued'
    store.save(job)
    worker.wake.set()
    return jsonify(_public(job))


@app.route('/api/jobs/<jid>')
def api_job(jid):
    job = store.get(jid)
    if not job:
        abort(404)
    return jsonify(_public(job, full=True))


@app.route('/api/jobs/<jid>/retry', methods=['POST'])
def api_retry(jid):
    job = store.get(jid)
    if not job or job['status'] != 'error':
        abort(404)
    job['status'], job['error'], job['message'] = 'queued', '', 'Retrying'
    store.save(job)
    worker.wake.set()
    return jsonify(_public(job))


@app.route('/api/jobs/<jid>', methods=['DELETE'])
def api_delete(jid):
    job = store.get(jid)
    if not job:
        abort(404)
    if worker.current == jid:
        return jsonify(error='This job is running; wait for it to finish.'), 409
    store.delete(jid)
    return jsonify(ok=True)


_NAME_RE = re.compile(r'^[A-Za-z0-9._ -]+$')


@app.route('/jobs/<jid>/<kind>/<name>.jpg')
def job_image(jid, kind, name):
    job = store.get(jid)
    if not job or not _NAME_RE.match(name) or kind not in ('before', 'after', 'full'):
        abort(404)
    d = store.dir(jid)
    p = os.path.join(d, 'out', name + '.jpg') if kind == 'full' else os.path.join(d, 'preview', f'{name}_{kind}.jpg')
    if not os.path.exists(p):
        abort(404)
    return send_file(p, mimetype='image/jpeg', as_attachment=kind == 'full', download_name=name + '.jpg',
                     max_age=3600)


@app.route('/jobs/<jid>/download.zip')
def job_zip(jid):
    job = store.get(jid)
    if not job:
        abort(404)
    d = store.dir(jid)
    zp = os.path.join(d, 'photos.zip')
    names = [p['name'] for p in job['photos']]
    if not os.path.exists(zp) or os.path.getmtime(zp) < job['updated']:
        with zipfile.ZipFile(zp + '.tmp', 'w', zipfile.ZIP_STORED) as z:
            for n in names:
                z.write(os.path.join(d, 'out', n + '.jpg'), n + '.jpg')
        os.replace(zp + '.tmp', zp)
    safe = re.sub(r'[^A-Za-z0-9 ._-]+', '', job['name'])[:80] or 'photos'
    return send_file(zp, mimetype='application/zip', as_attachment=True, download_name=f'{safe} - edited.zip')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', '8080')), threaded=True)
