"""Small Dropbox API client: OAuth (offline refresh token), shared-link listing and
download, and uploads into the folder the shared link points to."""
import json
import os
import threading
import time
import urllib.parse

import requests

API = 'https://api.dropboxapi.com/2'
CONTENT = 'https://content.dropboxapi.com/2'
_lock = threading.Lock()


class DropboxError(Exception):
    pass


class Dropbox:
    def __init__(self, data_dir, app_key, app_secret):
        self.path = os.path.join(data_dir, 'dropbox.json')
        self.app_key = app_key or ''
        self.app_secret = app_secret or ''
        self._access = None
        self._expires = 0
        self._root_ns = None

    # ---------------------------------------------------------------- oauth --
    @property
    def configured(self):
        return bool(self.app_key and self.app_secret)

    def _load(self):
        try:
            with open(self.path) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}

    def _save(self, data):
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(data, fh)
        os.replace(tmp, self.path)

    @property
    def connected(self):
        return bool(self._load().get('refresh_token'))

    @property
    def account_name(self):
        return self._load().get('account_name', '')

    def auth_url(self, redirect_uri, state):
        q = {'client_id': self.app_key, 'response_type': 'code', 'token_access_type': 'offline',
             'redirect_uri': redirect_uri, 'state': state}
        return 'https://www.dropbox.com/oauth2/authorize?' + urllib.parse.urlencode(q)

    def finish_auth(self, code, redirect_uri):
        r = requests.post('https://api.dropboxapi.com/oauth2/token', data={
            'code': code, 'grant_type': 'authorization_code', 'redirect_uri': redirect_uri},
            auth=(self.app_key, self.app_secret), timeout=30)
        if r.status_code != 200:
            raise DropboxError(f'Dropbox sign-in failed: {r.text[:300]}')
        tok = r.json()
        data = {'refresh_token': tok['refresh_token']}
        self._access = tok['access_token']
        self._expires = time.time() + int(tok.get('expires_in', 3600)) - 120
        self._save(data)
        try:
            acct = self.rpc('users/get_current_account', None)
            data['account_name'] = acct.get('name', {}).get('display_name', '') + ' (' + acct.get('email', '') + ')'
            self._save(data)
        except DropboxError:
            pass

    def disconnect(self):
        try:
            os.remove(self.path)
        except OSError:
            pass
        self._access = None
        self._root_ns = None

    def _token(self):
        with _lock:
            if self._access and time.time() < self._expires:
                return self._access
            rt = self._load().get('refresh_token')
            if not rt:
                raise DropboxError('Dropbox is not connected yet. Click "Connect Dropbox" first.')
            r = requests.post('https://api.dropboxapi.com/oauth2/token', data={
                'grant_type': 'refresh_token', 'refresh_token': rt},
                auth=(self.app_key, self.app_secret), timeout=30)
            if r.status_code != 200:
                raise DropboxError(f'Could not refresh Dropbox access: {r.text[:300]}')
            tok = r.json()
            self._access = tok['access_token']
            self._expires = time.time() + int(tok.get('expires_in', 3600)) - 120
            return self._access

    # ------------------------------------------------------------- requests --
    def _headers(self, extra=None, root=False):
        h = {'Authorization': 'Bearer ' + self._token()}
        if root:
            ns = self.root_namespace()
            if ns:
                h['Dropbox-API-Path-Root'] = json.dumps({'.tag': 'root', 'root': ns})
        if extra:
            h.update(extra)
        return h

    def rpc(self, endpoint, args, root=False, retries=4):
        for attempt in range(retries):
            r = requests.post(f'{API}/{endpoint}', headers=self._headers(
                {'Content-Type': 'application/json'} if args is not None else None, root=root),
                data=json.dumps(args) if args is not None else None, timeout=60)
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            if r.status_code != 200:
                raise DropboxError(f'{endpoint}: {r.status_code} {r.text[:300]}')
            return r.json()
        raise DropboxError(f'{endpoint}: Dropbox is busy, please try again later')

    def root_namespace(self):
        """Team accounts keep shared team folders under the team root namespace."""
        if self._root_ns is None:
            try:
                r = requests.post(f'{API}/users/get_current_account',
                                  headers={'Authorization': 'Bearer ' + self._token()}, timeout=30)
                self._root_ns = r.json().get('root_info', {}).get('root_namespace_id', '') if r.ok else ''
            except requests.RequestException:
                self._root_ns = ''
        return self._root_ns

    # --------------------------------------------------------- shared links --
    def link_metadata(self, url):
        return self.rpc('sharing/get_shared_link_metadata', {'url': url}, root=True)

    def list_link(self, url, sub='', depth=0):
        """All files under a shared folder link (recurses into subfolders, 3 levels)."""
        out = []
        res = self.rpc('files/list_folder', {'path': sub, 'shared_link': {'url': url}})
        entries = res.get('entries', [])
        while res.get('has_more'):
            res = self.rpc('files/list_folder/continue', {'cursor': res['cursor']})
            entries += res.get('entries', [])
        for e in entries:
            rel = (sub + '/' + e['name']) if sub else '/' + e['name']
            if e['.tag'] == 'file':
                out.append({'path': rel, 'name': e['name'], 'size': e.get('size', 0)})
            elif e['.tag'] == 'folder' and depth < 3 and e['name'].lower() != 'edited':
                out += self.list_link(url, rel, depth + 1)
        return out

    def download_link_file(self, url, rel_path, dest, retries=4):
        tmp = dest + '.part'
        for attempt in range(retries):
            try:
                with requests.post(f'{CONTENT}/sharing/get_shared_link_file', headers=self._headers(
                        {'Dropbox-API-Arg': json.dumps({'url': url, 'path': rel_path})}), stream=True,
                        timeout=120) as r:
                    if r.status_code == 429 or r.status_code >= 500:
                        time.sleep(2 ** attempt)
                        continue
                    if r.status_code != 200:
                        raise DropboxError(f'download {rel_path}: {r.status_code} {r.text[:200]}')
                    with open(tmp, 'wb') as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                os.replace(tmp, dest)
                return
            except requests.RequestException:
                time.sleep(2 ** attempt)
        raise DropboxError(f'download {rel_path}: failed after {retries} tries')

    def download_link_head(self, url, rel_path, dest, nbytes=4 * 1024 * 1024):
        """Fetch only the first bytes of a file (enough for the EXIF data)."""
        with requests.post(f'{CONTENT}/sharing/get_shared_link_file', headers=self._headers(
                {'Dropbox-API-Arg': json.dumps({'url': url, 'path': rel_path})}), stream=True, timeout=120) as r:
            if r.status_code != 200:
                raise DropboxError(f'read {rel_path}: {r.status_code} {r.text[:200]}')
            got = 0
            with open(dest, 'wb') as fh:
                for chunk in r.iter_content(256 * 1024):
                    fh.write(chunk)
                    got += len(chunk)
                    if got >= nbytes:
                        break

    def upload(self, local_path, dropbox_path, retries=4):
        size = os.path.getsize(local_path)
        for attempt in range(retries):
            try:
                if size <= 140 * 1024 * 1024:
                    with open(local_path, 'rb') as fh:
                        r = requests.post(f'{CONTENT}/files/upload', headers=self._headers({
                            'Content-Type': 'application/octet-stream',
                            'Dropbox-API-Arg': json.dumps({'path': dropbox_path, 'mode': 'overwrite',
                                                           'mute': True})}, root=True),
                            data=fh, timeout=300)
                    if r.status_code == 429 or r.status_code >= 500:
                        time.sleep(2 ** attempt)
                        continue
                    if r.status_code != 200:
                        raise DropboxError(f'upload {os.path.basename(local_path)}: {r.status_code} {r.text[:300]}')
                    return r.json()
                return self._upload_session(local_path, dropbox_path)
            except requests.RequestException:
                time.sleep(2 ** attempt)
        raise DropboxError(f'upload {os.path.basename(local_path)}: failed after {retries} tries')

    def _upload_session(self, local_path, dropbox_path, chunk=64 * 1024 * 1024):
        with open(local_path, 'rb') as fh:
            r = requests.post(f'{CONTENT}/files/upload_session/start', headers=self._headers(
                {'Content-Type': 'application/octet-stream', 'Dropbox-API-Arg': '{}'}, root=True),
                data=fh.read(chunk), timeout=300)
            r.raise_for_status()
            sid = r.json()['session_id']
            offset = fh.tell()
            while True:
                data = fh.read(chunk)
                if not data:
                    break
                r = requests.post(f'{CONTENT}/files/upload_session/append_v2', headers=self._headers({
                    'Content-Type': 'application/octet-stream',
                    'Dropbox-API-Arg': json.dumps({'cursor': {'session_id': sid, 'offset': offset}})}, root=True),
                    data=data, timeout=300)
                r.raise_for_status()
                offset += len(data)
            r = requests.post(f'{CONTENT}/files/upload_session/finish', headers=self._headers({
                'Content-Type': 'application/octet-stream',
                'Dropbox-API-Arg': json.dumps({'cursor': {'session_id': sid, 'offset': offset},
                                               'commit': {'path': dropbox_path, 'mode': 'overwrite'}})},
                root=True), data=b'', timeout=300)
            r.raise_for_status()
            return r.json()
