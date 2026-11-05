"""Job storage and the background worker that runs the HDR engine."""
import datetime
import gc
import json
import os
import shutil
import threading
import time
import traceback
import uuid

import cv2

from engine import pipeline, raw_io

STATUS_LABELS = {
    'receiving': 'Receiving files',
    'queued': 'Waiting in line',
    'downloading': 'Downloading RAWs from Dropbox',
    'grouping': 'Sorting brackets',
    'processing': 'Editing photos',
    'saving': 'Saving to Dropbox',
    'done': 'Done',
    'error': 'Stopped with an error',
}


class JobStore:
    def __init__(self, data_dir):
        self.root = os.path.join(data_dir, 'jobs')
        os.makedirs(self.root, exist_ok=True)
        self.lock = threading.RLock()

    def dir(self, jid):
        return os.path.join(self.root, jid)

    def create(self, name, source, link='', per_set=0, save_to_dropbox=True):
        jid = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]
        d = self.dir(jid)
        for sub in ('raw', 'out', 'preview'):
            os.makedirs(os.path.join(d, sub), exist_ok=True)
        job = {'id': jid, 'name': name, 'source': source, 'link': link, 'per_set': per_set,
               'save_to_dropbox': bool(save_to_dropbox and source == 'dropbox'),
               'status': 'receiving' if source == 'upload' else 'queued', 'message': '',
               'created': time.time(), 'updated': time.time(), 'raw_count': 0,
               'sets_total': 0, 'sets_done': 0, 'photos': [], 'failed': [], 'dropbox_dest': '',
               'saved_count': 0, 'error': ''}
        self.save(job)
        return job

    def save(self, job):
        with self.lock:
            job['updated'] = time.time()
            p = os.path.join(self.dir(job['id']), 'job.json')
            tmp = p + '.tmp'
            with open(tmp, 'w') as fh:
                json.dump(job, fh)
            os.replace(tmp, p)

    def get(self, jid):
        if not jid or '/' in jid or '..' in jid:
            return None
        try:
            with open(os.path.join(self.dir(jid), 'job.json')) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return None

    def all(self):
        jobs = []
        for jid in sorted(os.listdir(self.root), reverse=True):
            j = self.get(jid)
            if j:
                jobs.append(j)
        return jobs

    def delete(self, jid):
        if self.get(jid):
            shutil.rmtree(self.dir(jid), ignore_errors=True)


class Worker(threading.Thread):
    """Processes one job at a time (the engine needs ~3-4 GB of memory per photo)."""

    def __init__(self, store, dropbox, retention_days=14):
        super().__init__(daemon=True)
        self.store = store
        self.dbx = dropbox
        self.retention_days = retention_days
        self.wake = threading.Event()
        self.current = None

    def run(self):
        # jobs interrupted by a restart go back in line
        for j in self.store.all():
            if j['status'] in ('downloading', 'grouping', 'processing', 'saving'):
                j['status'] = 'queued'
                j['message'] = 'Resumed after a restart'
                self.store.save(j)
        last_cleanup = 0
        while True:
            if time.time() - last_cleanup > 3600:
                self.cleanup()
                last_cleanup = time.time()
            queued = [j for j in self.store.all() if j['status'] == 'queued']
            if not queued:
                self.wake.wait(5)
                self.wake.clear()
                continue
            job = sorted(queued, key=lambda j: j['created'])[0]
            self.current = job['id']
            try:
                self.process(job)
            except Exception as e:  # keep the worker alive whatever happens
                traceback.print_exc()
                job = self.store.get(job['id']) or job
                job['status'] = 'error'
                job['error'] = str(e)[:500]
                self.store.save(job)
            finally:
                self.current = None
                gc.collect()

    def cleanup(self):
        cutoff = time.time() - self.retention_days * 86400
        for j in self.store.all():
            if j['created'] < cutoff and j['status'] in ('done', 'error'):
                self.store.delete(j['id'])

    # ------------------------------------------------------------------------
    def _set(self, job, status, message=''):
        job['status'] = status
        job['message'] = message
        self.store.save(job)

    def process(self, job):
        d = self.store.dir(job['id'])
        raw_dir = os.path.join(d, 'raw')

        # 1. find the RAWs and read their EXIF (only the first few KB of each file)
        remote = {}  # local path -> Dropbox path (files are fetched per photo, then deleted)
        infos = {}
        if job['source'] == 'dropbox':
            self._set(job, 'downloading', 'Listing files')
            files = [f for f in self.dbx.list_link(job['link']) if raw_io.is_raw(f['name'])]
            if not files:
                raise RuntimeError('No RAW files found in that Dropbox folder.')
            job['raw_count'] = len(files)
            head = os.path.join(d, 'head.tmp')
            for i, f in enumerate(files, 1):
                local = os.path.join(raw_dir, f['path'].strip('/').replace('/', '__'))
                remote[local] = (f['path'], f['size'])
                self.dbx.download_link_head(job['link'], f['path'], head)
                infos[local] = raw_io.read_info(head)
                if i % 10 == 0 or i == len(files):
                    self._set(job, 'downloading', f'Reading RAW info {i} of {len(files)}')
            if os.path.exists(head):
                os.remove(head)
            if any(v['exposure'] is None for v in infos.values()):
                # no usable EXIF (e.g. some CR3 files): download everything up front instead
                for i, (local, (rp, size)) in enumerate(remote.items(), 1):
                    if not (os.path.exists(local) and os.path.getsize(local) == size):
                        self.dbx.download_link_file(job['link'], rp, local)
                    self._set(job, 'downloading', f'Downloaded {i} of {len(remote)} RAWs')
                infos, remote = {}, {}
            raws = sorted(set(infos) | set(remote) | {os.path.join(raw_dir, x) for x in os.listdir(raw_dir)
                                                      if raw_io.is_raw(x)})
            if job['save_to_dropbox'] and not job['dropbox_dest']:
                try:
                    meta = self.dbx.link_metadata(job['link'])
                    if meta.get('path_lower'):
                        job['dropbox_dest'] = meta['path_lower'].rstrip('/') + '/edited'
                    else:
                        job['dropbox_dest'] = '/HSH Edited/' + meta.get('name', job['name'])
                except Exception:
                    job['dropbox_dest'] = '/HSH Edited/' + job['name']
                self.store.save(job)
        else:
            raws = sorted(os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if raw_io.is_raw(f))
        job['raw_count'] = len(raws)
        if not raws:
            raise RuntimeError('No RAW files to edit.')

        # 2. group brackets
        self._set(job, 'grouping', 'Sorting brackets into photos')
        sets = pipeline.group_brackets(raws, per_set=int(job.get('per_set') or 0), infos=infos or None)
        job['sets_total'] = len(sets)
        done_names = {p['name'] for p in job['photos']}
        self.store.save(job)

        # 3. edit each photo
        for i, s in enumerate(sets, 1):
            first = os.path.splitext(os.path.basename(s[0]['path']))[0].split('__')[-1]
            name = first + '_HDR'
            if name in done_names:
                continue
            self._set(job, 'processing', f'Editing photo {i} of {len(sets)}')
            t0 = time.time()
            try:
                for it in s:  # fetch this photo's brackets from Dropbox
                    if it['path'] in remote and not os.path.exists(it['path']):
                        self.dbx.download_link_file(job['link'], remote[it['path']][0], it['path'])
                out, before, after = pipeline.process_set(s)
                cv2.imwrite(os.path.join(d, 'out', name + '.jpg'), cv2.cvtColor(out, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(os.path.join(d, 'preview', name + '_before.jpg'),
                            cv2.cvtColor(before, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
                cv2.imwrite(os.path.join(d, 'preview', name + '_after.jpg'),
                            cv2.cvtColor(after, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
                del out, before, after
                job['photos'].append({'name': name, 'frames': [os.path.basename(x['path']).split('__')[-1] for x in s],
                                      'seconds': round(time.time() - t0)})
            except Exception as e:
                traceback.print_exc()
                job['failed'].append({'name': name, 'error': str(e)[:300]})
            finally:
                if job['source'] == 'dropbox' and remote:
                    for it in s:  # free the disk right away
                        if it['path'] in remote and os.path.exists(it['path']):
                            os.remove(it['path'])
            job['sets_done'] = i
            self.store.save(job)
            gc.collect()

        # 4. save to Dropbox
        if job['save_to_dropbox'] and job['dropbox_dest']:
            names = [p['name'] for p in job['photos']]
            for k, n in enumerate(names, 1):
                self._set(job, 'saving', f'Saving {k} of {len(names)} to Dropbox')
                self.dbx.upload(os.path.join(d, 'out', n + '.jpg'), f"{job['dropbox_dest']}/{n}.jpg")
                job['saved_count'] = k
            self.store.save(job)

        # 5. tidy up: RAWs are no longer needed
        shutil.rmtree(raw_dir, ignore_errors=True)
        os.makedirs(raw_dir, exist_ok=True)
        msg = f"{len(job['photos'])} photos ready"
        if job['failed']:
            msg += f", {len(job['failed'])} could not be edited"
        self._set(job, 'done', msg)
