"""Supervise only this session's ROS descendants, including orphaned children."""
from __future__ import annotations
import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def process_info(pid):
    try:
        values=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        if values[0]=='Z': return None
        return {'pid':int(pid),'start':values[19],'ppid':int(values[1]),'pgrp':int(values[2])}
    except (OSError,IndexError,ValueError):return None


def process_start(pid):
    info=process_info(pid)
    return info['start'] if info else None


def live_members(record, supervisor=None):
    found={}
    for item in record.get('members',[]):
        live=process_info(item['pid'])
        if live and live['start']==item['start']:found[live['pid']]=live
    if supervisor is not None and process_start(supervisor)==record.get('start'):
        all_info={}
        for path in Path('/proc').glob('[0-9]*'):
            info=process_info(path.name)
            if info:all_info[info['pid']]=info
        parents={supervisor}
        while True:
            children={pid for pid,info in all_info.items() if info['ppid'] in parents and pid not in parents}
            if not children:break
            parents|=children
        found.update({pid:all_info[pid] for pid in parents if pid!=supervisor and pid in all_info})
    return list(found.values())


def write_record(directory,record):
    temp=directory/f'ros_process.{os.getpid()}.tmp'
    temp.write_text(json.dumps(record))
    temp.replace(directory/'ros_process.json')


def terminate_members(directory,record,supervisor=None):
    """Verify PID start times before every signal; never rely on a dead leader."""
    for sig,seconds in ((signal.SIGINT,5),(signal.SIGTERM,3),(signal.SIGKILL,2)):
        targets=live_members(record,supervisor)
        if not targets:return
        record['members']=targets
        write_record(directory,record)
        for item in targets:
            if process_start(item['pid'])==item['start']:
                try:os.kill(item['pid'],sig)
                except ProcessLookupError:pass
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            record['members']=live_members(record,supervisor)
            if not record['members']:return
            write_record(directory,record)
            time.sleep(.05)
    raise RuntimeError('Some ROS descendants remain alive; retain the control lock and inspect local robot state')


def stop(directory):
    directory=Path(directory)
    # Serialize cancellation against launch. A stop before startup prevents
    # all future node creation in this bundle, including a delayed docker exec.
    with (directory/'launch.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        (directory/'stop.requested').touch()
        manifest=directory/'ros_process.json'
        if not manifest.is_file():return
        record=json.loads(manifest.read_text())
    pid=int(record['pid'])
    alive=process_start(pid)==record['start']
    if alive:
        argv=Path(f'/proc/{pid}/cmdline').read_bytes()
        if str(directory).encode() not in argv or b'container_runner.py' not in argv:
            raise RuntimeError('Refusing to signal an unrelated reused supervisor')
        os.kill(pid,signal.SIGINT)
        deadline=time.monotonic()+12
        while time.monotonic()<deadline:
            if manifest.exists():record=json.loads(manifest.read_text())
            if not live_members(record,pid if process_start(pid)==record['start'] else None):
                return
            time.sleep(.05)
    # Handles a crashed/unresponsive supervisor using only recorded descendants.
    terminate_members(directory,record,pid if alive else None)
    record['state']='stopped'
    record['members']=[]
    write_record(directory,record)


def run(directory):
    if os.getpgrp()!=os.getpid():os.setsid()
    # Keep ROS grandchildren attached to this supervisor if their launch parent dies.
    if ctypes.CDLL(None,use_errno=True).prctl(36,1,0,0,0)!=0:
        raise OSError(ctypes.get_errno(),'Cannot enable Linux child subreaper')
    requested=False
    def request_stop(signum,frame):
        nonlocal requested
        requested=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,request_stop)
    record={'pid':os.getpid(),'start':process_start(os.getpid()),'members':[],'state':'starting'}
    launch=None
    with (directory/'launch.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (directory/'stop.requested').exists():return 0
        launch=subprocess.Popen(['ros2','launch',str(directory/'ros2/pi05_fr3.launch.py'),
                                 'config_file:='+str(directory/'configs/deploy.yaml')])
        record['members']=live_members(record,os.getpid())
        record['state']='running'
        write_record(directory,record)
    try:
        while not requested and not (directory/'stop.requested').exists() and launch.poll() is None:
            record['members']=live_members(record,os.getpid())
            write_record(directory,record)
            time.sleep(.1)
    finally:
        record['state']='stopping'
        write_record(directory,record)
        terminate_members(directory,record,os.getpid())
        record['members']=[]
        record['state']='stopped'
        write_record(directory,record)
    return 0


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('directory')
    ap.add_argument('--stop',action='store_true')
    args=ap.parse_args(argv)
    directory=Path(args.directory).resolve()
    if not str(directory).startswith('/tmp/pi05-deploy-'):
        raise ValueError('Unexpected deployment bundle path')
    if args.stop:stop(directory);return 0
    return run(directory)


if __name__=='__main__':
    try:raise SystemExit(main())
    except (OSError,ValueError,RuntimeError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        raise SystemExit(1)
