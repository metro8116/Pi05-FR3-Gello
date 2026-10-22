"""Supervise SSH, one ROS session and the preview GUI. Hardware starts only on explicit invocation."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

from config import ROOT, REPO_ROOT, load_config
from control_ownership import ControlLease
from prepare_local import docker_base, container_name, check_overlay, prepare
from protocol import VERSION, recv_packet, send_packet, validate_metadata

FAULT_LOCK = Path(os.environ.get('FR3_HARDWARE_FAULT_LOCK',str(REPO_ROOT/'runtime/gripper_hardware_fault.lock')))
RUNTIME_DIR = Path('/tmp') / f'pi05-fr3-{os.getuid()}'
BRIDGE_HEALTH_INTERVAL_S = .5
BRIDGE_HEALTH_TIMEOUT_S = .3
BRIDGE_HEALTH_MAX_SILENCE_S = 2.


def process_start(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except (OSError,IndexError):
        return None


def packet_call(host,port,packet,timeout=.5):
    with socket.create_connection((host,port),timeout=timeout) as conn:
        conn.settimeout(timeout)
        send_packet(conn,packet)
        return recv_packet(conn)


def assert_no_legacy_control():
    # Match executable names, never arbitrary strings in a shell command.
    names={'video_action_recorder.py','local_gello_data_bridge.py'}
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            args=p.read_bytes().split(b'\0')[:3]
            if any(Path(arg.decode(errors='replace')).name in names for arg in args if arg):
                raise RuntimeError('Close the GELLO collection GUI/control before opening deployment.')
        except (PermissionError,FileNotFoundError,ProcessLookupError):
            pass
    code=('from pathlib import Path; import json; names={"ros2_control_node","gello_publisher","franka_gripper_node","pi05_gripper_node"}; '
          'found=[]\n'
          'for p in Path("/proc").glob("[0-9]*/cmdline"):\n'
          ' try:\n'
          '  args=p.read_bytes().split(b"\\0")[:2]\n'
          '  if any(Path(a.decode(errors="replace")).name in names for a in args if a): found.append(p.parent.name)\n'
          ' except (OSError,ProcessLookupError): pass\n'
          'print(json.dumps(found))')
    result=subprocess.run(docker_base()+['exec',container_name(),'python3','-c',code],
                          check=True,capture_output=True,text=True,timeout=5)
    if json.loads(result.stdout):
        raise RuntimeError('Existing FR3/GELLO control processes are running. Stop the existing session first.')


def ssh_command(config):
    host=os.environ.get('PI05_SSH_HOST','')
    user=os.environ.get('PI05_SSH_USER','root')
    port=os.environ.get('PI05_SSH_PORT','')
    if (not host or host.startswith('YOUR_') or not re.fullmatch(r'[A-Za-z0-9_.:-]+',host)
            or not re.fullmatch(r'[A-Za-z0-9_.-]+',user)):
        raise ValueError('Fill PI05_SSH_HOST and PI05_SSH_USER in configs/autodl.env')
    if not port.isdigit() or not 1<=int(port)<=65535:
        raise ValueError('Fill PI05_SSH_PORT with the AutoDL SSH port')
    local=int(os.environ.get('PI05_LOCAL_PORT',config['model']['port']))
    remote=int(os.environ.get('PI05_REMOTE_PORT','8000'))
    if local!=config['model']['port'] or not 1024<=remote<=65535:
        raise ValueError('SSH local port must match model.port in deploy.yaml; choose a valid remote port')
    return ['ssh','-N','-T','-p',port,'-L',f'127.0.0.1:{local}:127.0.0.1:{remote}',
            '-o','ExitOnForwardFailure=yes','-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2',
            '-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',f'{user}@{host}']


def wait_metadata(config,process=None,timeout=15):
    deadline=time.monotonic()+timeout
    last=None
    while time.monotonic()<deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError('SSH exited; inspect ssh.log. First verify the host key and configure SSH key/agent login.')
        try:
            response=packet_call(config['model']['host'],config['model']['port'],{'version':VERSION,'type':'hello'},1)
            return validate_metadata(response)
        except (OSError,ValueError,EOFError) as exc:
            last=exc
            time.sleep(.15)
    raise RuntimeError(f'Inference server is not ready through the tunnel: {last}')


def stop_bridge(config):
    try:
        state=packet_call(config['bridge']['host'],config['bridge']['port'],{'version':VERSION,'type':'state'},.4)
        session=state.get('session_id')
        if session:
            return packet_call(config['bridge']['host'],config['bridge']['port'],
                               {'version':VERSION,'type':'stop','session_id':session},.4)
    except (OSError,ValueError,EOFError):
        pass
    return None


def refresh_child(process):
    owned=getattr(process,'fr3_members',{})
    if process_start(process.pid)==getattr(process,'fr3_start',None):
        for path in Path('/proc').glob('[0-9]*/stat'):
            try:
                fields=path.read_text().rsplit(')',1)[1].split()
                if fields[0]!='Z' and int(fields[2])==process.pid:
                    owned[int(path.parent.name)]=fields[19]
            except (OSError,IndexError,ValueError):pass
    process.fr3_members=owned
    return [{'pid':pid,'start':stamp} for pid,stamp in owned.items()
            if process_start(pid)==stamp]


def stop_host_members(members,term_timeout=3):
    for sig,seconds in ((signal.SIGTERM,term_timeout),(signal.SIGKILL,2)):
        live=[m for m in members if process_start(m['pid'])==m['start']]
        if not live:return
        for member in live:
            try:os.kill(member['pid'],sig)
            except ProcessLookupError:pass
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            if not any(process_start(m['pid'])==m['start'] for m in live):return
            time.sleep(.05)
    raise RuntimeError('Owned local deployment processes did not exit')


def stop_gui_members(process,timeout=12):
    """Let the GUI close its encoder pipes before signaling its descendants."""
    refresh_child(process)
    if process_start(process.pid)==process.fr3_start:
        try:os.kill(process.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        deadline=time.monotonic()+timeout
        while time.monotonic()<deadline and process_start(process.pid)==process.fr3_start:
            refresh_child(process)
            time.sleep(.05)
    # The exact recorded identities still fence cleanup if the GUI crashed or
    # ignored TERM. FFmpeg only receives TERM after the GUI's save allowance.
    stop_host_members(refresh_child(process),term_timeout=.5)


def stop_child(process,timeout=3):
    if getattr(process,'fr3_name',None)=='client':
        stop_gui_members(process,timeout)
    else:
        stop_host_members(refresh_child(process),term_timeout=timeout)
    try:process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:raise RuntimeError('Local deployment child did not exit')


def bundle_for_container(config,session):
    bundle=f'/tmp/pi05-deploy-{session}'
    subprocess.run(docker_base()+['exec',container_name(),'mkdir','-p',bundle+'/configs',bundle+'/ros2'],check=True)
    mapping={name:ROOT/name for name in ('protocol.py','config.py','bridge_core.py','fr3_policy_bridge.py','container_runner.py')}
    mapping['ros2/pi05_fr3.launch.py']=ROOT/'ros2/pi05_fr3.launch.py'
    mapping['configs/controllers_pi05.yaml']=ROOT/'configs/controllers_pi05.yaml'
    config_copy=ROOT/'runs'/session/'deploy.yaml'
    config_copy.parent.mkdir(parents=True,exist_ok=True)
    config_copy.write_text(json.dumps(config,indent=2)+'\n')
    mapping['configs/deploy.yaml']=config_copy
    for target,source in mapping.items():
        subprocess.run(docker_base()+['cp',str(source),container_name()+':'+bundle+'/'+target],
                       check=True,stdout=subprocess.DEVNULL)
    return bundle


def stop_container_bundle(bundle):
    subprocess.run(docker_base()+['exec',container_name(),'python3',bundle+'/container_runner.py',bundle,'--stop'],
                   check=True,timeout=30)


def inspect_host_python():
    python=os.environ.get('PI05_HOST_PYTHON',sys.executable)
    if not Path(python).is_file():
        raise FileNotFoundError(f'Host camera Python not found: {python}; set PI05_HOST_PYTHON')
    subprocess.run([python,'-c','import tkinter, numpy, cv2, PIL, pyrealsense2'],check=True,
                   capture_output=True,text=True,timeout=15)
    return python


def deployment_failure(logs, fallback):
    """Keep the actual ROS fatal/error ahead of subsequent transport failures."""
    try:
        with (logs/'ros.log').open('rb') as source:
            source.seek(0,os.SEEK_END)
            source.seek(max(0,source.tell()-131072))
            lines=source.read().decode('utf-8',errors='replace').splitlines()
    except OSError:
        lines=[]
    for level in ('FATAL','ERROR'):
        matches=[]
        for line in lines:
            if not re.search(r'\['+level+r'\]\s+\[\d',line):
                continue
            message=re.search(r'\[[^\]]+\]:\s*(.*)$',line)
            text=message.group(1) if message else line
            if level=='ERROR' and any(noise in text for noise in (
                    'Bridge process exiting','Connection refused','Connection closed','Broken pipe')):
                continue
            matches.append((line,text))
        if matches:
            line,text=matches[0] if level=='FATAL' else matches[-1]
            return {'reason':text,
                    'source':'ros.log','source_line':line}
    return {'reason':str(fallback),'source':'supervisor','source_line':str(fallback)}


def start(config,config_path,no_tunnel=False,network_only=False):
    children=[]
    handles=[]
    bundle=None
    session=uuid.uuid4().hex
    logs=ROOT/'logs'/session
    logs.mkdir(parents=True)
    session_status=logs/'session_status.json'
    gui=None
    failure=None
    stop_requested=False
    manifest=None
    def write_manifest(state='starting'):
        if network_only:return
        RUNTIME_DIR.mkdir(mode=0o700,parents=True,exist_ok=True)
        value={'pid':os.getpid(),'start':process_start(os.getpid()),'session':session,
               'bundle':bundle,'config':config,'logs':str(logs),'state':state,
               'children':[{'pid':p.pid,'start':p.fr3_start,'name':p.fr3_name,
                            'members':refresh_child(p)} for p in children]}
        temporary=RUNTIME_DIR/'session.tmp'
        temporary.write_text(json.dumps(value,indent=2))
        temporary.replace(RUNTIME_DIR/'session.json')
    def interrupt(signum,frame):
        nonlocal stop_requested
        stop_requested=True
    def publish_failure(phase):
        try:
            temporary=session_status.with_suffix('.tmp')
            temporary.write_text(json.dumps(dict(version=1,phase=phase,logs=str(logs),
                                                session=session,**failure),ensure_ascii=False))
            temporary.replace(session_status)
            return True
        except OSError as exc:
            print(f'Cannot publish deployment fault to GUI: {exc}. Logs: {logs}',file=sys.stderr,flush=True)
            return False
    previous={sig:signal.signal(sig,interrupt) for sig in (signal.SIGINT,signal.SIGTERM)}
    def child(argv,name,**kwargs):
        log=(logs/(name+'.log')).open('w')
        handles.append(log)
        process=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,**kwargs)
        process.fr3_start=process_start(process.pid)
        process.fr3_name=name
        process.fr3_members={process.pid:process.fr3_start}
        children.append(process)
        write_manifest()
        return process
    lease=None
    try:
        if not network_only:
            lease=ControlLease('pi05').acquire()
            if FAULT_LOCK.exists():
                raise RuntimeError(f'Hardware fault lock is present: {FAULT_LOCK}')
            host_python=inspect_host_python()
            check_overlay()
            assert_no_legacy_control()
            write_manifest()
        tunnel=None if no_tunnel else child(ssh_command(config),'ssh')
        metadata=wait_metadata(config,tunnel)
        print(json.dumps({'server':metadata,'local_logs':str(logs)},ensure_ascii=False,indent=2),flush=True)
        if network_only:
            return
        if metadata['mock']:
            raise RuntimeError('Mock server is for communication checks only; robot stack was not started.')
        if stop_requested:
            return
        bundle=bundle_for_container(config,session)
        write_manifest()
        command=('source /opt/ros/humble/setup.bash && '
                 'source /opt/ros/humble/franka/setup.bash && '
                 'source /workspace/ros2/install/setup.bash && '
                 'source /workspace/ros2/pi05_deploy_overlay/install/setup.bash && '
                 'exec /usr/bin/python3 '+shlex.quote(bundle+'/container_runner.py')+' '+shlex.quote(bundle))
        ros=child(docker_base()+['exec',container_name(),'bash','-lc',command],'ros')
        deadline=time.monotonic()+45
        while time.monotonic()<deadline and not stop_requested:
            if ros.poll() is not None:
                raise RuntimeError(f'ROS launch exited; inspect {logs}/ros.log')
            try:
                state=packet_call(config['bridge']['host'],config['bridge']['port'],{'version':VERSION,'type':'state'},.4)
                if state.get('ready') and not state.get('armed'):
                    break
            except (OSError,ValueError,EOFError):
                pass
            time.sleep(.1)
        else:
            if stop_requested:return
            raise RuntimeError(f'ROS state/bridge did not become ready; inspect {logs}/ros.log and Franka Desk')
        env=os.environ.copy()
        env['FR3_CONTROL_FD']=str(lease.handle.fileno())
        gui=child([host_python,str(ROOT/'deploy_fr3.py'),'--config',str(config_path),
                   '--session-status',str(session_status)],'client',
                  env=env,pass_fds=(lease.handle.fileno(),))
        write_manifest('preview')
        print('Preview ready. Explicitly enable model execution in the local GUI. Closing it stops this deployment session.',flush=True)
        last_bridge_reply=time.monotonic()
        next_bridge_check=last_bridge_reply+BRIDGE_HEALTH_INTERVAL_S
        bridge_error=None
        while not stop_requested:
            if gui.poll() is not None:
                break
            if ros.poll() is not None:
                raise RuntimeError(f'ROS launch exited unexpectedly; inspect {logs}/ros.log')
            if tunnel and tunnel.poll() is not None:
                raise RuntimeError(f'SSH tunnel exited unexpectedly; inspect {logs}/ssh.log')
            # A live launch process does not prove that its bridge is responsive.
            # This host-side check also covers a blocked ROS callback or shutdown.
            now=time.monotonic()
            if now>=next_bridge_check:
                next_bridge_check=now+BRIDGE_HEALTH_INTERVAL_S
                try:
                    state=packet_call(config['bridge']['host'],config['bridge']['port'],
                                      {'version':VERSION,'type':'state'},BRIDGE_HEALTH_TIMEOUT_S)
                    if state.get('type')!='state' or state.get('version')!=VERSION:
                        raise ValueError('Bridge did not return a version 1 state response')
                    last_bridge_reply=time.monotonic()
                    bridge_error=None
                    if state.get('exit_error'):
                        raise RuntimeError(str(state['exit_error']))
                except (OSError,ValueError,EOFError) as exc:
                    bridge_error=exc
            silence=time.monotonic()-last_bridge_reply
            if silence>=BRIDGE_HEALTH_MAX_SILENCE_S:
                raise RuntimeError(f'Bridge state RPC unresponsive for {silence:.1f}s; '
                                   f'stopping locally. Last error: {bridge_error}. Logs: {logs}')
            write_manifest('preview')
            time.sleep(.1)
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
        if gui is not None and gui.poll() is None and not stop_requested:
            failure=deployment_failure(logs,exc)
            if not publish_failure('stopping'):
                # Without the terminal-state notification, the old GUI cannot
                # safely be retained as a diagnostic window.
                failure=None
        raise
    finally:
        if bundle is not None:
            stop_bridge(config)
            # Keep the kernel control lease and recovery journal until the
            # container confirms all this session's ROS descendants exited.
            while True:
                try:
                    write_manifest('stopping')
                    stop_container_bundle(bundle)
                    break
                except (OSError,RuntimeError,subprocess.SubprocessError) as exc:
                    write_manifest('stop_unconfirmed')
                    print(f'Cannot confirm ROS stop; retaining control lock and retrying: {exc}. Logs: {logs}',file=sys.stderr,flush=True)
                    time.sleep(2)
        retained_gui=False
        for process in reversed(children):
            if process is gui and failure is not None and not stop_requested:
                retained_gui=True
            else:
                if process is gui:
                    stop_child(process,timeout=12)
                else:
                    stop_child(process)
        if retained_gui:
            publish_failure('stopped')
            # Keep the lease and recovery journal while the inert diagnostic
            # window is open. A new deployment requires closing this window.
            while not stop_requested and gui.poll() is None:
                write_manifest('faulted')
                time.sleep(.1)
            stop_child(gui,timeout=12)
        for handle in handles:handle.close()
        if lease:lease.release()
        path=RUNTIME_DIR/'session.json'
        if path.is_file():
            try:
                if json.loads(path.read_text()).get('session')==session:path.unlink()
            except (OSError,ValueError):pass
        for sig,handler in previous.items():signal.signal(sig,handler)


def stop():
    manifest=RUNTIME_DIR/'session.json'
    if not manifest.is_file():
        print('No supervised PI05 deployment session is recorded.')
        return
    record=json.loads(manifest.read_text())
    pid=int(record['pid'])
    if process_start(pid)==record.get('start'):
        command=Path(f'/proc/{pid}/cmdline').read_bytes()
        if b'local_session.py' not in command:
            raise RuntimeError('Recorded PID belongs to another process; refusing to signal it')
        stop_bridge(record['config'])
        os.kill(pid,signal.SIGTERM)
        deadline=time.monotonic()+45
        while time.monotonic()<deadline and process_start(pid)==record['start']:
            time.sleep(.1)
        if process_start(pid)==record['start']:
            raise RuntimeError('Supervisor did not exit. Inspect its logs and local robot state.')
    else:
        # The supervisor may have crashed; only use this exact recorded ROS bundle.
        stop_bridge(record['config'])
        if record.get('bundle'):stop_container_bundle(record['bundle'])
        for child in record.get('children',[]):
            if child.get('name')=='client':
                process=SimpleNamespace(pid=child['pid'],fr3_start=child['start'],
                                        fr3_members={item['pid']:item['start'] for item in child.get('members',[child])})
                stop_gui_members(process,12)
            else:
                stop_host_members(child.get('members',[child]))
    if manifest.exists():manifest.unlink()
    print('PI05 deployment stop requested and supervisor exited. Verify the robot state in Desk.')


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',default=os.environ.get('PI05_CONFIG',str(ROOT/'configs/deploy.yaml')))
    modes=ap.add_mutually_exclusive_group()
    modes.add_argument('--check',action='store_true',help='Read-only local dependency/config/overlay checks')
    modes.add_argument('--prepare',action='store_true',help='Build/test isolated controller overlay; no robot nodes')
    modes.add_argument('--network-check',action='store_true',help='Only verify SSH/server metadata; no cameras or robot')
    modes.add_argument('--stop',action='store_true')
    ap.add_argument('--no-tunnel',action='store_true',help='Use an already established local tunnel')
    args=ap.parse_args(argv)
    try:
        if args.stop:stop();return 0
        config=load_config(args.config)
        if args.prepare:
            with ControlLease('pi05'):
                assert_no_legacy_control()
                print(json.dumps(prepare(build=True),indent=2))
        elif args.check:
            print(json.dumps({'config':str(Path(args.config).resolve()),'host_python':inspect_host_python(),
                              'controller_overlay':check_overlay(),'hardware_started':False},indent=2))
        else:start(config,Path(args.config).resolve(),args.no_tunnel,args.network_check)
        return 0
    except (OSError,ValueError,RuntimeError,subprocess.SubprocessError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        return 1


if __name__=='__main__':raise SystemExit(main())
