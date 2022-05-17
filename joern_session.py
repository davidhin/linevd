# %%
import subprocess
from pathlib import Path
import time
import logging
import signal
import sys
import pexpect


class JoernSession:
    def __init__(self, worker_id: int=0):
        self.proc = pexpect.spawn("joern --nocolors")
        print("started process")
        self.read_until_prompt()

        if worker_id != 0:
            workspace = f"workers/{worker_id}"
            self.switch_workspace(workspace)
            
    def read_until_prompt(self):
        self.proc.expect(".*joern>")

    def close(self):
        self.proc.sendline("exit")
        self.proc.sendline("N")
    
    def send_line(self, cmd: str):
        print(f"send_line {cmd}")
        self.proc.sendline(cmd)

    """
    Joern commands
    """

    def run_script(self, script: str, params, import_first=True):
        if import_first:
            scriptdir: Path = Path("storage/external")
            scriptdir_str = str(scriptdir)
            if scriptdir_str.endswith("/"):
                scriptdir_str = scriptdir_str[:-1]
            scriptdir_str = scriptdir_str.replace("/", ".")
            self.send_line(f"""import $file.{scriptdir_str}.{script}""")
            self.read_until_prompt()

        params_str = ", ".join(f'{k}="{v}"' for k, v in params.items())
        self.send_line(f"""{script}.exec({params_str})""")
        self.read_until_prompt()

    def switch_workspace(self, filepath: str):
        self.send_line(f"""switchWorkspace("{filepath}")""")
        self.read_until_prompt()

    def import_code(self, filepath: str):
        self.send_line(f"""importCode("{filepath}")""")
        self.read_until_prompt()
        
    def delete(self):
        self.send_line(f"delete")
        self.read_until_prompt()

    def list_workspace(self):
        self.send_line("workspace")
        self.read_until_prompt()


def test_interaction():
    sess = JoernSession()
    try:
        sess.list_workspace()
        sess.import_code("x42/c/X42.c")
        sess.list_workspace()
        sess.delete()
        sess.list_workspace()
    finally:
        sess.close()

def test_worker():
    sess1 = JoernSession(worker_id=1)
    sess2 = JoernSession(worker_id=2)
    try:
        sess1.import_code("x42/c/X42.c")
        sess1.list_workspace()
        sess2.list_workspace()
        sess2.import_code("x42/c/X42.c")
        sess2.list_workspace()
    finally:
        sess1.delete()
        sess2.delete()
        sess1.close()
        sess2.close()


def test_script():
    sess = JoernSession()
    try:
        sess.import_code("x42/c/X42.c")
        sess.run_script("get_dataflow_output", params={"filename": "x42/c/X42.c", "problem": "reachingdef"})
        sess.delete()
    finally:
        sess.close()
