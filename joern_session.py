# %%
import subprocess
from pathlib import Path
import time
import logging
import signal
import sys
import pexpect

import re

def shesc(sometext):
    """
    Delete ANSI escape sequences from string
    """
    # 7-bit C1 ANSI sequences
    ansi_escape = re.compile(r'''
        \x1B  # ESC
        (?:   # 7-bit C1 Fe (except CSI)
            [@-Z\\-_]
        |     # or [ for CSI, followed by a control sequence
            \[
            [0-?]*  # Parameter bytes
            [ -/]*  # Intermediate bytes
            [@-~]   # Final byte
        )
    ''', re.VERBOSE)
    return ansi_escape.sub('', sometext)


class JoernSession:
    def __init__(self, worker_id: int=0):
        self.proc = pexpect.spawn("joern --nocolors")
        self.read_until_prompt()

        if worker_id != 0:
            workspace = f"workers/{worker_id}"
            self.switch_workspace(workspace)
            
    def read_until_prompt(self, zonk_line=False):
        pattern = "joern>"
        if zonk_line:
            pattern += ".*\n"
        out = self.proc.expect(pattern)
        return shesc(self.proc.before.decode()).strip()

    def close(self):
        self.proc.sendline("exit")
        self.proc.sendline("N")
    
    def send_line(self, cmd: str):
        self.proc.sendline(cmd)
        self.read_until_prompt(zonk_line=True)  # Skip the echoed prompt "joern>"

    """
    Joern commands
    """
    
    def run_command(self, command):
        self.send_line(command)
        return self.read_until_prompt()

    def run_script(self, script: str, params, import_first=True):
        if import_first:
            scriptdir: Path = Path("storage/external")
            scriptdir_str = str(scriptdir)
            if scriptdir_str.endswith("/"):
                scriptdir_str = scriptdir_str[:-1]
            scriptdir_str = scriptdir_str.replace("/", ".")
            self.run_command(f"""import $file.{scriptdir_str}.{script}""")

        params_str = ", ".join(f'{k}="{v}"' for k, v in params.items())
        return self.run_command(f"""{script}.exec({params_str})""")

    def switch_workspace(self, filepath: str):
        return self.run_command(f"""switchWorkspace("{filepath}")""")

    def import_code(self, filepath: str):
        return self.run_command(f"""importCode("{filepath}")""")
        
    def delete(self):
        return self.run_command(f"delete")

    def list_workspace(self):
        return self.run_command("workspace")

    def cpg_path(self):
        project_path = self.run_command("print(project.path)")
        cpg_path = Path(project_path) / "cpg.bin"
        return cpg_path


def test_get_cpg():
    sess = JoernSession()
    try:
        sess.import_code("x42/c/X42.c")
        cpg_path = sess.cpg_path()
        assert cpg_path.exists(), cpg_path
        sess.delete()
    finally:
        sess.close()


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
