# %%
import subprocess


class JoernSession:
    def __init__(self, worker_id: int=0):
        self.proc = subprocess.Popen(["joern"], stdin=subprocess.PIPE, encoding="utf-8")
        if worker_id != 0:
            self.switch_workspace(f"workers/{worker_id}")

    def close(self):
        try:
            outs, errs = self.proc.communicate("exit\n", timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            outs, errs = self.proc.communicate(timeout=15)

    """
    Joern commands
    """
    
    def send_line(self, cmd: str):
        self.proc.stdin.write(f"{cmd}\n")

    def switch_workspace(self, filepath: str):
        return self.send_line(f"""switchWorkspace("{filepath}")\n""")

    def import_code(self, filepath: str):
        return self.send_line(f"""importCode("{filepath}")\n""")
        
    def delete(self):
        return self.send_line(f"delete\n")

    def list_workspace(self):
        return self.send_line("workspace\n")



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
