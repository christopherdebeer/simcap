"""One-command offline rerun: python ml/calibration_lite/run.py --data-dir PATH."""
import argparse
from pathlib import Path
import subprocess
import sys

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-dir',type=Path,required=True)
    args=ap.parse_args();root=Path(__file__).resolve().parent
    for script in ['experiment.py','export_validate.py','diagnostics.py']:
        command=[sys.executable,str(root/script),'--data-dir',str(args.data_dir.resolve())]
        if script=='experiment.py':command+=['--output',str(root/'results')]
        subprocess.run(command,check=True)
if __name__=='__main__':main()
