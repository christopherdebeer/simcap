"""Fetch the exact public source files and verify their SHA256 checksums."""
import argparse,concurrent.futures,hashlib,json,urllib.request
from pathlib import Path
from urllib.parse import quote

DATA_COMMIT='287f3942b2a4d7fefb4f8ddd9b82b86e6b7ad143'
ROOT=Path(__file__).resolve().parent
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--labeled-only',action='store_true');args=ap.parse_args()
    rows=json.loads((ROOT/'results/inventory.json').read_text())
    if args.labeled_only:rows=[r for r in rows if r.get('binary_labeled_blocks',0)]
    args.output.mkdir(parents=True,exist_ok=True)
    def fetch(row):
        p=args.output/row['file']
        if p.exists():content=p.read_bytes()
        else:
            url=f'https://raw.githubusercontent.com/christopherdebeer/simcap/{DATA_COMMIT}/GAMBIT/'+quote(row['file'])
            content=urllib.request.urlopen(url,timeout=60).read()
        if hashlib.sha256(content).hexdigest()!=row['sha256']:raise ValueError('Hash mismatch: '+row['file'])
        p.write_bytes(content)
        return row['file']
    with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:
        for name in pool.map(fetch,rows):print(name)
if __name__=='__main__':main()
