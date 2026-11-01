# MOT17 보행자: 차량과 같은 계수기(track_count)로 세고 채점. 사용: bash bench/mot/run_mot.sh <방법이름> [track_count 옵션...]
cd "$(dirname "$0")/../.."; M=$1; shift
python3 - "$M" "$@" <<'PY'
import json,subprocess,sys
from pathlib import Path
M=sys.argv[1]; X=sys.argv[2:]
for c in json.load(open('bench/dataset_mot.json'))['clips']:
    cache=Path(f"bench/cache/{c['id']}_rfdetrS.npz")
    if not cache.exists(): continue
    subprocess.run(['python3','bench/methods/track_count.py','--cache',str(cache),'--lines',c['gates'],'--out',f"bench/preds/{M}/{c['id']}.json"]+X,check=True,capture_output=True)
PY
echo "== $M $*"; python3 bench/score.py --dataset bench/dataset_mot.json --method $M --split mot 2>/dev/null | grep -E "MOT17|^전체|view="
