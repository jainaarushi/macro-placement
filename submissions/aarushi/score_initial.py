import sys, torch, numpy as np
sys.path.insert(0, '/home/ubuntu/macro-place-challenge-2026')
from macro_place.benchmark import Benchmark
from macro_place.objective import compute_proxy_cost
from macro_place.loader import load_benchmark_from_dir

BENCHES = ['ibm01','ibm02','ibm03','ibm04','ibm06','ibm07','ibm08',
           'ibm09','ibm10','ibm11','ibm12','ibm13','ibm14','ibm15',
           'ibm16','ibm17','ibm18']

def legalize(pos_t, bm, max_passes=300):
    pos = pos_t.numpy().copy()
    nh = bm.num_hard_macros
    sz = bm.macro_sizes.numpy()
    fixed = bm.macro_fixed.numpy()
    cw, ch = bm.canvas_width, bm.canvas_height
    for _ in range(max_passes):
        moved = False
        for i in range(nh):
            if fixed[i]: continue
            for j in range(i+1, nh):
                dx,dy = abs(pos[i,0]-pos[j,0]), abs(pos[i,1]-pos[j,1])
                sx=(sz[i,0]+sz[j,0])/2+1e-3; sy=(sz[i,1]+sz[j,1])/2+1e-3
                if dx<sx and dy<sy:
                    ox,oy=sx-dx,sy-dy
                    if ox<=oy:
                        d=ox/2+1e-3; s=1 if pos[i,0]>pos[j,0] else -1
                        if not fixed[i]: pos[i,0]+=s*d
                        if not fixed[j]: pos[j,0]-=s*d
                    else:
                        d=oy/2+1e-3; s=1 if pos[i,1]>pos[j,1] else -1
                        if not fixed[i]: pos[i,1]+=s*d
                        if not fixed[j]: pos[j,1]-=s*d
                    moved=True
        for i in range(nh):
            if fixed[i]: continue
            pos[i,0]=np.clip(pos[i,0],sz[i,0]/2,cw-sz[i,0]/2)
            pos[i,1]=np.clip(pos[i,1],sz[i,1]/2,ch-sz[i,1]/2)
        if not moved: break
    return torch.tensor(pos, dtype=torch.float32)

REPLACE={'ibm01':0.9976,'ibm02':1.837,'ibm03':1.3222,'ibm04':1.3024,
         'ibm06':1.6187,'ibm07':1.4633,'ibm08':1.4285,'ibm09':1.1194,
         'ibm10':1.5009,'ibm11':1.1774,'ibm12':1.7261,'ibm13':1.3355,
         'ibm14':1.5436,'ibm15':1.5159,'ibm16':1.478,'ibm17':1.6446,'ibm18':1.7722}

results = {}
print('{:>8}  {:>4}  {:>9}  {:>7}  {:>7}  {:>4}h {:>4}s'.format(
    'bench','ov','leg_proxy','RePlAce','vs_rep','hard','soft'))
print('-'*60)
for b in BENCHES:
    bm = Benchmark.load('benchmarks/processed/public/'+b+'.pt')
    _,plc = load_benchmark_from_dir('external/MacroPlacement/Testcases/ICCAD04/'+b)
    init_c = compute_proxy_cost(bm.macro_positions, bm, plc)
    leg = legalize(bm.macro_positions.clone(), bm)
    leg_c = compute_proxy_cost(leg, bm, plc)
    rep = REPLACE.get(b,0)
    pct = (rep-leg_c['proxy_cost'])/rep*100 if rep else 0
    print('{:>8}  {:>4}  {:>9.4f}  {:>7.4f}  {:>+6.1f}%  {:>4} {:>4}'.format(
        b, init_c['overlap_count'], leg_c['proxy_cost'], rep, pct,
        bm.num_hard_macros, bm.num_soft_macros))
    results[b] = leg_c['proxy_cost']
    import sys; sys.stdout.flush()

avg=sum(results.values())/len(results)
avg_rep=sum(REPLACE[b] for b in BENCHES)/len(BENCHES)
print('-'*60)
print('{:>8}  {:>4}  {:>9.4f}  {:>7.4f}  {:>+6.1f}%'.format(
    'AVG','', avg, avg_rep, (avg_rep-avg)/avg_rep*100))
print('Target (reference DREAMPlace): 1.1464')
