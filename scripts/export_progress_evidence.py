"""Publish small derived artifacts for the 2026-10-24 documentation snapshot."""
import hashlib
import json
from pathlib import Path
import shutil

root=Path(__file__).resolve().parents[1]
files={
    'output/formal_analysis/7f832b6b_audit_v1/cells.csv':'docs/measurements/formal_cells_2026-10-24.csv',
    'output/cross_family_costs/20261023T204321Z/historical_audit/historical_four_family.csv':'docs/measurements/historical_family_cost_audit_2026-10-24.csv',
    'output/cudagraph_chain/batch_remeasure_20261023_202936/analysis/seq_cells.csv':'docs/measurements/cudagraph_chain_corrected_2026-10-24.csv',
    'output/figures/experiment_workflow/experiment_workflow.png':'docs/figures/experiment_workflow.png',
    'output/figures/experiment_workflow/experiment_workflow.svg':'docs/figures/experiment_workflow.svg',
    'output/figures/specdec_cost_ledger_redesign/specdec_cost_story.png':'docs/figures/specdec_cost_story.png',
    'output/figures/specdec_cost_ledger_redesign/specdec_cost_story.svg':'docs/figures/specdec_cost_story.svg',
}
index={'date_hong_kong':'2026-10-24','purpose':'documentation snapshot, not new experiment',
       'formal_source_commit':'82dd4b2fc8de5c8ba87470921335e2d4642eed3f',
       'development_runs_dirty_worktree':True,
       'text_normalization':'Published text uses LF; generated SVG trailing whitespace removed. Source hashes refer to original local bytes.',
       'published':[],'local_evidence':[]}
for src,dst in files.items():
    source,target=root/src,root/dst
    if not source.is_file():raise FileNotFoundError(source)
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source,target)
    if target.suffix in ('.csv','.svg'):
        content=target.read_bytes().replace(b'\r\n',b'\n')
        if target.suffix=='.svg':
            content=b'\n'.join(line.rstrip() for line in content.split(b'\n'))
        target.write_bytes(content)
    index['published'].append({'source':src,'published':dst,'bytes':target.stat().st_size,
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
for name in [
    'output/formal_analysis/7f832b6b_audit_v1/report.md',
    'output/formal_analysis/7f832b6b_audit_v1/summary.json',
    'output/cost_profile_2k/20261023T075505Z_a64b2ec3/manifest.json',
    'output/draft_propose_trace/20261024_013123_prereg_v1/verification.json',
    'output/cudagraph_chain/batch_remeasure_20261023_202936/target.json',
    'output/cudagraph_chain/batch_remeasure_20261023_202936/draft.json',
    'output/cross_family_costs/20261023T204321Z/failure.json',
    'output/cross_family_costs/20261023T204641Z/failure.json']:
    p=root/name
    index['local_evidence'].append({'path':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
dest=root/'docs/measurements/progress_evidence_2026-10-24.json'
dest.write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
print('Published',len(files),'small artifacts; large traces and runtime remain local.')
