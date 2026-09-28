"""Package immutable six-branch results, with M0, verified arrays and an offline summary."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import time
import zipfile

BRANCHES = ['standard','fixed_2m','fixed_3m_fn_off','fixed_3m_fn_on','mixed_2m','mixed_3m_fn_off']
POINTS = ['p001','p005','p020','p050','p100']
STAGES = ['b1__flickr30k','b1__coco','a__lcs','a__coco','b__MMEB12']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def identity_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def flatten(value, prefix=''):
    result={}
    if isinstance(value,dict):
        for key,item in value.items():result.update(flatten(item,prefix+'/'+key if prefix else key))
    elif isinstance(value,(int,float)) and not isinstance(value,bool):result[prefix]=value
    elif value is None:result[prefix]=None
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();root=args.root.resolve();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    state=json.loads((root/'outputs/evaluation/formal_full_v1/queue/status.json').read_text())
    inventory={}; summaries=[]; vectors={}; checks=[]
    def add(path,arc,light=False,expected=None):
        path=Path(path).resolve()
        if not path.is_relative_to(root):raise ValueError('Source outside repository')
        digest=sha(path)
        if expected and digest!=expected:raise ValueError('SHA mismatch: '+str(path))
        item={'source':str(path),'archive_path':arc,'bytes':path.stat().st_size,'sha256':digest,'light':light}
        if arc in inventory and inventory[arc]!=item:raise ValueError('Archive path collision')
        inventory[arc]=item
    def artifact(directory,arc):
        directory=Path(directory)
        marker=json.loads((directory/'complete.json').read_text())
        if marker['status']!='complete':raise ValueError('Incomplete source artifact')
        for name,digest in marker['files'].items():
            add(directory/name,arc+'/'+name,name.endswith('.json'),digest)
        add(directory/'complete.json',arc+'/complete.json',True)
        return marker
    scopes=[('m0','m0')]+[(b,p) for b in BRANCHES for p in POINTS]
    for branch,point in scopes:
        for stage in STAGES:
            key=('clip' if branch=='m0' else 'clip_'+branch)+'__'+point+'__'+stage
            entry=state['completed'][key]
            assert len(entry['outputs'])==(12 if stage=='b__MMEB12' else 1)
            add(entry['log'],'logs/evaluation/'+key+'.log',True)
            for directory in entry['outputs']:
                directory=Path(directory);summary=json.loads((directory/'summary.json').read_text());ident=summary['identity']
                kind=ident['stage'];checkpoint=ident['embeddings']['checkpoint'] if kind=='A0-A6' else ident['checkpoint']
                assert checkpoint['model']=='clip' and checkpoint['point']==point
                if branch=='m0':assert checkpoint['branch'] is None
                else:assert checkpoint['branch']==branch and checkpoint['seed']==42
                task=ident['embeddings']['view']['probe_name'] if kind=='A0-A6' else ident['dataset'].get('task',ident['dataset'].get('dataset'))
                arc=f'results/{branch}/{point}/{kind}/{task}'
                marker=artifact(directory,arc)
                assert marker['identity']==ident
                numeric={k:v for section in ['A0','A1','A2','A3','A4','A5','A6'] for k,v in flatten(summary.get(section,{}),section).items()} if kind=='A0-A6' else flatten(summary['metrics'])
                summaries.append({'branch':branch,'point':point,'stage':kind,'task':task,'numeric':numeric,
                                  'summary_path':arc+'/summary.json','summary_sha256':sha(directory/'summary.json'),
                                  'checkpoint':checkpoint})
                if kind=='A0-A6':
                    for field in ['embeddings','reference']:
                        eid=ident[field];digest=identity_hash(eid)
                        if digest not in vectors:
                            folder=root/'outputs/evaluation/formal_full_v1/embeddings'/digest
                            vmarker=artifact(folder,'embeddings/'+digest)
                            assert vmarker['identity']==eid
                            vectors[digest]={'identity':eid,'directory':'embeddings/'+digest}
    assert len(summaries)==496 and len(vectors)==62
    assert len({(x['branch'],x['point'],x['stage'],x['task']) for x in summaries})==496
    for branch in BRANCHES:
        folder=root/'outputs/training/formal_full_v1'/('clip_'+branch)/'seed_42'
        manifest=json.loads((folder/'run_manifest.json').read_text())
        assert manifest['status']=='complete' and manifest['completed_steps']==15003
        for file in folder.rglob('*'):
            if file.is_file() and file.suffix in ['.json','.jsonl','.log']:
                add(file,'training/'+branch+'/'+file.relative_to(folder).as_posix(),True)
    add(root/'outputs/training_queues/formal_full_seed42_20260922/train_runs.frozen.yaml','protocol/train_runs.frozen.yaml',True)
    add(root/'outputs/evaluation/formal_full_v1/queue/formal_ab.frozen.yaml','protocol/formal_ab.frozen.yaml',True)
    for name,digest in state['identity']['sources'].items():add(root/name,'source/'+name,True,digest)
    # Copy the existing reviewed workbook/figures; no recalculation or metric changes.
    folder=root/'outputs/tables/clip_six_branches_20260923'
    for file in folder.glob('*.xlsx'):add(file,'tables/'+file.name,True)
    plots=root/'outputs/plots/clip_six_branches_20260923'
    for file in plots.rglob('*'):
        if file.is_file() and file.suffix in ['.png','.svg','.html','.json','.md']:
            add(file,'retrieval_plots/'+file.relative_to(plots).as_posix(),True)
    loss=root/'outputs/plots/clip_six_branches_loss_20260923'
    for file in loss.rglob('*'):
        if file.is_file() and file.suffix in ['.png','.svg','.json','.jsonl','.md']:
            add(file,'loss/'+file.relative_to(loss).as_posix(),True)
    readme='''# CLIP 六分支 A / B1 / Local 结果包

先打开 index.html：可按任务、指标筛选，表格列为1/5/20/50/100%检查点，行为六分支，并附M0基线。
tables/ 中是已核对的B1/Local Excel，retrieval_plots/ 和 loss/ 是此前轨迹图及损失图。

范围：CLIP ViT-L/14，seed42，1个epoch=15003步。六分支：Standard、Fixed-2M、Fixed-3M FN-off、Fixed-3M FN-on、Mixed-2M、Mixed-3M FN-off。没有加入第七分支，也没有混入旧先验或新BEiT-3两轮计划。

共31个状态（六分支×五点+共享M0），每状态16份结果：A的LCS10000/COCO4407两份、B1的COCO/Flickr两份、Local十二份，合计496份。A0–A6所有数值字段都可在离线总览中检索；嵌套完整原始JSON保留。B1包含双向R@1/5/10、mR；Local包含全部现有任务指标。

## 文件
- results/：按分支/检查点/测评类别/数据集排列的原始summary与complete文件；完整包另含逐样本、逐查询及分布NPZ数组。
- embeddings/：A使用的62份原始向量及来源元数据，包含共享M0；完整包包含raw.npz，小包只保留元数据。
- summaries.json：496行汇总索引，原始数值精度保留；numeric路径使用斜杠展开，原始层级见summary.json。
- training/：训练/验证日志、运行清单、检查点元数据；没有权重。
- protocol/：当次冻结训练/评测配置；source/是按评测报告SHA核验的源代码。
- package_manifest.json：来源绝对路径、包内路径、文件大小及SHA-256。旧JSON中的服务器路径属于来源信息，离线查看请用包内索引。

小型汇总包与完整数据包的所有JSON数值一致。小包不含任何NPZ数组；完整包包含全部已完成结果数组及A原始向量。不含模型、优化器、图片、训练/验证集或原始下游数据集。

## 口径
离线总览按原始比例显示召回率（0.12345=12.345%），保留完整数值在悬停提示；Excel召回率显示为百分比。排名越低越好，但A类几何/模长等指标没有统一的优劣方向，表中不标“最佳”。null表示未定义，不填0。
进度对应旧一轮训练，不能直接与新两轮计划的相同百分比混合比较。不同分支训练loss不能直接当作共同任务评分。M0是共享基线，不是重复六次训练。

## English
Six completed CLIP branches, seed 42, one epoch, five checkpoints each, plus the shared M0: 496 result summaries and 62 raw embedding artifacts. Open index.html offline. Numeric values are unchanged; recall is stored as a fraction. The full archive adds all per-sample/query arrays and raw embeddings; the compact archive contains summaries, logs, protocol, reviewed plots/workbook and provenance only. No model weights or source datasets are included. All source artifact hashes were verified before packaging.
'''
    (out/'README.md').write_text(readme,encoding='utf-8')
    payload={'branches':BRANCHES,'points':POINTS,'rows':summaries,'embeddings':vectors,'scope':'CLIP seed42 one epoch; six completed branches plus M0'}
    (out/'summaries.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>CLIP 六分支 A/B1/Local 结果</title>
<style>body{font:15px system-ui,sans-serif;color:#18243a;background:#f4f6fa;margin:32px}h1{margin-bottom:8px}.note{color:#536176}select,input{font:inherit;padding:8px;margin:8px 12px 16px 0;max-width:95%}table{border-collapse:collapse;background:white;min-width:900px;width:100%}td,th{padding:12px;border:1px solid #dce1ea;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#e8edf5}.scroll{overflow:auto}a{color:#245db0}.links{margin:18px 0}.missing{color:#8490a1}</style>
<h1>CLIP 六分支 A / B1 / Local</h1><p class="note">seed 42 · 1 epoch · 五个检查点 + M0 · 496份原始结果。训练未重跑，指标未重算。</p>
<div class="links"><a href="README.md">口径与文件说明</a> · <a href="retrieval_plots/index.html">检索轨迹图</a> · <a href="summaries.json">完整数值索引</a></div>
<label>任务 <select id="task"></select></label><label>筛选指标 <input id="filter" placeholder="例如 A0、delta_out、R@1"></label><br>
<label>指标 <select id="metric" style="min-width:500px"></select></label>
<p id="baseline"></p><div class="scroll"><table><thead><tr><th>分支</th><th>1%</th><th>5%</th><th>20%</th><th>50%</th><th>100%</th></tr></thead><tbody id="body"></tbody></table></div>
<p class="note">原始数值显示至8位有效数字，悬停查看完整精度；R@K为0–1比例。点击单元格查看原始summary。null显示“未定义”。不按A类指标数值大小判断优劣。</p>
<script>const DATA=__DATA__;
const task=document.getElementById('task'),metric=document.getElementById('metric'),filter=document.getElementById('filter');
const keys=[...new Set(DATA.rows.map(r=>r.stage+' | '+r.task))];
for(const k of keys){let o=new Option(k,k);task.add(o)}
function relevant(){return DATA.rows.filter(r=>r.stage+' | '+r.task===task.value)}
function metrics(){let old=metric.value;metric.innerHTML='';let paths=[...new Set(relevant().flatMap(r=>Object.keys(r.numeric)))].filter(k=>k.toLowerCase().includes(filter.value.toLowerCase())).sort();for(let k of paths)metric.add(new Option(k,k));if(paths.includes(old))metric.value=old;render()}
function value(v){return v===null?'未定义':v===undefined?'—':Number(v).toPrecision(8)}
function render(){let rows=relevant(),m=metric.value;document.getElementById('body').innerHTML='';let base=rows.find(r=>r.branch==='m0');document.getElementById('baseline').textContent='M0：'+(base?value(base.numeric[m]):'—');for(let b of DATA.branches){let tr=document.createElement('tr'),th=document.createElement('td');th.textContent=b;tr.append(th);for(let p of DATA.points){let r=rows.find(r=>r.branch===b&&r.point===p),td=document.createElement('td');if(r){let a=document.createElement('a');a.href=r.summary_path;a.textContent=value(r.numeric[m]);a.title=String(r.numeric[m]);td.append(a)}else{td.textContent='—'}tr.append(td)}document.getElementById('body').append(tr)}}
task.onchange=metrics;filter.oninput=metrics;metric.onchange=render;metrics();</script></html>'''
    # Inline data keeps the overview usable from file:// without a local web server.
    data=json.dumps({'rows':summaries,'branches':BRANCHES,'points':POINTS},ensure_ascii=False,allow_nan=False).replace('</','<\\/')
    page=page.replace('__DATA__',data)
    (out/'index.html').write_text(page,encoding='utf-8')
    for name in ['README.md','summaries.json','index.html']:add(out/name,name,True)
    manifest={'schema_version':1,'created_epoch':time.time(),'branches':BRANCHES,'points':POINTS,
              'summary_count':len(summaries),'embedding_artifacts':len(vectors),'source_artifacts_verified':True,
              'files':list(inventory.values())}
    (out/'package_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    archives=[]
    for mode in ['summary','full']:
        target=out/('CLIP_six_branches_AB1Local_'+mode+'.zip')
        selected=[x for x in inventory.values() if mode=='full' or x['light']]
        with zipfile.ZipFile(target,'x',allowZip64=True) as z:
            for entry in selected:
                compression=zipfile.ZIP_STORED if entry['archive_path'].endswith('.npz') else zipfile.ZIP_DEFLATED
                z.write(entry['source'],entry['archive_path'],compress_type=compression,compresslevel=1)
            z.write(out/'package_manifest.json','package_manifest.json',compress_type=zipfile.ZIP_DEFLATED,compresslevel=1)
        # Read every member back and compare its uncompressed SHA, not only the ZIP CRC.
        with zipfile.ZipFile(target) as z:
            assert len(z.namelist())==len(set(z.namelist()))==len(selected)+1
            for entry in selected:
                h=hashlib.sha256()
                with z.open(entry['archive_path']) as f:
                    for chunk in iter(lambda:f.read(4*1024*1024),b''):h.update(chunk)
                if h.hexdigest()!=entry['sha256']:raise ValueError('Archive read-back SHA mismatch')
            assert json.loads(z.read('package_manifest.json'))==manifest
        report={'name':target.name,'bytes':target.stat().st_size,'sha256':sha(target),'members':len(selected)+1,'all_member_hashes_verified':True}
        archives.append(report);print(json.dumps(report),flush=True)
    (out/'verification.json').write_text(json.dumps({'archives':archives,'summary_count':496,'embedding_artifacts':62},indent=2),encoding='utf-8')


if __name__=='__main__':main()
