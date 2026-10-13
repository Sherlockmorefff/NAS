"""Read code entries only from a finite set of known historical archives."""
import hashlib,tarfile,zipfile
from .common import *
class Counted:
 def __init__(self,f):self.f=f;self.bytes=0
 def read(self,*args):
  b=self.f.read(*args);self.bytes+=len(b);return b
 def seek(self,*args):return self.f.seek(*args)
 def tell(self):return self.f.tell()
 def seekable(self):return True
def recover(root,repo):
 if (root/'backup_search.json').exists():return read(root/'backup_search.json')
 discovered=read(root/'source_discovery.json')['archive_paths_not_opened']
 selected=[p for p in discovered if 'dkl_uncertainty_diagnosis_20260720' in p or p.startswith((V1+'/',V2+'/',ORIGIN+'/'))]
 targets={v['recorded_code_path']:v['recorded_sha256'] for v in read(root/'historical_version_search.json')['bindings']}
 write(root/'backup_allowlist.json',{'frozen_at':now(),'archives':selected,'entry_filter':'regular .py entries whose basename matches the recorded source-hash target filenames; no JSON/model/score member opened','targets':targets,'basis':'Exact known same-project archive paths from P0; complete loose snapshots and Git search already completed.'})
 rows=[];total_bytes=0
 for rel in selected:
  p=repo/rel;archive_hash=sha(p);total_bytes+=p.stat().st_size;entries=[]
  with p.open('rb') as f:
   c=Counted(f)
   if rel.endswith('.zip'):
    with zipfile.ZipFile(c) as z:
     for info in z.infolist():
      if Path(info.filename).name not in {Path(t).name for t in targets} or not info.filename.endswith('.py'):continue
      b=z.read(info);entries.append((info.filename,b))
   else:
    with tarfile.open(fileobj=c,mode='r:*') as t:
     for info in t:
      if not info.isfile() or not info.name.endswith('.py') or Path(info.name).name not in {Path(v).name for v in targets}:continue
      b=t.extractfile(info).read();entries.append((info.name,b))
   total_bytes+=c.bytes
  matches=[];checked=[]
  for name,b in entries:
   h=hashlib.sha256(b).hexdigest();checked.append({'entry':name,'sha256':h})
   for target,expected in targets.items():
    if Path(name).name==Path(target).name and h==expected:
     out=root/'recovered_code'/h/Path(target).name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(b);matches.append({'entry':name,'target':target,'sha256':h,'saved_path':str(out.relative_to(root))})
  rows.append({'archive':rel,'archive_sha256':archive_hash,'code_entries_checked':checked,'exact_matches':matches,'opened_data_or_weight_members':False})
 result={'archives':rows,'mixed_file_byte_reads':2*len(rows),'hash_reads':len(rows),'bytes_read':total_bytes,'performance_field_decodes':0,'performance_field_exposures':0,'performance_field_writes':0,'performance_field_uses':0,'all_known_loose_snapshots_and_relevant_git_versions_checked_first':True}
 write(root/'backup_search.json',result);return result
