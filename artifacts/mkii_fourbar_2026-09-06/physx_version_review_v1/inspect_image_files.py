import hashlib,io,json,re,subprocess,tarfile,time,uuid

def command(argv):
    result=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
    return result.stdout

def inspect(target):return json.loads(command(['docker','inspect',target]))[0]

def file_bytes(container,path):
    blob=command(['docker','cp',container+':'+path,'-'])
    with tarfile.open(fileobj=io.BytesIO(blob),mode='r:') as archive:
        member=next(m for m in archive.getmembers() if m.isfile())
        return archive.extractfile(member).read()

label='hexapod-physx-metadata-'+uuid.uuid4().hex[:12]
report={'operation':'never-started image filesystem inspection','gpu_process_started':False,'container_started':False,'name':label,'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'files':[]}
container=None
try:
    image=json.loads(command(['docker','image','inspect','isaac-lab-base']))[0]
    report['image']={k:image.get(k) for k in ['Id','RepoTags','RepoDigests','Created','Os','Architecture']}
    container=command(['docker','create','--name',label,'--network','none','--read-only','--entrypoint','/bin/true','--label','hexapod.metadata_only=true','-e','NVIDIA_VISIBLE_DEVICES=void','isaac-lab-base']).decode().strip()
    current=inspect(container)
    report['container_id']=container
    report['state_before']={k:current['State'].get(k) for k in ['Status','Running','Pid','StartedAt']}
    report['container_image_id']=current['Image']
    if current['State']['Running'] or current['State']['Pid']!=0:raise RuntimeError('Metadata container unexpectedly running')
    stem='/isaac-sim/extscache/'
    extension='omni.physx-110.1.13+110.1.2.la64.r.cp312.u7f4/'
    paths=['/isaac-sim/VERSION',stem+extension+'config/extension.toml',stem+extension+'docs/CHANGELOG.md',stem+'omni.physx.foundation-110.1.13+110.1.2.la64.r.cp312.u7f4/config/extension.toml',stem+'omni.physx.gpu-110.1.13+110.1.2.la64.r.cp312.u7f4/config/extension.toml',stem+'omni.physx.bundle-110.1.13+110.1.2.la64.r.cp312.u7f4/docs/CHANGELOG.md',stem+extension+'bin/libomni.physx.plugin.so']
    for path in paths:
        try:
            data=file_bytes(container,path)
            entry={'path':path,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
            if path.endswith('.so'):
                strings=[(m.start(),m.group().decode()) for m in re.finditer(rb'[\x20-\x7e]{4,}',data)]
                needles=['lib_physx_build','c38f7d1','110.1.13','5.9.0','Detected an articulation','SnXmlSerialization.cpp','PhysX SDK']
                selected=set()
                for i,(_,s) in enumerate(strings):
                    if any(n in s for n in needles):selected.update(range(max(0,i-3),min(len(strings),i+4)))
                entry['selected_printable_strings']=[{'offset':strings[i][0],'text':strings[i][1]} for i in sorted(selected)]
            else:entry['text']=data.decode()
            report['files'].append(entry)
        except Exception as error:report['files'].append({'path':path,'error':str(error)})
    report['state_after']={k:inspect(container)['State'].get(k) for k in ['Status','Running','Pid','StartedAt']}
finally:
    if container:
        current=inspect(container)
        if current['State']['Running'] or current['State']['Pid']!=0:raise RuntimeError('Will not remove an unexpectedly running metadata container')
        command(['docker','rm',container]);report['exact_stopped_container_removed']=True
print(json.dumps(report,indent=2))
