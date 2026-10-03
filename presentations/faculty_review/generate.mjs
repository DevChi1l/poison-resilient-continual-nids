/** Editable faculty deck. Data, copy, sources and figure paths live in content.json. */
import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL, fileURLToPath} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
const contentPath=path.resolve(process.argv[2]??path.join(here,'content.json'));
const output=path.resolve(process.argv[3]??path.join(here,'faculty_review.pptx'));
const build=path.resolve(process.env.PRESENTATION_BUILD_DIR??path.join(here,'.build'));
const runtime=process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES;
if(!runtime) throw Error('Set CODEX_PRIMARY_RUNTIME_NODE_MODULES to the supplied runtime.');
const {Presentation,PresentationFile}=await import(pathToFileURL(path.join(runtime,'@oai/artifact-tool/dist/artifact_tool.mjs')));
const skill=process.env.PRESENTATIONS_SKILL_DIR??'/root/.codex/skills/builtins/presentations';
const {resolvePresentationFont,finalizePresentation}=await import(pathToFileURL(path.join(skill,'container_tools/artifact_tool_utils.mjs')));
const d=JSON.parse(await fs.readFile(contentPath,'utf8')),c=d.theme;
const font=resolvePresentationFont({fontFamily:c.font});
const metricSlot=d.result_slots.per_class_metrics;
if(metricSlot.status==='VERIFIED' && metricSlot.metrics_json_path){
 const m=JSON.parse(await fs.readFile(path.resolve(path.dirname(contentPath),metricSlot.metrics_json_path),'utf8'));
 metricSlot.table={headers:['Class','Precision','Recall','F1','Support'],rows:Array.from({length:8},(_,i)=>{const a=m.per_class[String(i)];if(!a)throw Error('Missing class '+i);if(!['precision','recall','f1'].every(k=>typeof a[k]==='number'&&Number.isFinite(a[k])&&a[k]>=0&&a[k]<=1)||!Number.isSafeInteger(a.support)||a.support<0)throw Error('Invalid or missing metrics for class '+i);return [a.name,...['precision','recall','f1'].map(k=>Number(a[k]).toFixed(3)),String(a.support)];})};
}
const p=Presentation.create({slideSize:{width:1280,height:720}});
await fs.mkdir(build,{recursive:true}); await fs.mkdir(path.dirname(output),{recursive:true});
function text(s,str,x,y,w,h,size=28,color=c.ink,bold=false){
 const a=s.shapes.add({geometry:'textbox',name:str.slice(0,55),position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 a.text=String(str); a.text.style={typeface:font,fontSize:size,color,bold,autoFit:'none'}; return a;
}
function box(s,x,y,w,h,fill,line='none'){return s.shapes.add({geometry:'rect',position:{left:x,top:y,width:w,height:h},fill,line:{fill:line,width:line==='none'?0:1.5}})}
function table(s,headers,rows,x,y,w,h,widths){
 const values=[headers,...rows]; const t=s.tables.add({rows:values.length,columns:headers.length,left:x,top:y,width:w,height:h,values,columnWidths:widths});
 t.borders.assign({fill:'#D8E1E4',width:1,style:'solid'});
 for(let r=0;r<values.length;r++)for(let k=0;k<headers.length;k++){
  const a=t.getCell(r,k);a.fill=r===0?c.navy:r%2?'#FFFFFF':'#EDF2F3';a.text.style={typeface:font,fontSize:24,color:r===0?'#FFFFFF':c.ink,bold:r===0||k===0,autoFit:'none'};
 }
 return t;
}
function base(v,i){const s=p.slides.add();s.background.fill=c.paper;
 text(s,v.status,64,30,1140,30,18,v.status.startsWith('PENDING')?c.rust:c.teal,true);
 text(s,v.title,64,79,1150,100,40,c.navy,true);
 text(s,String(i+1).padStart(2,'0'),1176,665,50,27,18,c.muted);
 s.speakerNotes.textFrame.setText(v.notes+'\n\nEvidence:\n'+v.sources.map(x=>x.startsWith('Library:')||x.startsWith('User')?x:`https://github.com/DevChi1l/poison-resilient-continual-nids/blob/${d.evidence_base_sha}/${x.split(':')[0]}`).join('\n')+'\n\nEvidence base: '+d.evidence_base_sha+'. Review date: '+d.as_of+'.');return s;}
function callout(s,v){if(v.callout)text(s,v.callout,64,599,1120,60,24,c.rust,true)}
function columns(s,v,y=320,h=220){text(s,v.leftTitle,64,y,535,42,27,c.teal,true);text(s,v.left,64,y+55,535,h,27);text(s,v.rightTitle,672,y,545,42,27,c.teal,true);text(s,v.right,672,y+55,545,h,27)}
function steps(s,v,y=190,h=135){const n=v.steps.length,g=26,w=(1152-g*(n-1))/n;v.steps.forEach((a,j)=>{const x=64+j*(w+g);box(s,x,y,w,h,'#E5EFF0');text(s,a[0],x+17,y+16,w-32,42,n===4?24:26,c.teal,true);text(s,a[1],x+17,y+62,w-32,h-60,23);if(j<n-1)s.shapes.add({geometry:'rightArrow',position:{left:x+w+4,top:y+h/2-7,width:18,height:14},fill:c.teal,line:{fill:'none',width:0}});});}
function metrics(s,v){v.metrics.forEach((m,j)=>{let x=64+j*393;text(s,m[0],x,193,376,84,58,j===2?c.rust:c.teal,true);text(s,m[1],x,288,360,67,23,c.muted);});}
async function slot(s,k,x,y,w,h){
 const v=d.result_slots[k];text(s,v.title,x,y,w,45,29,c.teal,true);
 if(v.status==='VERIFIED'&&v.image_path){
  const f=path.resolve(path.dirname(contentPath),v.image_path);const bytes=await fs.readFile(f);s.images.add({blob:new Uint8Array(bytes),contentType:f.toLowerCase().endsWith('.jpg')?'image/jpeg':'image/png',fit:'contain',alt:v.caption,position:{left:x,top:y+54,width:w,height:h-128}});
 }else if(v.status==='VERIFIED'&&v.table){table(s,v.table.headers,v.table.rows,x,y+54,w,h-128);}
 else{box(s,x,y+54,w,h-128,'#EDF2F3','#AFBEC3');text(s,'AWAITING RESULTS',x+24,y+88,w-48,44,26,c.muted,true);text(s,'No final data supplied',x+24,y+147,w-48,70,24,c.muted);}
 text(s,v.caption,x,y+h-63,w,66,22,c.muted);
}
for(let i=0;i<d.slides.length;i++){
 const v=d.slides[i],s=base(v,i);
 switch(v.layout){
 case 'cover': break;
 case 'mapping':text(s,v.lead,64,216,550,85,46,c.teal,true);text(s,v.sub,64,324,550,145,30);text(s,v.callout,64,520,540,120,26,c.rust,true);table(s,v.headers,v.rows,686,187,530,438,[70,300,160]);break;
 case 'protocol':steps(s,v);columns(s,v,355,165);callout(s,v);break;
 case 'model':steps(s,v,190,148);columns(s,v,376,160);callout(s,v);break;
 case 'baseline':metrics(s,v);columns(s,v,377,150);callout(s,v);break;
 case 'continual':steps(s,v,187,142);table(s,v.headers,v.rows,64,371,1152,166,[420,244,244,244]);callout(s,v);break;
 case 'poison':columns(s,v,190,170);table(s,v.headers,v.rows,64,412,1152,163,[360,248,248,296]);callout(s,v);break;
 case 'mitigation':steps(s,v,185,128);table(s,v.headers,v.rows,64,357,1152,198,[420,366,366]);callout(s,v);break;
 case 'replay':table(s,v.headers,v.rows,64,197,1152,355,[500,326,326]);callout(s,v);break;
 case 'novelty':metrics(s,v);columns(s,v,382,150);callout(s,v);break;
 case 'preparation':metrics(s,v);columns(s,v,382,150);callout(s,v);break;
 case 'perclass':{const q=d.result_slots.per_class_metrics;table(s,q.table.headers,q.table.rows,64,183,1152,385,[310,218,208,208,208]);callout(s,v);break;}
 case 'slots':{let n=v.slot_ids.length,g=36,w=(1152-g*(n-1))/n;for(let j=0;j<n;j++)await slot(s,v.slot_ids[j],64+j*(w+g),196,w,372);callout(s,v);break;}
 case 'closing':columns(s,v,214,265);callout(s,v);break;
 }
}
// Cover uses the same editable objects as all other slides; update through object collection.
const cover=p.slides.items[0];
const coverShapes=cover.shapes.items;
coverShapes[0].text.style={typeface:font,fontSize:18,color:'#72CCD0',bold:true};
coverShapes[1].position={left:64,top:106,width:1120,height:185};coverShapes[1].text.style={typeface:font,fontSize:68,color:'#FFFFFF',bold:true,autoFit:'none'};
coverShapes[2].text.style={typeface:font,fontSize:18,color:'#B4C7CF'};
text(cover,d.slides[0].subtitle,68,310,1100,48,32,'#9CDADD');
text(cover,d.slides[0].question,68,391,1050,102,32,'#FFFFFF');
d.slides[0].objectives.forEach((a,i)=>text(cover,a,68+i*400,540,355,86,25,'#D8E8EC'));
text(cover,d.slides[0].footer,68,665,1050,35,20,'#B4C7CF');
const candidate=path.join(build,'candidate.pptx');await(await PresentationFile.exportPptx(p)).save(candidate);
const requirements={explicitTotalSlideCount:d.slides.length,requiredNativeTableOwnerSlides:[2,6,7,8,9,13],requiredNativeChartOwnerSlides:[]};
await finalizePresentation({...requirements,workspaceDir:process.env.PRESENTATION_WORKSPACE??path.dirname(path.dirname(path.dirname(output))),candidatePath:candidate,finalPath:output,pythonExecutable:process.env.CODEX_PRIMARY_RUNTIME_PYTHON,integrityValidatorPath:path.join(skill,'container_tools/inspect_presentation_package_integrity.py'),layoutValidatorPath:path.join(skill,'container_tools/inspect_presentation_layout_geometry.py'),layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit',...requirements.requiredNativeTableOwnerSlides.flatMap(n=>['--require-native-table-slide',String(n)])],fontPolicy:{basis:'design',families:[font]},verifyArtifactToolImport:true,receiptPath:path.join(build,'validation.json')});
for(let i=0;i<p.slides.items.length;i++){const s=p.slides.items[i];const image=await p.export({slide:s,format:'png',scale:1});await fs.writeFile(path.join(build,`slide-${String(i+1).padStart(2,'0')}.png`),new Uint8Array(await image.arrayBuffer()));}
console.log(JSON.stringify({output,slides:p.slides.items.length,font,previews:build}));
