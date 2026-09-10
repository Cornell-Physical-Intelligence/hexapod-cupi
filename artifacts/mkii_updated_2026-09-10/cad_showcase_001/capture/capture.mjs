import {createRequire} from 'node:module';
import fs from 'node:fs';
import path from 'node:path';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
const require=createRequire(import.meta.url);
const {chromium}=require('/Users/andreboufama/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const out=path.resolve(path.dirname(new URL(import.meta.url).pathname),'..');
const browser=await chromium.launch({channel:'chromium',headless:true,args:['--use-angle=metal','--enable-gpu']});
try{
 const page=await browser.newPage({viewport:{width:1920,height:1080},deviceScaleFactor:1});
 const errors=[];page.on('pageerror',e=>errors.push(String(e)));
 await page.goto('http://127.0.0.1:8371/artifacts/mkii_updated_2026-09-10/cad_showcase_001/capture/index.html',{waitUntil:'domcontentloaded'});
 await page.waitForFunction(()=>document.body.dataset.modelReady==='true',{timeout:120000});
 const initial=await page.evaluate(()=>window.showcaseFrame(0));
 console.log(JSON.stringify({ready:initial,errors}));
 await page.screenshot({path:path.join(out,'poster_final.png')});
 if(process.argv.includes('--preview-only'))process.exitCode=0;
 else{
  const fps=25,frames=600,trace=[];
  const encoder=spawn('/opt/homebrew/bin/ffmpeg',['-hide_banner','-loglevel','warning','-y','-f','image2pipe','-vcodec','mjpeg','-framerate',String(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',path.join(out,'hexapod_detailed_cad_showcase_final.mp4')],{stdio:['pipe','ignore','pipe']});
  let stderr='';encoder.stderr.on('data',d=>{stderr+=d.toString()});
  for(let i=0;i<frames;i++){
   trace.push(await page.evaluate(t=>window.showcaseFrame(t),i/fps));
   const frame=await page.screenshot({type:'jpeg',quality:94});
   if(!encoder.stdin.write(frame))await once(encoder.stdin,'drain');
   if(i%50===0)console.log(JSON.stringify({frame:i,total:frames}));
  }
  encoder.stdin.end();const [code]=await once(encoder,'close');
  fs.writeFileSync(path.join(out,'capture_trace_final.json'),JSON.stringify({fps,frames,duration_s:frames/fps,errors,trace},null,2)+'\n');
  fs.writeFileSync(path.join(out,'ffmpeg_final.log'),stderr);if(code!==0)throw Error('ffmpeg exit '+code);
  console.log(JSON.stringify({video:path.join(out,'hexapod_detailed_cad_showcase_final.mp4'),bytes:fs.statSync(path.join(out,'hexapod_detailed_cad_showcase_final.mp4')).size}));
 }
}finally{await browser.close();}
