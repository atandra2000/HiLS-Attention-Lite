import {spawn} from 'node:child_process';
import {mkdtempSync,writeFileSync,existsSync,readFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';
import assert from 'node:assert/strict';
const root=resolve(process.argv[2]||'.');
const file=join(root,'hils_visual_guide.html');
const out=process.argv[3]||tmpdir();
const chrome=spawn('/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',['--headless','--no-first-run','--no-default-browser-check','--remote-debugging-pipe',`--user-data-dir=${mkdtempSync(join(tmpdir(),'hils-guide-'))}`],{stdio:['ignore','ignore','ignore','pipe','pipe']});
let id=0,buffer='';const pending=new Map(),errors=[];
chrome.stdio[4].on('data',chunk=>{buffer+=chunk.toString();let end;while((end=buffer.indexOf('\0'))>=0){const m=JSON.parse(buffer.slice(0,end));buffer=buffer.slice(end+1);if(m.id){const p=pending.get(m.id);pending.delete(m.id);m.error?p.reject(m.error):p.resolve(m.result);}else if(m.method==='Runtime.exceptionThrown')errors.push(m.params);}});
function call(method,params={},sessionId){return new Promise((resolve,reject)=>{const n=++id;pending.set(n,{resolve,reject});chrome.stdio[3].write(JSON.stringify({id:n,method,params,sessionId})+'\0');});}
const timeout=setTimeout(()=>{chrome.kill();process.exit(1)},45000);
try{
const {targetId}=await call('Target.createTarget',{url:'about:blank'});
const {sessionId}=await call('Target.attachToTarget',{targetId,flatten:true});
const c=(method,params)=>call(method,params,sessionId);
const evaluate=async expression=>{const r=await c('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});assert(!r.exceptionDetails,JSON.stringify(r.exceptionDetails));return r.result.value;};
await c('Runtime.enable');await c('Page.enable');await c('Network.enable');await c('Network.emulateNetworkConditions',{offline:true,latency:0,downloadThroughput:0,uploadThroughput:0});
await c('Page.navigate',{url:pathToFileURL(file).href});
for(let i=0;i<100;i++){if(await evaluate("document.readyState==='complete' && !!document.getElementById('routing-grid')"))break;await new Promise(r=>setTimeout(r,50));}
const checks=[];
for(const theme of ['light','dark'])for(const [width,height] of [[1440,900],[390,844]]){
 await c('Emulation.setEmulatedMedia',{features:[{name:'prefers-color-scheme',value:theme}]});
 await c('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:false});
 await evaluate('scrollTo({top:0,behavior:"instant"})');
 await evaluate('new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))');
 const m=await evaluate('({width:innerWidth,scrollWidth:document.documentElement.scrollWidth})');assert(m.scrollWidth<=width,JSON.stringify(m));checks.push({...m,theme});
 const shot=await c('Page.captureScreenshot',{format:'png'});writeFileSync(join(out,`hils-guide-${width}-${theme}.png`),Buffer.from(shot.data,'base64'));
}
const anchors=await evaluate("[...document.querySelectorAll('a')].map(a=>({href:a.href,valid:!a.getAttribute('href').startsWith('#')||!!document.querySelector(a.getAttribute('href'))}))");
for(const a of anchors){assert(a.valid,a.href);if(a.href.startsWith('file:'))assert(existsSync(fileURLToPath(a.href.split('#')[0])),a.href);}
const math=await evaluate(`(()=>{
 const failures=[];let cases=0;
 for(let q=0;q<16;q++)for(let k=1;k<=8;k++)for(let h=0;h<4;h++)for(const equal of [false,true]){
 const r=selectExample(q,k,h,equal);cases++;
 if(r.selected[0]!==q||r.selected.some(j=>j>q)||new Set(r.selected).size!==r.selected.length||r.selected.length!==Math.min(k,q+1)||r.padding!==k-r.selected.length||Math.abs(r.weights.reduce((a,b)=>a+b,0)-1)>1e-12)failures.push({q,k,h,equal});
 if(equal&&JSON.stringify(r.selected)!==JSON.stringify(Array.from({length:Math.min(k,q+1)},(_,i)=>q-i)))failures.push('tie order');
 if(JSON.stringify(r.selected)!==JSON.stringify(selectExample(q,k,0,equal).selected))failures.push('head membership');
 }
 return {cases,failures,rates:[0,1999,2000,53407,53657,53907,61036,61037].map(s=>[s,scheduleAt(s)])};
})()`);assert.equal(math.failures.length,0);assert.equal(math.rates[0][1],.0003/2000);assert.equal(math.rates[1][1],.0003);assert(Math.abs(math.rates[4][1]-.0003)<1e-14);assert(Math.abs(math.rates.at(-1)[1]-.000015)<1e-14);
const interaction=await evaluate(`(()=>{const before=document.getElementById('routing-grid').dataset.weights;document.getElementById('head').value=2;document.getElementById('head').dispatchEvent(new Event('input'));const after=document.getElementById('routing-grid').dataset.weights;document.getElementById('train-step').value=0;document.getElementById('train-step').dispatchEvent(new Event('input'));return {changed:before!==after,phase:document.getElementById('schedule-result').textContent}})()`);assert(interaction.changed);assert(interaction.phase.includes('Phase A'));
await evaluate("document.getElementById('query').focus()");const before=await evaluate('+document.activeElement.value');await c('Input.dispatchKeyEvent',{type:'keyDown',key:'ArrowRight',code:'ArrowRight',windowsVirtualKeyCode:39});await c('Input.dispatchKeyEvent',{type:'keyUp',key:'ArrowRight',code:'ArrowRight',windowsVirtualKeyCode:39});assert.equal(await evaluate('+document.activeElement.value'),before+1);
for(const section of ['architecture','routing','data','training','optimization','decode','evidence']){
 await c('Emulation.setDeviceMetricsOverride',{width:1440,height:900,deviceScaleFactor:1,mobile:false});
 await evaluate(`document.getElementById('${section}').scrollIntoView({behavior:'instant'})`);
 await evaluate('new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))');
 writeFileSync(join(out,`hils-guide-${section}.png`),Buffer.from((await c('Page.captureScreenshot',{format:'png'})).data,'base64'));
}
await c('Emulation.setDeviceMetricsOverride',{width:390,height:844,deviceScaleFactor:1,mobile:false});await evaluate("document.getElementById('routing-grid').scrollIntoView({behavior:'instant'})");await evaluate('new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))');writeFileSync(join(out,'hils-guide-lab-mobile.png'),Buffer.from((await c('Page.captureScreenshot',{format:'png'})).data,'base64'));
await c('Emulation.setDeviceMetricsOverride',{width:1440,height:900,deviceScaleFactor:1,mobile:false});
await evaluate("document.getElementById('schedule').scrollIntoView({block:'center',behavior:'instant'})");
await evaluate('new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)))');
writeFileSync(join(out,'hils-guide-schedule.png'),Buffer.from((await c('Page.captureScreenshot',{format:'png'})).data,'base64'));
const portal=join(root,'../docs_html/docs/guides/master-guide.html');
if(existsSync(portal)){
 const source=readFileSync(portal,'utf8');
 assert(source.includes('href="../../../docs/hils_visual_guide.html"'));
 assert(source.includes('href="../../../docs/hils_receipts.json"'));
}
assert.equal(errors.length,0);
const bytes=readFileSync(file);const receipt={status:'pass',artifact:{sha256:createHash('sha256').update(bytes).digest('hex'),bytes:bytes.length},offline:true,viewports:checks,links:anchors.length,routingAndSchedule:math,interaction,keyboard:'pass',runtimeErrors:errors};writeFileSync(join(out,'hils-guide-browser.json'),JSON.stringify(receipt,null,2));console.log(JSON.stringify(receipt));
}finally{clearTimeout(timeout);chrome.kill();}
