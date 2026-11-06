'use strict';
// Independent SCREEN-PIXEL capture. Never access the simulator iframe's
// contentWindow, HTML, canvas, internal generator or ground-truth labels.
const $ = id => document.getElementById(id);
const live = $('live'), ctx = live.getContext('2d');
const picker = $('selectCanvas'), pickCtx = picker.getContext('2d');
const scratch = document.createElement('canvas'), scratchCtx=scratch.getContext('2d');
const video=$('capture');
let media=null, roi=null, running=false, busy=false, timer=null, lastBoxes=[], lastFrame=null;
let selected=null, drag=null, lastStats={};
const state = message => $('status').textContent=message;
const clamp=(n,a,b)=>Math.max(a,Math.min(b,n));
function clearStats(){ for(const id of ['count','countGood','countBad']) $(id).textContent='0'; $('latency').textContent='—'; }
function cameraState(text){$('cameraState').textContent=text;}
function renderFrame(frame,results=[]){
  live.width=frame.width;live.height=frame.height;
  ctx.drawImage(frame,0,0);
  const threshold=Number($('threshold').value)/100;
  let good=0,bad=0;
  for(const item of results){
    const fail=item.defect_probability>=threshold;
    const color=fail?'#ff778b':'#67ebbb';
    if(fail) bad++; else good++;
    const x=item.x,y=item.y,w=item.w,h=item.h;
    ctx.lineWidth=Math.max(3,Math.min(w,h)/85);
    ctx.strokeStyle=color;ctx.strokeRect(x,y,w,h);
    const label=(fail?'БРАК':'НОРМА')+' '+Math.round(100*(fail?item.defect_probability:1-item.defect_probability))+'%';
    const fontSize=Math.max(16,Math.min(24,Math.round(w/11)));
    ctx.font=`bold ${fontSize}px system-ui`;
    const tw=ctx.measureText(label).width;
    const th=fontSize+10;
    const top=Math.max(0,y-th);
    ctx.fillStyle=color;ctx.fillRect(x,top,tw+17,th);
    ctx.fillStyle='#0b2420';ctx.fillText(label,x+8,top+fontSize+2);
  }
  $('count').textContent=String(results.length);
  $('countGood').textContent=String(good);
  $('countBad').textContent=String(bad);
}
function reclassify(){if(lastFrame)renderFrame(lastFrame,lastBoxes);}
$('threshold').addEventListener('input',()=>{$('thresholdValue').textContent=$('threshold').value+'%';reclassify()});
$('fps').addEventListener('input',()=>{$('fpsValue').textContent=$('fps').value+' к/с'});
function stopCapture(){
  if(lastFrame && typeof lastFrame.close === 'function')lastFrame.close();
  running=false;clearTimeout(timer);timer=null;
  if(media){for(const track of media.getTracks())track.stop();media=null;}
  video.srcObject=null;busy=false;roi=null;lastFrame=null;lastBoxes=[];
  $('start').disabled=false;$('roi').disabled=true;$('stop').disabled=true;
  $('placeholder').hidden=false;cameraState('ОЖИДАНИЕ ИСТОЧНИКА');clearStats();
  state('Захват остановлен.');
}
$('stop').addEventListener('click',stopCapture);
$('start').addEventListener('click',async()=>{
  if(!navigator.mediaDevices?.getDisplayMedia){
    state('Захват экрана недоступен: нужен HTTPS-домен или localhost, а также поддерживаемый браузер.');return;
  }
  try{
    media=await navigator.mediaDevices.getDisplayMedia({video:{frameRate:12},audio:false});
    video.srcObject=media;
    await video.play();
    if(!video.videoWidth)await new Promise(resolve=>video.addEventListener('loadedmetadata',resolve,{once:true}));
    media.getVideoTracks()[0].addEventListener('ended',stopCapture,{once:true});
    $('start').disabled=true;$('roi').disabled=false;$('stop').disabled=false;
    state('Источник подключён. Выдели прямоугольник вокруг камеры.');
    openROI();
  }catch(err){state('Не удалось подключиться: '+err.message);stopCapture();}
});
$('roi').addEventListener('click',openROI);
function openROI(){
  if(!media||!video.videoWidth)return;
  running=false;clearTimeout(timer);
  $('roiModal').hidden=false;
  // This is a capture frame from the browser, not simulator API data.
  picker.width=video.videoWidth;picker.height=video.videoHeight;
  pickCtx.drawImage(video,0,0,picker.width,picker.height);
  $('selection').style.display='none';$('applyRoi').disabled=true;
  selected=null;
}
const wrap=$('selectWrap');
function point(ev){const box=picker.getBoundingClientRect();return {x:clamp((ev.clientX-box.left)/box.width,0,1),y:clamp((ev.clientY-box.top)/box.height,0,1)};}
function drawSelection(){
  if(!selected)return;
  const box=picker.getBoundingClientRect();
  $('selection').style.left=(selected.x*100)+'%';
  $('selection').style.top=(selected.y*100)+'%';
  $('selection').style.width=(selected.w*100)+'%';
  $('selection').style.height=(selected.h*100)+'%';
  $('selection').style.display='block';
  $('applyRoi').disabled=selected.w<.05||selected.h<.05;
}
wrap.addEventListener('pointerdown',ev=>{if(ev.button!==0)return;wrap.setPointerCapture(ev.pointerId);drag=point(ev);selected={x:drag.x,y:drag.y,w:0,h:0};drawSelection()});
wrap.addEventListener('pointermove',ev=>{if(!drag)return;const p=point(ev);selected={x:Math.min(p.x,drag.x),y:Math.min(p.y,drag.y),w:Math.abs(p.x-drag.x),h:Math.abs(p.y-drag.y)};drawSelection()});
wrap.addEventListener('pointerup',ev=>{if(!drag)return;drag=null;drawSelection()});
$('cancelRoi').addEventListener('click',()=>{ $('roiModal').hidden=true;if(roi){running=true;schedule();} });
$('applyRoi').addEventListener('click',()=>{
  if(!selected || selected.w<.05 || selected.h<.05)return;
  roi=selected;
  $('roiModal').hidden=true;running=true;
  $('placeholder').hidden=true;
  cameraState('● PYTHON LIVE');state('Поток передаётся в Python: только выделенные пиксели.');
  schedule();
});
function schedule(){if(!running)return;clearTimeout(timer);timer=setTimeout(processFrame,1000/Number($('fps').value));}
async function processFrame(){
  if(!running || busy || !roi || !media)return;
  busy=true;
  try{
    const W=video.videoWidth,H=video.videoHeight;
    const sx=Math.round(roi.x*W),sy=Math.round(roi.y*H),sw=Math.round(roi.w*W),sh=Math.round(roi.h*H);
    const scale=Math.min(1,1200/Math.max(sw,sh));
    scratch.width=Math.round(sw*scale);scratch.height=Math.round(sh*scale);
    scratchCtx.drawImage(video,sx,sy,sw,sh,0,0,scratch.width,scratch.height);
    const snapshot=await createImageBitmap(scratch);
    const blob=await new Promise((resolve,reject)=>scratch.toBlob(b=>b?resolve(b):reject(Error('JPEG encoder failed')),'image/jpeg',.80));
    const response=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'image/jpeg'},body:blob,cache:'no-store'});
    if(!response.ok)throw Error('HTTP '+response.status+': '+await response.text());
    const result=await response.json();
    if(running){
      if(lastFrame && typeof lastFrame.close === 'function')lastFrame.close();
      lastFrame=snapshot;lastBoxes=result.detections;
      renderFrame(snapshot,lastBoxes);
      $('latency').textContent=Math.round(result.latency_ms)+' мс';
      state('Анализ: '+result.detections.length+' лотков · '+Math.round(result.latency_ms)+' мс на VPS.');
    }else snapshot.close();
  }catch(err){state('Ошибка анализа: '+err.message);cameraState('ОШИБКА ПОТОКА');}
  finally{busy=false;schedule();}
}
fetch('/api/health').then(r=>r.json()).then(data=>{state(data.ok?'Python-модель загружена на сервере. Подключи вкладку.':'Модель пока не готова.')}).catch(()=>state('Python API недоступен. Проверь контейнер приложения.'));
