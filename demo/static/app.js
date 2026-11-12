(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const state = {
    status: null, results: null, examples: [], current: null, mode: 'point', source: 'automatic',
    manualPoint: null, manualBox: null, last: {}, lastSource: null, currentRequest: false,
    boxDrag: null, opacity: 72, showGt: false, presentationStep: 0,
  };
  const failureData = [
    {id:475292, category:'Person', label:'Part instead of whole', point_iou:.20, box_iou:.88, explanation:'A single point identifies only part of the person, not the full object.'},
    {id:1116028, category:'Keyboard', label:'One key, not keyboard', point_iou:.01, box_iou:.88, explanation:'The point prompt selects one key instead of the entire keyboard.'},
    {id:1917299, category:'Carrot', label:'Instances merged', point_iou:.58, box_iou:.25, explanation:'The box contains adjacent objects and the model merges several carrots.'},
    {id:1169777, category:'Backpack', label:'Both prompts struggle', point_iou:.19, box_iou:.25, explanation:'A small backpack is mistaken for a different image region.'},
  ];
  const milestones = ['dataset','prompt','encoder','decoder','evaluation'];
  const fmt = (num, n=3) => Number.isFinite(Number(num)) ? Number(num).toFixed(n) : '—';
  const escapeHtml = (v) => String(v).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
  async function json(url, init) {
    const r = await fetch(url, init);
    if (!r.ok) { const err = await r.json().catch(() => ({detail:r.statusText})); throw new Error(err.detail || `${url}: ${r.status}`); }
    return r.json();
  }
  let toastTimer;
  function toast(message, error=false) {
    const el = $('toast'); el.textContent = message; el.classList.toggle('error', error); el.classList.add('show');
    window.clearTimeout(toastTimer); toastTimer = window.setTimeout(() => el.classList.remove('show'), 4700);
  }
  function feedback(message, type='') { const el=$('runFeedback'); el.textContent=message; el.className=`run-feedback ${type}`; }
  function showStatus(v) {
    state.status=v;
    const ready=v.dataset_ready && v.checkpoint_ready && v.core_ready;
    $('sideStatusTitle').textContent=ready?'Ready for live inference':v.dataset_ready?'Dataset connected':'Dataset not connected';
    $('sideStatusDescription').textContent=ready?'SAM2 weights and COCO data detected. Your local lab is ready.':v.dataset_ready?'Connect the SAM2 checkpoint to enable live inference.':'Connect COCO images and annotations to enable interactive predictions.';
    $('sideDot').classList.toggle('ready',ready);
    $('dataBadge').classList.toggle('ready',v.dataset_ready);
    $('dataBadge').innerHTML=`<span class="badge-dot"></span>${ready?'Live inference available':v.dataset_ready?'COCO loaded · checkpoint '+(v.checkpoint_ready?'ready':'missing'):'COCO dataset not connected'}`;
    $('runButton').disabled=!state.current || !ready;
    $('savedButton').disabled=!state.current || !v.saved_masks_ready || !state.current.saved_available;
    for(const [prefix,valid,yes,no] of [['Coco',v.dataset_ready,'Detected','Missing'],['Weights',v.checkpoint_ready,'Detected','Missing'],['Saved',v.saved_masks_ready,'Available','Not found']]) {
      $(`status${prefix}Icon`).textContent=valid?'✓':'○'; $(`status${prefix}Icon`).classList.toggle('ok',valid);
      $(`status${prefix}Text`).textContent=valid?yes:no;
    }
    if(!state.current) feedback('Connect the COCO dataset to enable live predictions. Saved aggregate results remain available.');
    else if(!ready) feedback(v.saved_masks_ready?'Live model not ready. You can load saved test predictions.':'The SAM2 checkpoint is missing. Follow the setup instructions.');
    else feedback('Ready. Run the selected prompt against the real SAM2 model.');
  }
  function showResults(d) {
    state.results = d;
    const p=d.point||{}, b=d.box||{}, comparison=d.comparison?.paired||{};
    const gain=b.mean_iou-p.mean_iou;
    $('topObjects').textContent=Number(p.n_objects||6941).toLocaleString('en-US');
    $('topImages').textContent=Number(p.n_images||991).toLocaleString('en-US');
    $('topPoint').textContent=fmt(p.mean_iou,4);
    $('topBox').textContent=fmt(b.mean_iou,4);
    $('topGain').textContent='+'+fmt(gain,4);
    $('bigDelta').textContent='+'+fmt(gain,4);
    $('pointPercent').textContent=fmt(100*p.mean_iou,2)+'%';
    $('boxPercent').textContent=fmt(100*b.mean_iou,2)+'%';
    $('boxBetterShare').textContent=fmt(100*comparison.box_better_share,1)+'%';
    $('resultsSource').textContent=d.from_repository?'SOURCE / SAVED TEST FILES':'SOURCE / BUNDLED TEST SNAPSHOT';
    for (const [id,value] of [['iouPointBar',p.mean_iou],['iouBoxBar',b.mean_iou],['dicePointBar',p.mean_dice],['diceBoxBar',b.mean_dice]]) $(id).style.width=(100*value)+'%';
    $('iouValue').textContent=fmt(p.mean_iou,3)+' / '+fmt(b.mean_iou,3);
    $('diceValue').textContent=fmt(p.mean_dice,3)+' / '+fmt(b.mean_dice,3);
    $('runtimeEncoder').innerHTML=`~${Math.round((p.mean_encoder_ms_per_image+b.mean_encoder_ms_per_image)/2)}<span>ms</span>`;
    $('runtimePrompt').innerHTML=`~${Math.round((p.mean_prompt_ms_per_object+b.mean_prompt_ms_per_object)/2)}<span>ms</span>`;
    $('runtimeDevice').textContent=(d.point_info?.device||'mps:0').toUpperCase()+' / FP32';
    const thresholdData=d.thresholds||[];
    $('thresholdRows').innerHTML=thresholdData.map(v=>`<div class="threshold-row"><b>≥ ${escapeHtml(v.threshold)}</b><span>${fmt(v.point,1)}%</span><span>${fmt(v.box,1)}%</span><strong>+${fmt(v.box-v.point,1)} pp</strong></div>`).join('');
  }
  function renderGallery() {
    if (!state.examples.length) {
      $('sampleStrip').innerHTML='<div class="gallery-empty"><svg><use href="#i-info"/></svg> Add COCO val2017 to explore actual image instances.</div>';
      return;
    }
    $('sampleStrip').innerHTML=state.examples.map(item=>`<button class="sample-option ${state.current?.annotation_id===item.id?'selected':''}" type="button" data-annotation="${item.id}" title="${escapeHtml(item.title)}"><img class="sample-thumb" src="/api/image/${item.id}?thumb=true" alt="${escapeHtml(item.category)} thumbnail" loading="lazy"/><span class="sample-copy"><b>${escapeHtml(item.category)}</b><small>${escapeHtml(item.kind)}</small></span></button>`).join('');
    $('sampleStrip').querySelectorAll('[data-annotation]').forEach(el=>el.addEventListener('click',()=>selectExample(Number(el.dataset.annotation))));
  }
  function clearExamplePrediction() {
    state.manualPoint=null; state.manualBox=null; state.last={}; state.lastSource=null;
    ['pointOverlay','boxOverlay','stagePredMask'].forEach(id=>{ $(id).removeAttribute('src'); });
    $('pointCardScore').textContent='—'; $('boxCardScore').textContent='—';
    $('pointCardDetail').textContent='Run point inference to compare'; $('boxCardDetail').textContent='Run box inference to compare';
    $('livePointIoU').textContent='—'; $('liveBoxIoU').textContent='—'; $('livePointDice').textContent='Dice —'; $('liveBoxDice').textContent='Dice —';
    $('encoderLatency').textContent='—'; $('promptLatency').textContent='—'; $('metricOrigin').textContent='NO RUN';
    $('comparisonWarning').textContent='No predictions have been generated for this instance.';
    resetPipeline(); syncPromptReadout(); syncMarker();
  }
  async function selectExample(id, scroll=false) {
    if(state.currentRequest){ toast('Complete the current inference before switching samples.');return; }
    feedback('Loading COCO image and reference mask…','running');
    try {
      const item=await json(`/api/example/${id}`);
      state.current=item;clearExamplePrediction();
      $('sourceImage').src=item.image_url;
      for(const imageId of ['groundImage','pointImage','boxImage']) $(imageId).src=item.image_url;
      $('stageGtMask').src=item.ground_truth; $('groundOverlay').src=item.ground_truth;
      $('stagePlaceholder').hidden=true;
      $('imageStage').classList.add('interactive');
      $('currentImageTitle').textContent=`${item.category.charAt(0).toUpperCase()+item.category.slice(1)}`;
      $('currentImageInfo').textContent=`${item.width} × ${item.height} px · annotation #${id}`;
      $('stageStateBadge').textContent='COCO IMAGE · READY FOR INFERENCE';
      $('stageStateBadge').className='stage-state-badge live';
      $('interactionHint').textContent='Choose research baseline or draw a manual prompt';
      $('runButton').disabled=!(state.status?.checkpoint_ready&&state.status?.core_ready);
      $('savedButton').disabled=!(state.status?.saved_masks_ready&&state.current.saved_available);
      $('comparisonWarning').textContent='Select a method to generate a real segmentation.';
      feedback('Selected image loaded. Choose Point, Box, or Both and run inference.');
      renderGallery();syncPromptReadout();syncMarker();syncMaskOpacity();
      if(scroll) $('playground').scrollIntoView({behavior:'smooth',block:'start'});
    }catch(e){feedback(e.message,'error');toast(e.message,true);}
  }
  function setMode(mode){
    state.mode=mode;
    document.querySelectorAll('.mode-option').forEach(el=>el.classList.toggle('active',el.dataset.mode===mode));
    $('imageStage').classList.toggle('mode-point',mode==='point');
    $('imageStage').classList.toggle('mode-box',mode==='box');
    $('runButtonText').textContent=mode==='compare'?'Compare both prompts':mode==='point'?'Run point inference':'Run box inference';
    syncPromptReadout();syncMarker();renderCurrentMask();
  }
  function setSource(source){state.source=source;syncPromptReadout();syncMarker();}
  function syncPromptReadout(){
    const cur=state.current;
    if(!cur){ $('promptReadout').textContent='No image selected';$('promptReadoutSub').textContent='Connect the COCO dataset to choose a sample';return; }
    const manual=state.source==='manual';
    let text='',sub='';
    if(state.mode==='point'){
      const p=manual?state.manualPoint:cur.point;
      text=p?`Point · (${Math.round(p[0])}, ${Math.round(p[1])})`:'Click inside the image';
      sub=manual?'Manual positive-point prompt':'Baseline point from GT distance transform';
    }else if(state.mode==='box'){
      const b=manual?state.manualBox:cur.box;
      text=b?`Box · ${b.map(x=>Math.round(x)).join(', ')}`:'Drag a bounding box on the image';
      sub=manual?'Manual rectangular prompt':'Oracle box from COCO annotation';
    }else{
      text=manual?'Manual point + manual box':'Center point + oracle box';
      sub=manual?'Select a point and box in their respective tabs first':'Same SAM2 model and exact same COCO object';
    }
    $('promptReadout').textContent=text;$('promptReadoutSub').textContent=sub;
  }
  function imageGeometry() {
    if(!state.current)return null;
    const stage=$('imageStage');const {width,height}=state.current;const size=Math.min(stage.clientWidth/width,stage.clientHeight/height);
    const displayW=width*size,displayH=height*size;
    return {scale:size,width:stage.clientWidth,height:stage.clientHeight,offsetX:(stage.clientWidth-displayW)/2,offsetY:(stage.clientHeight-displayH)/2};
  }
  function pixelAtEvent(ev){
    const g=imageGeometry();if(!g)return null;
    const r=$('imageStage').getBoundingClientRect();let x=(ev.clientX-r.left-g.offsetX)/g.scale,y=(ev.clientY-r.top-g.offsetY)/g.scale;
    if(x<0||y<0||x>=state.current.width||y>=state.current.height)return null;
    return [Math.round(x),Math.round(y)];
  }
  function screenRectFromBox(b){const g=imageGeometry();if(!g)return null;return {left:g.offsetX+b[0]*g.scale,top:g.offsetY+b[1]*g.scale,width:(b[2]-b[0])*g.scale,height:(b[3]-b[1])*g.scale};}
  function syncMarker(){
    const cur=state.current;const pointEl=$('pointMarker'),boxEl=$('boxMarker');
    pointEl.hidden=true;boxEl.hidden=true;
    if(!cur)return;
    const g=imageGeometry();if(!g)return;
    const point=state.source==='manual'?state.manualPoint:cur.point,box=state.source==='manual'?state.manualBox:cur.box;
    if(point&&(state.mode==='point'||state.mode==='compare')){
      pointEl.style.left=(g.offsetX+point[0]*g.scale)+'px';pointEl.style.top=(g.offsetY+point[1]*g.scale)+'px';pointEl.hidden=false;
    }
    if(box&&(state.mode==='box'||state.mode==='compare')){
      const r=screenRectFromBox(box);if(r){boxEl.style.left=r.left+'px';boxEl.style.top=r.top+'px';boxEl.style.width=r.width+'px';boxEl.style.height=r.height+'px';boxEl.hidden=false;}
    }
  }
  function onStagePointerDown(ev){
    if(!state.current||state.source!=='manual'||state.currentRequest)return;
    const p=pixelAtEvent(ev);if(!p)return;
    if(state.mode==='point'){state.manualPoint=p;syncPromptReadout();syncMarker();toast('Point selected. Run inference to generate a mask.');}
    else if(state.mode==='box') {state.boxDrag=p;state.manualBox=null;$('imageStage').setPointerCapture(ev.pointerId);syncMarker();}
  }
  function onStagePointerMove(ev){
    if(!state.boxDrag||state.mode!=='box')return;
    const g=imageGeometry();if(!g)return;
    const r=$('imageStage').getBoundingClientRect();
    const x=Math.max(0,Math.min(state.current.width,(ev.clientX-r.left-g.offsetX)/g.scale));
    const y=Math.max(0,Math.min(state.current.height,(ev.clientY-r.top-g.offsetY)/g.scale));
    const [x1,y1]=state.boxDrag;
    state.manualBox=[Math.min(x1,x),Math.min(y1,y),Math.max(x1,x),Math.max(y1,y)];syncMarker();
  }
  function onStagePointerUp(ev){
    if(!state.boxDrag)return;
    onStagePointerMove(ev);state.boxDrag=null;
    if(state.manualBox && (state.manualBox[2]-state.manualBox[0]<3||state.manualBox[3]-state.manualBox[1]<3))state.manualBox=null;
    syncPromptReadout();syncMarker();if(state.manualBox)toast('Bounding box selected. Run inference to generate a mask.');
  }
  function resetPipeline(){document.querySelectorAll('.pipeline-step').forEach(el=>el.classList.remove('active','completed'));$('pipelineStatus').textContent='Waiting for input';}
  function pipeline(stage,message){
    const idx=milestones.indexOf(stage);
    if(stage==='saved'){
      document.querySelectorAll('.pipeline-step').forEach((el,i)=>el.classList.toggle('completed',i<2));
      $('pipelineStatus').textContent=message;return;
    }
    if(idx<0){$('pipelineStatus').textContent=message;return;}
    document.querySelectorAll('.pipeline-step').forEach((el,i)=>{el.classList.toggle('completed',i<idx);el.classList.toggle('active',i===idx);});
    $('pipelineStatus').textContent=message;
  }
  function pipelineComplete(){document.querySelectorAll('.pipeline-step').forEach(el=>{el.classList.remove('active');el.classList.add('completed');});$('pipelineStatus').textContent='Pipeline complete · real evaluation';}
  function syncMaskOpacity(){ const opacity=state.opacity/100;['stagePredMask','pointOverlay','boxOverlay'].forEach(id=>$(id).style.opacity=opacity);$('stageGtMask').style.opacity=state.showGt?opacity:0;$('groundOverlay').style.opacity=opacity;$('opacityValue').textContent=state.opacity+'%'; }
  function renderCurrentMask(){
    const pref=state.mode==='point'?'point':state.mode==='box'?'box':state.last.box?'box':'point';
    const mask=state.last[pref]?.mask;
    if(mask)$('stagePredMask').src=mask;else $('stagePredMask').removeAttribute('src');
    if(state.lastSource){$('stageStateBadge').textContent=state.lastSource==='saved'?'SAVED PREDICTION · NOT LIVE INFERENCE':'LIVE SAM2 INFERENCE';$('stageStateBadge').className='stage-state-badge '+(state.lastSource==='saved'?'saved':'live');}
    syncMaskOpacity();
  }
  function fillPredictions(payload) {
    const r=payload.results||{};state.last={...state.last,...r};state.lastSource=payload.source;
    $('metricOrigin').textContent=payload.source==='saved'?'SAVED TEST':'LIVE SAM2';
    $('comparisonWarning').textContent=payload.source==='saved'?'Saved test-set predictions. No live model inference was performed.':'Masks generated by the actual SAM2 pipeline, evaluated against COCO ground truth.';
    for(const method of ['point','box']){
      const v=state.last[method];if(!v)continue;
      $(`${method}Overlay`).src=v.mask;
      $(`${method}CardScore`).textContent='IoU '+fmt(v.iou,3);
      $(`${method}CardDetail`).textContent=`Dice ${fmt(v.dice,3)} · score ${fmt(v.score,3)}`;
      $(`live${method[0].toUpperCase()+method.slice(1)}IoU`).textContent=fmt(v.iou,3);
      $(`live${method[0].toUpperCase()+method.slice(1)}Dice`).textContent='Dice '+fmt(v.dice,3);
    }
    $('encoderLatency').textContent=payload.encoder_ms==null?'saved test':payload.encoder_cache_hit?'cached image':fmt(payload.encoder_ms,1)+' ms';
    const timings=Object.values(r).map(v=>v.prompt_ms).filter(Number.isFinite);
    $('promptLatency').textContent=timings.length?timings.map(v=>fmt(v,1)).join(' / ')+' ms':'—';
    renderCurrentMask();pipelineComplete();
    feedback(payload.source==='saved'?'Saved research masks loaded. These are precomputed, not live inference.':`SAM2 finished inference on ${escapeHtml(payload.device)}. IoU and Dice are computed from the masks.`,'success');
    toast(payload.source==='saved'?'Saved test predictions loaded':'Live SAM2 prediction complete');
  }
  async function streamInference(useSaved=false){
    if(state.currentRequest||!state.current){toast('Select a COCO image first.',true);return;}
    if(state.source==='manual'&&!useSaved){
      if((state.mode==='point'||state.mode==='compare')&&!state.manualPoint){toast('Click a point in the image first.',true);return;}
      if((state.mode==='box'||state.mode==='compare')&&!state.manualBox){toast('Draw a bounding box first.',true);return;}
    }
    if(useSaved&&state.source!=='automatic'){toast('Saved results use the original automatic baseline prompts. Select Research baseline.',true);return;}
    state.currentRequest=true;$('runButton').disabled=true;$('savedButton').disabled=true;resetPipeline();
    feedback(useSaved?'Loading saved test masks…':'Running actual SAM2 inference. First initialization may take longer…','running');
    $('runButtonText').textContent='Processing…';
    try{
      const res=await fetch('/api/infer-stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({annotation_id:state.current.annotation_id,mode:useSaved?'compare':state.mode,source:state.source,point:state.manualPoint,box:state.manualBox,use_saved:useSaved})});
      if(!res.ok)throw new Error(`Server responded with ${res.status}`);
      if(!res.body)throw new Error('This browser does not support streamed inference responses.');
      const reader=res.body.getReader(),decoder=new TextDecoder();let buffer='',gotResult=false;
      while(true){const {value,done}=await reader.read();if(done)break;buffer+=decoder.decode(value,{stream:true});const chunks=buffer.split('\n\n');buffer=chunks.pop()||'';
        for(const chunk of chunks){const line=chunk.split('\n').find(x=>x.startsWith('data: '));if(!line)continue;
          const msg=JSON.parse(line.slice(6));if(msg.type==='stage'){pipeline(msg.stage,msg.message);feedback(msg.message,'running');}
          else if(msg.type==='result'){fillPredictions(msg);gotResult=true;}
          else if(msg.type==='error')throw new Error(msg.message||'Inference failed');
        }
      }
      if(!gotResult)throw new Error('No model prediction was returned.');
    }catch(err){feedback(err.message,'error');toast(err.message,true);$('pipelineStatus').textContent='Pipeline stopped: '+err.message;}
    finally{state.currentRequest=false;$('runButton').disabled=!(state.current && state.status?.checkpoint_ready && state.status?.core_ready);$('savedButton').disabled=!(state.current&&state.status?.saved_masks_ready&&state.current.saved_available);setMode(state.mode);}
  }
  function renderFailures(){
    $('failureGrid').innerHTML=failureData.map((v,i)=>`<button class="failure-card" data-failure="${v.id}" aria-label="Open ${escapeHtml(v.category)} failure case"><div class="failure-visual">${state.status?.dataset_ready?`<img src="/api/image/${v.id}?thumb=true" alt="COCO ${escapeHtml(v.category)} image" loading="lazy"/>`:''}<span class="failure-num">CASE 0${i+1}</span>${!state.status?.dataset_ready?`<span class="fallback-figure-label">ANNOTATION #${v.id}</span>`:''}</div><div class="failure-text"><small>${escapeHtml(v.category.toUpperCase())} · ${escapeHtml(v.label.toUpperCase())}</small><strong>${escapeHtml(v.label)}</strong><p>${escapeHtml(v.explanation)}</p><div class="failure-metrics"><span>POINT <b class="point">${fmt(v.point_iou,2)}</b></span><span>BOX <b class="box">${fmt(v.box_iou,2)}</b></span></div></div></button>`).join('');
    $('failureGrid').querySelectorAll('[data-failure]').forEach(el=>el.addEventListener('click',async()=>{
      if(!state.status?.dataset_ready){toast('The COCO dataset must be connected to inspect the original image. The results shown are from the saved research report.');return;}
      await selectExample(Number(el.dataset.failure),true);setMode('compare');setSource('automatic');document.querySelector('input[name="promptSource"][value="automatic"]').checked=true;
    }));
    if(!state.status?.saved_results_ready){$('failureFigureLink').style.opacity='.45';$('failureFigureLink').addEventListener('click',ev=>{ev.preventDefault();toast('The original failure-analysis figure becomes available when demo/ is inside the research repository.');});}
  }
  function resetSelection(){if(!state.current)return;clearExamplePrediction();$('stageStateBadge').textContent='COCO IMAGE · READY FOR INFERENCE';$('stageStateBadge').className='stage-state-badge live';feedback('Prompt selection reset.');}
  function presentationView(){const i=state.presentationStep;
    const section=i<=2?'playground':i===3?'research':'failures';
    document.querySelectorAll('.section-block').forEach(el=>el.classList.toggle('presentation-focus',el.id===section));
    $('presentationStep').textContent=['1 / 5 — Select a COCO image','2 / 5 — Run point inference','3 / 5 — Run box inference','4 / 5 — Compare aggregate results','5 / 5 — Explain one failure'][i];
    if(i===1)setMode('point');if(i===2)setMode('box');
    document.getElementById(section).scrollIntoView({behavior:'smooth',block:'start'});
  }
  function togglePresentation(flag){document.body.classList.toggle('presentation-mode',flag);$('presentationHint').hidden=!flag;if(flag){state.presentationStep=0;presentationView();}else{document.querySelectorAll('.section-block').forEach(el=>el.classList.remove('presentation-focus'));$('playground').scrollIntoView({behavior:'smooth'});}}
  function bind(){
    document.querySelectorAll('.mode-option').forEach(el=>el.addEventListener('click',()=>setMode(el.dataset.mode)));
    document.querySelectorAll('input[name="promptSource"]').forEach(el=>el.addEventListener('change',()=>setSource(el.value)));
    $('runButton').addEventListener('click',()=>streamInference(false));$('savedButton').addEventListener('click',()=>streamInference(true));
    $('resetButton').addEventListener('click',resetSelection);
    $('imageStage').addEventListener('pointerdown',onStagePointerDown);$('imageStage').addEventListener('pointermove',onStagePointerMove);$('imageStage').addEventListener('pointerup',onStagePointerUp);$('imageStage').addEventListener('pointercancel',onStagePointerUp);
    $('opacitySlider').addEventListener('input',ev=>{state.opacity=Number(ev.target.value);syncMaskOpacity();});
    $('showGtToggle').addEventListener('change',ev=>{state.showGt=ev.target.checked;syncMaskOpacity();});
    $('presentationBtn').addEventListener('click',()=>togglePresentation(true));$('exitPresentation').addEventListener('click',()=>togglePresentation(false));
    $('nextStep').addEventListener('click',()=>{state.presentationStep=Math.min(4,state.presentationStep+1);presentationView();});
    $('prevStep').addEventListener('click',()=>{state.presentationStep=Math.max(0,state.presentationStep-1);presentationView();});
    document.addEventListener('keydown',ev=>{if(ev.key==='Escape'&&document.body.classList.contains('presentation-mode'))togglePresentation(false);});
    $('copySetup').addEventListener('click',async()=>{try{await navigator.clipboard.writeText($('setupCode').textContent);toast('Setup commands copied.');}catch{toast('Select the terminal commands and copy them manually.');}});
    $('mobileMenu').addEventListener('click',()=>$('sidebar').classList.toggle('open'));
    document.querySelectorAll('.sidebar-nav a').forEach(el=>el.addEventListener('click',()=>$('sidebar').classList.remove('open')));
    window.addEventListener('resize',()=>{if(state.current)syncMarker();});
    const observer=new IntersectionObserver(entries=>{const matches=entries.filter(e=>e.isIntersecting);if(!matches.length)return;const id=matches[0].target.id;document.querySelectorAll('.nav-link').forEach(a=>a.classList.toggle('active',a.getAttribute('href')==='#'+id));},{rootMargin:'-10% 0px -75% 0px',threshold:0});
    for(const id of ['top','playground','research','failures','method'])observer.observe($(id));
  }
  async function init(){
    bind();setMode('point');syncMaskOpacity();renderFailures();
    const [status,results,examples]=await Promise.allSettled([json('/api/status'),json('/api/results'),json('/api/examples')]);
    if(status.status==='fulfilled'){showStatus(status.value);}else{feedback('Unable to connect to the local backend. Start: python -m demo.app','error');toast('Local backend could not be reached.',true);}
    if(results.status==='fulfilled')showResults(results.value);
    if(examples.status==='fulfilled'){state.examples=examples.value.items||[];renderGallery();if(state.examples.length) await selectExample(state.examples[0].id);}
    else renderGallery();renderFailures();
  }
  document.addEventListener('DOMContentLoaded',init);
})();
