import React,{useEffect,useMemo,useState} from 'react';
import {createRoot} from 'react-dom/client';
import {Activity,AlertTriangle,ArrowRight,AudioLines,Check,ChevronRight,Clock3,FileImage,History,Info,Lightbulb,LayoutDashboard,LockKeyhole,LogOut,Mail,Mic,RefreshCcw,ShieldCheck,Sparkles,Stethoscope,UploadCloud,UserRound,X,PanelLeftClose,PanelLeftOpen} from 'lucide-react';
import './styles.css';
import './insights.css';

const API=import.meta.env.VITE_API_URL||'';
const AUDIO_TYPES=['audio/wav','audio/x-wav','audio/wave'];
class ApiError extends Error{constructor(m,status){super(m);this.status=status}}
async function api(path,{method='GET',json,form}={}){
 const opt={method,credentials:'include',headers:{}};
 if(json!==undefined){opt.headers['Content-Type']='application/json';opt.body=JSON.stringify(json)}
 if(form)opt.body=form;
 let res;try{res=await fetch(API+path,opt)}catch{throw new ApiError('Cannot reach the backend. Is it running on port 8000?',0)}
 if(!res.ok){let d='Request failed.';try{const j=await res.json();d=typeof j.detail==='string'?j.detail:Array.isArray(j.detail)?j.detail.map(x=>String(x.msg).replace(/^Value error, /,'')).join(' '):d}catch{}throw new ApiError(d,res.status)}
 return res.json()}
const postFile=(path,file)=>{const fd=new FormData();fd.append('file',file);return api(path,{method:'POST',form:fd})};
const fromServer=(a)=>({id:a.id,date:new Date(a.created_at).toLocaleString(),audio:a.audio,xray:a.xray});
const initials=(n='')=>n.trim().split(/\s+/).slice(0,2).map(w=>w[0]).join('').toUpperCase()||'U';
const IMAGE_TYPES=['image/png','image/jpeg','image/jpg','image/webp'];

function GoogleIcon(){return <svg width="18" height="18" viewBox="0 0 48 48" aria-hidden="true"><path fill="#EA4335" d="M24 9.5c3.5 0 6.6 1.2 9.1 3.6l6.8-6.8C35.8 2.4 30.3 0 24 0 14.6 0 6.5 5.4 2.6 13.2l7.9 6.1C12.4 13.6 17.7 9.5 24 9.5z"/><path fill="#4285F4" d="M46.5 24.5c0-1.6-.1-3.1-.4-4.5H24v9h12.7c-.6 3-2.3 5.5-4.8 7.2l7.7 6c4.5-4.2 6.9-10.3 6.9-17.7z"/><path fill="#FBBC05" d="M10.5 28.7c-.5-1.4-.8-3-.8-4.7s.3-3.2.8-4.7l-7.9-6.1C.9 16.4 0 20.1 0 24s.9 7.6 2.6 10.8l7.9-6.1z"/><path fill="#34A853" d="M24 48c6.5 0 11.9-2.1 15.9-5.8l-7.7-6c-2.1 1.4-4.9 2.3-8.2 2.3-6.3 0-11.6-4.1-13.5-9.8l-7.9 6.1C6.5 42.6 14.6 48 24 48z"/></svg>}

function Field({icon,label,children}){return <label className="field"><span>{icon}{label}</span>{children}</label>}

function AuthPage({providers,onAuth}){
 const [mode,setMode]=useState('login'); const [form,setForm]=useState({name:'',email:'',password:''}); const [busy,setBusy]=useState(false);
 const [err,setErr]=useState(()=>new URLSearchParams(window.location.search).get('auth_error')||'');
 useEffect(()=>{if(window.location.search)window.history.replaceState({},'',window.location.pathname)},[]);
 const set=(k)=>(e)=>setForm(f=>({...f,[k]:e.target.value}));
 const submit=async(e)=>{e.preventDefault();setErr('');setBusy(true);
  try{const u=await api(mode==='login'?'/api/auth/login':'/api/auth/register',{method:'POST',json:mode==='login'?{email:form.email,password:form.password}:form});onAuth(u)}
  catch(x){setErr(x.message)}finally{setBusy(false)}};
 const login=mode==='login';
 return <div className="authPage"><div className="authModal">
  <div className="authLogo"><Activity size={21}/></div>
  <div className="eyebrow">RESPIRA ACCOUNT</div>
  <h2>{login?'Welcome back.':'Create your account.'}</h2>
  <p>{login?'Sign in to run assessments and see your saved history.':'Register to save your assessments securely.'}</p>
  {providers.google&&<><a className="googleBtn" href={API+'/api/auth/google/login'}><GoogleIcon/>Continue with Google</a><div className="authDivider"><span>or</span></div></>}
  <form onSubmit={submit}>
   {!login&&<Field icon={<UserRound size={16}/>} label="Full name"><input type="text" value={form.name} onChange={set('name')} placeholder="Dr. Jane Doe" autoComplete="name" required maxLength={80}/></Field>}
   <Field icon={<Mail size={16}/>} label="Email"><input type="email" value={form.email} onChange={set('email')} placeholder="you@example.com" autoComplete="email" required/></Field>
   <Field icon={<LockKeyhole size={16}/>} label="Password"><input type="password" value={form.password} onChange={set('password')} placeholder={login?'Your password':'At least 8 characters'} autoComplete={login?'current-password':'new-password'} required minLength={login?1:8} maxLength={72}/></Field>
   {err&&<div className="authError" role="alert">{err}</div>}
   <button className="predictBtn full" disabled={busy}>{busy?<><span className="spinner"/>Please wait…</>:<>{login?'Sign in':'Create account'} <ArrowRight size={17}/></>}</button>
  </form>
  <button type="button" className="switchAuth" onClick={()=>{setMode(login?'register':'login');setErr('')}}>{login?"Don't have an account? Create one":'Already have an account? Sign in'}</button>
  <div className="authNote"><Info size={13}/> Model output is supportive, not a medical diagnosis.</div>
 </div></div>}

function App(){
 const [user,setUser]=useState(undefined); const [providers,setProviders]=useState({google:false});
 useEffect(()=>{api('/api/auth/providers').then(setProviders).catch(()=>{});api('/api/auth/me').then(setUser).catch(()=>setUser(null))},[]);
 if(user===undefined)return <div className="splash"><span className="spinner dark"/></div>;
 if(!user)return <AuthPage providers={providers} onAuth={setUser}/>;
 return <Workspace user={user} onLogout={()=>setUser(null)}/>}

function Workspace({user,onLogout}){
 const [page,setPage]=useState('assessment'); const [audio,setAudio]=useState(null); const [xray,setXray]=useState(null); const [result,setResult]=useState(null); const [busy,setBusy]=useState(false); const [toast,setToast]=useState(''); const [history,setHistory]=useState([]); const [sidebar,setSidebar]=useState(()=>window.innerWidth>900);
 useEffect(()=>{api('/api/assessments').then(l=>setHistory(l.map(fromServer))).catch(()=>{})},[]);
 const logout=async()=>{await api('/api/auth/logout',{method:'POST'}).catch(()=>{});onLogout()};
 const notify=(m)=>{setToast(m);setTimeout(()=>setToast(''),2600)};
 const remove=(type)=>type==='audio'?setAudio(null):setXray(null);
 const predict=async()=>{if(!audio&&!xray){notify('Upload an audio file, an X-ray, or both first.');return} setBusy(true);setResult(null);
  try{const [a,x]=await Promise.all([audio?postFile('/api/predict/audio',audio):null,xray?postFile('/api/predict/xray',xray):null]);
   const saved=await api('/api/assessments',{method:'POST',json:{audio:a,xray:x}}).catch(()=>null);
   const r={id:saved?.id||Date.now(),date:saved?new Date(saved.created_at).toLocaleString():new Date().toLocaleString(),audio:a,xray:x?{...x,preview:URL.createObjectURL(xray)}:null};
   setResult(r);if(saved)setHistory(h=>[fromServer(saved),...h]);else notify('Result shown, but it could not be saved to your history.');
   setTimeout(()=>document.getElementById('results')?.scrollIntoView({behavior:'smooth'}),80)}
  catch(e){if(e.status===401){notify('Your session expired. Please sign in again.');onLogout()}else notify(e.message||'Prediction failed.')}
  finally{setBusy(false)}};
 const reset=()=>{setAudio(null);setXray(null);setResult(null);setPage('assessment');window.scrollTo({top:0,behavior:'smooth'})};
 return <div className="app">
  <aside className={'sidebar '+(sidebar?'open':'closed')}>
   <div className="brand"><div className="brandMark"><Activity size={21}/></div>{sidebar&&<div><strong>respira</strong><span>respiratory intelligence</span></div>}</div>
   <nav><button className={page==='assessment'?'active':''} onClick={()=>setPage('assessment')}><LayoutDashboard size={19}/>{sidebar&&'Assessment'}</button><button className={page==='history'?'active':''} onClick={()=>setPage('history')}><History size={19}/>{sidebar&&'Assessment history'}</button></nav>
   {sidebar&&<div className="sideBottom"><div className="trust"><ShieldCheck size={18}/><div><b>Your results are safe with us</b></div></div><div className="profile"><div className="avatar">{initials(user.name)}</div><div className="profileText"><b>{user.name}</b><span>{user.email}</span></div><button aria-label="Sign out" title="Sign out" onClick={logout}><LogOut size={16}/></button></div></div>}
  </aside>
  <main className={'main '+(sidebar?'':'wide')}>
   <header><button className="iconBtn" onClick={()=>setSidebar(v=>!v)}>{sidebar?<PanelLeftClose size={20}/>:<PanelLeftOpen size={20}/>}</button><div className="headerTitle"><span>{page==='history'?'Assessment history':'Respiratory assessment'}</span><small>AI-assisted screening workspace</small></div><div className="headerRight"><i/><div className="miniAvatar" title={user.email}>{initials(user.name)}</div></div></header>
   {page==='assessment'?<Assessment audio={audio} xray={xray} setAudio={setAudio} setXray={setXray} remove={remove} predict={predict} busy={busy} result={result} reset={reset}/>:<HistoryPage history={history} onOpen={(r)=>{setResult(r);setPage('assessment');setTimeout(()=>document.getElementById('results')?.scrollIntoView({behavior:'smooth'}),100)}}/>}
   <footer><span>R</span><span><Info size={14}/> Model output is supportive, not a medical diagnosis.</span></footer>
  </main>
  {toast&&<div className="toast"><Check size={17}/>{toast}</div>}
 </div>
}

function Assessment({audio,xray,setAudio,setXray,remove,predict,busy,result,reset}){
 return <div className="content">
  <section className="hero"><div><div className="eyebrow"><Stethoscope size={15}/> RESPIRATORY SCREENING</div><h3 className="project">AI-Assisted Multimodal Respiratory Disease Screening System Using Chest Sounds and X-Ray Analysis</h3><p>Upload a chest sound recording, a chest X-ray, or both. Respira organizes the model output into a clear clinical view.</p></div><div className="heroArt"><div className="pulseRing"><Activity size={44}/></div><span>Sound + imaging</span></div></section>
  <section className="uploadSection"><div className="sectionHeading"><div><h2>Start an assessment</h2><p>Choose one input or combine both for a broader screening view.</p></div><span className="inputCount">{[audio,xray].filter(Boolean).length}/2 inputs</span></div>
   <div className="uploadGrid">
    <UploadCard type="audio" file={audio} onFile={setAudio} onRemove={()=>remove('audio')}/>
    <UploadCard type="xray" file={xray} onFile={setXray} onRemove={()=>remove('xray')}/>
   </div>
   <div className="predictRow"><div className="ready"><span className={audio||xray?'readyDot on':'readyDot'}/>{audio||xray?<><b>Ready to analyze</b><span> {audio&&xray?'Both inputs selected':audio?'WAV audio selected':'X-ray selected'}</span></>:<span>Add at least one input</span>}</div><button className="predictBtn" disabled={busy||(!audio&&!xray)} onClick={predict}>{busy?<><span className="spinner"/>Analyzing…</>:<>Predict results <ArrowRight size={18}/></>}</button></div>
  </section>
  {result&&<Results result={result} reset={reset}/>} 
  {!result&&<HowItWorks/>}
 </div>
}

function UploadCard({type,file,onFile,onRemove}){
 const audioType=type==='audio'; const [drag,setDrag]=useState(false); const inputId='upload-'+type;
 const accept=audioType?'.wav':'.png,.jpg,.jpeg,.webp';
 const validate=(f)=>{if(!f)return;if(audioType&&!AUDIO_TYPES.includes(f.type)&&!f.name.match(/\.wav$/i)){alert('Audio must be in WAV format only.');return}if(!audioType&&!IMAGE_TYPES.includes(f.type)&&!f.name.match(/\.(png|jpe?g|webp)$/i)){alert('Please choose a PNG, JPG, JPEG or WEBP image.');return}if(f.size>25*1024*1024){alert('File must be 25 MB or smaller.');return}onFile(f)};
 return <div className={'uploadCard '+(file?'hasFile':'')+(drag?' dragging':'')} onDragOver={e=>{e.preventDefault();setDrag(true)}} onDragLeave={()=>setDrag(false)} onDrop={e=>{e.preventDefault();setDrag(false);validate(e.dataTransfer.files[0])}}>
  <div className="cardTop"><div className={'uploadIcon '+(audioType?'sound':'scan')}>{audioType?<Mic size={22}/>:<FileImage size={22}/>}</div><div><span className="cardLabel">{audioType?'CHEST SOUND':'CHEST X-RAY'}</span><h3>{audioType?'Respiratory audio':'Chest radiograph'}</h3></div>{file&&<div className="checkCircle"><Check size={15}/></div>}</div>
  {!file?<><div className="dropArea"><UploadCloud size={28}/><b>Drop your {audioType?'WAV recording':'X-ray'} here</b><span>or <label htmlFor={inputId}>browse from device</label></span><small>{audioType?'WAV only':'PNG, JPG or WEBP'} · up to 25 MB</small><input id={inputId} type="file" accept={accept} hidden onChange={e=>validate(e.target.files[0])}/></div><div className="cardHint">{audioType?<><AudioLines size={16}/> WAV format only for the respiratory sound model.</>:<><span className="scanTiny">+</span> Use a frontal chest X-ray where possible.</>}</div></>:<FilePreview file={file} audio={audioType} onRemove={onRemove}/>} 
 </div>
}
function FilePreview({file,audio,onRemove}){
 const url=useMemo(()=>URL.createObjectURL(file),[file]);
 useEffect(()=>()=>URL.revokeObjectURL(url),[url]);
 return <div className="filePreview">{audio?<div className="audioPreview"><audio controls src={url}/><div className="wave">{Array.from({length:34},(_,i)=><i key={i} style={{height:(10+(Math.sin(i*1.8)+1)*9)+'px'}}/> )}</div></div>:<div className="imagePreview"><img src={url} alt="Uploaded chest X-ray preview"/></div>}<div className="fileMeta"><div><b>{file.name}</b><span>{formatBytes(file.size)} · {audio?'WAV audio':'X-ray image'}</span></div><button onClick={onRemove} aria-label="Remove file"><X size={17}/></button></div></div>}
function HowItWorks(){return <section className="how"><div className="sectionHeading"><div><h2>Designed around the clinical workflow</h2><p>Simple inputs, transparent outputs, no unnecessary complexity.</p></div></div><div className="howGrid"><div><span>01</span><h3>Upload</h3><p>Add the WAV recording, X-ray, or both from the assessment screen.</p></div><div><span>02</span><h3>Predict</h3><p>Your connected ML service will process each selected modality.</p></div><div><span>03</span><h3>Review</h3><p>See the predicted finding and confidence in one structured view.</p></div></div></section>}
function Results({result,reset}){const suggestions=getSuggestions(result);return <section className="results" id="results"><div className="resultHeader"><div><div className="eyebrow"><Check size={15}/> ANALYSIS COMPLETE</div><h2>Assessment results</h2><p>{result.date} · Model predictions from selected inputs</p></div><button className="secondaryBtn" onClick={reset}><RefreshCcw size={16}/> New assessment</button></div><div className="resultGrid">{result.audio&&<ResultCard title="Chest sound" icon={<AudioLines size={20}/>} label="Respiratory finding" value={result.audio.label} confidence={result.audio.confidence} extra={result.audio.extra} probabilities={result.audio.probabilities}/>} {result.xray&&<ResultCard title="Chest X-ray" icon={<FileImage size={20}/>} label="Predicted finding" value={result.xray.label} confidence={result.xray.confidence} extra={result.xray.extra} image={result.xray.preview} probabilities={result.xray.probabilities}/>}</div><AiInsights result={result}/><Suggestions suggestions={suggestions}/><div className="clinicalNote"><div className="noteIcon"><Info size={18}/></div><div><b>Clinical interpretation</b><p>This interface reports model predictions for screening support. A qualified healthcare professional should review the original clinical data before making a diagnosis or treatment decision.</p></div></div></section>}
const insightsCache=new Map();
const insightPayload=(p)=>p?{predicted_class:p.predicted_class,confidence:p.confidence,probabilities:p.probabilities||{}}:null;
const SOURCE_LABEL={audio:'Chest sound',xray:'Chest X-ray'};
const AGREEMENT={both_abnormal:'Both inputs show an abnormal pattern',both_normal:'Both inputs show a normal pattern',differ:'The two inputs differ'};

function SourceBadge({source}){return <span className={'sourceBadge '+source}>{source==='audio'?<AudioLines size={12}/>:<FileImage size={12}/>}{SOURCE_LABEL[source]}</span>}

function AiInsights({result}){
 const key=String(result.id);
 const [state,setState]=useState(()=>insightsCache.has(key)?{status:'ok',data:insightsCache.get(key)}:{status:'loading'});
 const [attempt,setAttempt]=useState(0);
 useEffect(()=>{
  if(insightsCache.has(key)){setState({status:'ok',data:insightsCache.get(key)});return}
  let cancelled=false;setState({status:'loading'});
  api('/api/insights',{method:'POST',json:{audio:insightPayload(result.audio),xray:insightPayload(result.xray)}})
   .then(d=>{insightsCache.set(key,d);if(!cancelled)setState({status:'ok',data:d})})
   .catch(e=>{if(!cancelled)setState({status:'error',message:e.status===401?'Your session expired. Please sign in again.':(e.message||'AI insights are unavailable right now.')})});
  return()=>{cancelled=true};
 },[key,attempt]);
 const both=!!(result.audio&&result.xray);
 return <section className="insights" aria-live="polite">
  <div className="insightsHeading"><div className="insightsIcon"><Sparkles size={19}/></div><div><div className="eyebrow">AI HEALTH INSIGHTS</div><h2>Understanding your results</h2><p>A plain-language explanation of the model predictions above, generated by AI. It is educational and does not change the predictions.</p></div><span className="aiTag">AI-GENERATED</span></div>
  {state.status==='loading'&&<InsightsSkeleton both={both}/>}
  {state.status==='error'&&<div className="insightError" role="alert"><AlertTriangle size={20}/><div><b>AI insights are unavailable right now.</b><p>{state.message} Your model results above are not affected.</p></div><button className="secondaryBtn" onClick={()=>setAttempt(a=>a+1)}><RefreshCcw size={14}/> Try again</button></div>}
  {state.status==='ok'&&<InsightsBody data={state.data} both={both}/>}
  <InsightsDisclaimer/>
 </section>}

function InsightsSkeleton({both}){return <div className="skel" aria-busy="true"><div className="skelNote"><span className="spinner dark" style={{width:16,height:16}}/> Preparing your explanation…</div><i className="block"/><div className="row2"><i/><i/></div>{both&&<i className="block"/>}<i style={{width:'70%'}}/><i style={{width:'55%'}}/></div>}

function InsightsBody({data,both}){
 return <>
  <p className="insightSummary">{data.summary}</p>
  <div className="insightSection"><h3>What the models detected</h3><div className="findingGrid">{data.findings.map(f=><article className="findingCard" key={f.source}><div className="findingTop"><SourceBadge source={f.source}/><span className="confChip">Model confidence <b>{f.confidence}%</b></span></div><h4>{f.detected}</h4><p>{f.meaning}</p></article>)}</div></div>
  <div className="insightSection"><h3>Possible associated conditions</h3>{data.possible_associations.length>0?<div className="assocGrid">{data.possible_associations.map((a,i)=><article className="assocCard" key={a.source+i}>{both&&<SourceBadge source={a.source}/>}<h4>{a.condition}</h4><p>{a.why}</p></article>)}</div>:<p className="insightNote">No specific conditions are suggested for these results. A normal model result does not rule out every respiratory condition, so persistent symptoms should still be checked by a clinician.</p>}<p className="insightNote" style={{marginTop:10}}>These are possible associations only, not diagnoses.</p></div>
  {both&&data.combined_interpretation&&<div className="insightSection"><h3>Combined interpretation</h3><div className="combinedCard">{data.agreement&&<span className={'agreeChip '+data.agreement}>{AGREEMENT[data.agreement]}</span>}<p>{data.combined_interpretation}</p></div></div>}
  <div className="insightSection"><h3>Recommended next steps</h3><ul className="insightList numbered">{data.recommended_next_steps.map((t,i)=><li key={i}>{t}</li>)}</ul></div>
  <div className="insightSection"><h3>Limitations</h3><ul className="insightList">{data.limitations.map((t,i)=><li key={i}>{t}</li>)}</ul></div>
 </>}

function InsightsDisclaimer(){return <div className="insightsFooter"><div className="disclaimer"><Info size={16}/><p><b>Medical disclaimer.</b> This explanation is AI-generated educational information. It is not medical advice, a diagnosis or a treatment plan, and the AI can make mistakes. Model confidence is not the probability that you have a disease. Always consult a qualified healthcare professional.</p></div><div className="emergency"><AlertTriangle size={16}/><p><b>Seek urgent medical care</b> if you have severe difficulty breathing, chest pain, blue or grey lips, confusion, coughing up blood or a very high fever. Do not wait for these results. Call your local emergency number.</p></div></div>}

function Suggestions({suggestions}){return <section className="suggestions"><div className="suggestionsHeading"><div className="suggestionIcon"><Lightbulb size={19}/></div><div><div className="eyebrow">NEXT STEPS</div><h2>Suggestions after analysis</h2><p>General guidance based on the displayed model findings. These suggestions are not a diagnosis or treatment plan.</p></div></div><div className="suggestionGrid">{suggestions.map((item,i)=><article className="suggestionCard" key={i}><span className="suggestionNumber">0{i+1}</span><div><h3>{item.title}</h3><p>{item.text}</p></div></article>)}</div></section>}

function getSuggestions(result){
 const out=[];
 const a=result.audio?.label||''; const x=result.xray?.label||'';
 if(a.includes('CRACKLES')||a.includes('WHEEZES')) out.push({title:'Consider clinical review',text:'An abnormal respiratory sound pattern was detected. Consider review by a qualified healthcare professional, especially if symptoms are present or persistent.'});
 if(a.includes('CRACKLES + WHEEZES')) out.push({title:'Review the complete respiratory picture',text:'Because both sound patterns were detected, review the recording together with symptoms, examination findings and any available imaging.'});
 if(a==='NORMAL') out.push({title:'No abnormal sound pattern predicted',text:'The model did not detect crackles or wheezes in this recording. Persistent or worsening symptoms should still be evaluated by a clinician.'});
 if(x && x!=='NO PNEUMONIA') out.push({title:'Discuss the X-ray finding with a clinician',text:'The model indicates a pneumonia-related pattern. A clinician should review the original X-ray and clinical context before any diagnosis or treatment decision.'});
 if(x==='NO PNEUMONIA') out.push({title:'Continue clinical evaluation when needed',text:'No pneumonia pattern was predicted by the model, but a normal model result does not rule out every respiratory condition. Persistent symptoms should still be evaluated.'});
 if(result.audio&&result.xray) out.push({title:'Compare both modalities',text:'Use the chest-sound and X-ray predictions together as screening support. Differences between the two should be reviewed by a healthcare professional.'});
 out.push({title:'Keep the original records',text:'Keep the original WAV recording and X-ray available so a clinician can review the source data alongside the model output.'});
 return out.slice(0,4);
}
function ResultCard({title,icon,label,value,confidence,extra,image,probabilities}){return <article className="resultCard"><div className="resultCardTop"><span className="resultIcon">{icon}</span><div><span>{title}</span><b>{label}</b></div><span className="confidence">{confidence}%</span></div>{image&&<img className="resultImage" src={image} alt="Uploaded X-ray"/>}<div className="resultMain"><span className="resultLabel">MODEL PREDICTION</span><h3>{value}</h3><div className="confidenceBar"><i style={{width:confidence+'%'}}/></div><div className="confidenceRow"><span>Confidence</span><b>{confidence}%</b></div>{probabilities&&<div className="probList">{Object.entries(probabilities).map(([k,v])=><div className="probRow" key={k}><span>{k}</span><div className="probBar"><i style={{width:(v*100)+'%'}}/></div><b>{(v*100).toFixed(1)}%</b></div>)}</div>}</div>{extra&&<div className="extraFinding"><Check size={16}/>{extra}</div>}</article>}
function HistoryPage({history,onOpen}){return <div className="content"><section className="historyHero"><div><div className="eyebrow"><History size={15}/> ASSESSMENT HISTORY</div><h1>Previous assessments</h1><p>Your saved assessments, stored securely in your account.</p></div><div className="historyStat"><b>{history.length}</b><span>saved assessments</span></div></section>{history.length===0?<div className="empty"><Clock3 size={30}/><h3>No assessments yet</h3><p>Your completed predictions will appear here.</p></div>:<div className="historyList">{history.map(r=><button className="historyItem" key={r.id} onClick={()=>onOpen(r)}><div className="historyIcon">{r.audio&&r.xray?<Activity size={19}/>:r.audio?<AudioLines size={19}/>:<FileImage size={19}/>}</div><div className="historyInfo"><b>{r.audio&&r.xray?'Combined assessment':r.audio?'Chest sound assessment':'Chest X-ray assessment'}</b><span>{r.date}</span></div><div className="historyPred"><span>{r.audio?.label||r.xray?.label}</span>{r.audio&&r.xray&&<small>+ 1 more result</small>}</div><ChevronRight size={18}/></button>)}</div>}</div>}
function formatBytes(b){if(!b)return '0 KB';return (b/1024/1024).toFixed(2)+' MB'}

createRoot(document.getElementById('root')).render(<App/>);