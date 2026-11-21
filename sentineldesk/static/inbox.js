const byId = id => document.getElementById(id);
const statuses = {draft_ready:"Draft ready", escalate:"Escalated", review:"In review", needs_clarification:"Needs clarification"};
let tickets = [], selected, nextId = 1;
function state(t) { return t.resolved ? "resolved" : t.result?.decision.status || "new"; }
function title(t) { return t.subject || "Untitled ticket"; }
function stateLabel(t) { return t.resolved ? "Resolved" : statuses[t.result?.decision.status] || "New"; }
function node(tag, text) {
  const element = document.createElement(tag);
  element.textContent = text;
  return element;
}
function list() {
  const term = byId("filter").value.toLowerCase();
  const view = byId("view").value;
  const shown = tickets.filter(t => (view === "all" || state(t) === view) &&
    (t.id+" "+t.subject+" "+t.body).toLowerCase().includes(term));
  if (!shown.includes(selected)) selected = shown[0];
  const nav = byId("list");
  nav.replaceChildren();
  byId("count").textContent = shown.length+" tickets";
  for (const t of shown) {
    const btn = node("button","");
    btn.className="item";
    btn.setAttribute("aria-current",String(t===selected));
    const meta = node("span","");
    meta.className="item-meta";
    const badge=node("small",stateLabel(t));
    badge.dataset.state=state(t);
    badge.className="state-tag";
    meta.append(node("small",t.id+(t.lang ? " · "+t.lang : "")),badge);
    const preview=node("span",t.body);
    preview.className="item-preview";
    btn.append(meta,node("strong",title(t)),preview);
    btn.onclick=()=>{selected=t;render();};
    nav.append(btn);
  }
  if (!shown.length) nav.append(node("p","No matching emails."));
}
function render() {
  list();
  byId("workspace").hidden=!selected;
  byId("no-selection").hidden=!!selected;
  if (!selected) return;
  const t=selected, r=t.result;
  byId("subject").textContent=title(t);
  byId("body").textContent=t.body;
  byId("ref").textContent=t.id+(t.lang ? " · "+t.lang : "");
  byId("run").disabled=t.busy || !!r;
  byId("run").hidden=!!r;
  byId("status").textContent=t.busy?"Triaging…":stateLabel(t);
  byId("status").dataset.state=state(t);
  byId("resolve").hidden=!r;
  byId("resolve").textContent=t.resolved?"Reopen ticket":"Mark resolved";
  byId("history").hidden=!r;
  byId("error").textContent=t.error || "";
  byId("result").hidden=!r;
  byId("compose").hidden=!r?.draft || t.resolved;
  const labels=byId("labels");
  labels.replaceChildren();
  for(const [key,val] of Object.entries({Queue:r?.triage?.queue.label || "Unassigned", Priority:r?.triage?.priority.label || "Awaiting triage", Type:r?.triage?.type.label || "Awaiting triage", Language:r?.language || t.lang || "Awaiting triage"})){
    if(val){const div=node("div","");div.append(node("dt",key),node("dd",val));labels.append(div);}
  }
  if (!r) return;
  byId("reason").textContent=[r.decision.destination.replaceAll("_"," "), ...r.decision.reasons.map(x=>x.message)].join(" · ");
  const evidence=byId("evidence");
  evidence.replaceChildren();
  for(const p of r.precedents){
    const d=node("details","");d.append(node("summary",p.ticket_id+" · "+(p.subject || "Untitled ticket")+" · "+p.score.toFixed(2)),node("p",p.answer_snippet));evidence.append(d);
  }
  if(!r.precedents.length)evidence.append(node("p","No precedents available."));
  byId("reply").value=t.edited ?? r.draft?.reply_text ?? "";
  byId("recipient").value=t.recipient || "";
}
byId("run").onclick=async()=>{
  const t=selected;t.busy=true;t.error="";render();
  try{
    const res=await fetch("/api/triage",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({subject:t.subject,body:t.body})});
    if(!res.ok)throw Error();
    t.result=await res.json();
    t.lang=(t.result.language || t.lang).toUpperCase();
    if(t.result.decision.reasons.some(r=>r.code==="TRIAGE_UNAVAILABLE")){t.result=null;throw Error();}
  }catch{t.error="Triage is unavailable. Try again.";}
  finally{t.busy=false;render();}
};
byId("filter").oninput=render;
byId("view").onchange=render;
byId("resolve").onclick=()=>{selected.resolved=!selected.resolved;render();};
byId("new-ticket").onclick=()=>byId("create-dialog").showModal();
byId("cancel").onclick=()=>byId("create-dialog").close();
byId("create-form").onsubmit=e=>{
  e.preventDefault();
  const body=byId("new-body").value.trim();
  if (!body) {byId("new-body").focus();return;}
  selected={id:"LOCAL-"+String(nextId++).padStart(3,"0"),subject:byId("new-subject").value.trim(),body,lang:""};
  tickets.unshift(selected);
  byId("view").value="all";byId("filter").value="";
  byId("create-dialog").close();e.target.reset();render();
};
byId("reply").oninput=e=>selected.edited=e.target.value;
byId("recipient").oninput=e=>selected.recipient=e.target.value;
byId("gmail").onclick=()=>{
  if(!byId("recipient").reportValidity())return;
  const params=new URLSearchParams({view:"cm",fs:"1",to:byId("recipient").value.trim(),su:"Re: "+selected.subject,body:byId("reply").value});
  window.open("https://mail.google.com/mail/?"+params,"_blank","noopener,noreferrer");
};
fetch("/static/inbox-samples.json").then(r=>{if(!r.ok)throw Error();return r.json();}).then(rows=>{
  tickets=rows.map((t,i)=>({...t,id:"DEMO-"+String(i+1).padStart(3,"0")}));
  selected=tickets[0];render();
}).catch(()=>byId("error").textContent="Could not load the sample inbox. Refresh to retry.");
