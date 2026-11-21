const byId = id => document.getElementById(id);
const statuses = {draft_ready:"Draft ready", escalate:"Escalated", review:"In review", needs_clarification:"Needs clarification"};
let tickets = [], selected;
function node(tag, text) {
  const element = document.createElement(tag);
  element.textContent = text;
  return element;
}
function list() {
  const term = byId("filter").value.toLowerCase();
  const shown = tickets.filter(t => (t.subject+" "+t.body).toLowerCase().includes(term));
  const nav = byId("list");
  nav.replaceChildren();
  byId("count").textContent = shown.length+" emails";
  for (const t of shown) {
    const btn = node("button","");
    btn.className="item";
    btn.setAttribute("aria-current",String(t===selected));
    btn.append(node("small",t.id+" · "+t.lang+" · "+(statuses[t.result?.decision.status] || "New")),node("strong",t.subject));
    btn.onclick=()=>{selected=t;render();};
    nav.append(btn);
  }
  if (!shown.length) nav.append(node("p","No matching emails."));
}
function render() {
  list();
  const t=selected, r=t.result;
  byId("subject").textContent=t.subject;
  byId("body").textContent=t.body;
  byId("ref").textContent=t.id+" · "+t.lang;
  byId("run").disabled=t.busy || !!r;
  byId("run").hidden=!!r;
  byId("status").textContent=t.busy?"Triaging…":statuses[r?.decision.status] || "New";
  byId("error").textContent=t.error || "";
  byId("result").hidden=!r;
  byId("compose").hidden=!r?.draft;
  if (!r) return;
  const labels=byId("labels");
  labels.replaceChildren();
  for(const [key,val] of Object.entries({Type:r.triage?.type.label, Queue:r.triage?.queue.label, Priority:r.triage?.priority.label, Destination:r.decision.destination.replaceAll("_"," ")})){
    if(val){const div=node("div","");div.append(node("dt",key),node("dd",val));labels.append(div);}
  }
  byId("reason").textContent=r.decision.reasons.map(x=>x.message).join(" ");
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
    if(t.result.decision.reasons.some(r=>r.code==="TRIAGE_UNAVAILABLE")){t.result=null;throw Error();}
  }catch{t.error="Triage is unavailable. Try again.";}
  finally{t.busy=false;render();}
};
byId("filter").oninput=list;
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
