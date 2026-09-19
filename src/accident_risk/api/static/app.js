"use strict";

const $ = (id) => document.getElementById(id);
const route = [
  {lat:25.1869,lon:55.2671,name:"Business Bay"},{lat:25.1916,lon:55.2694,name:"Dubai Water Canal"},
  {lat:25.1972,lon:55.2744,name:"Downtown Dubai"},{lat:25.2048,lon:55.2708,name:"Financial Centre"},
  {lat:25.2117,lon:55.2739,name:"DIFC"},{lat:25.2201,lon:55.2826,name:"Museum of the Future"},
  {lat:25.2282,lon:55.2878,name:"Dubai World Trade Centre"},{lat:25.2354,lon:55.2854,name:"Al Satwa"},
  {lat:25.2409,lon:55.2705,name:"City Walk"},{lat:25.2323,lon:55.2638,name:"Jumeirah"},
  {lat:25.2203,lon:55.2662,name:"Sheikh Zayed Road"},{lat:25.2070,lon:55.2704,name:"Financial Centre"},
  {lat:25.1972,lon:55.2744,name:"Downtown Dubai"},{lat:25.1916,lon:55.2694,name:"Dubai Water Canal"},
  {lat:25.1869,lon:55.2671,name:"Business Bay"}
];
const riskBag=[18,44,68,46,88,72,24,52];
const state={running:true,tick:0,distance:0,speeds:[],stream:`demo-${Date.now()}`,prediction:null,risk:18,targetRisk:18,riskIndex:0,riskOrder:[],sending:false,map:null,marker:null,progressLine:null};
const routeLengths=route.slice(1).map((point,index)=>{
  const previous=route[index],lat1=previous.lat*Math.PI/180,lat2=point.lat*Math.PI/180,dLat=lat2-lat1,dLon=(point.lon-previous.lon)*Math.PI/180;
  const a=Math.sin(dLat/2)**2+Math.cos(lat1)*Math.cos(lat2)*Math.sin(dLon/2)**2;
  return 6371*2*Math.atan2(Math.sqrt(a),Math.sqrt(1-a));
});
const routeDistance=routeLengths.reduce((sum,length)=>sum+length,0);

function jitter(amount){return (Math.random()-.5)*amount;}
function nextRiskTarget(){
  if(state.riskIndex>=state.riskOrder.length){state.riskOrder=[...riskBag];for(let i=state.riskOrder.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[state.riskOrder[i],state.riskOrder[j]]=[state.riskOrder[j],state.riskOrder[i]];}state.riskIndex=0;}
  state.targetRisk=state.riskOrder[state.riskIndex]+jitter(5);state.riskIndex++;
}
function routePosition(distance){
  let remaining=distance%routeDistance;
  for(let index=0;index<routeLengths.length;index++){
    if(remaining<=routeLengths[index]){
      const start=route[index],end=route[index+1],mix=remaining/routeLengths[index],lat1=start.lat*Math.PI/180,lat2=end.lat*Math.PI/180,dLon=(end.lon-start.lon)*Math.PI/180;
      const bearing=(Math.atan2(Math.sin(dLon)*Math.cos(lat2),Math.cos(lat1)*Math.sin(lat2)-Math.sin(lat1)*Math.cos(lat2)*Math.cos(dLon))*180/Math.PI+360)%360;
      return {lat:start.lat+(end.lat-start.lat)*mix,lon:start.lon+(end.lon-start.lon)*mix,name:mix<.5?start.name:end.name,index,bearing};
    }
    remaining-=routeLengths[index];
  }
  return {...route[0],index:0,bearing:0};
}

function simulatedReading(tick){
  if(tick%12===0)nextRiskTarget();
  state.risk+=(state.targetRisk-state.risk)*.42+jitter(3);
  const risk=Math.max(5,Math.min(96,state.risk)),position=routePosition(state.distance),wave=Math.sin(tick/3);
  let event="Steady cruising",speed=45+wave*7,ax=wave*.35,ay=jitter(.35),gz=jitter(.05),steering=jitter(.03),heart=74+jitter(5),spo2=98,quality=97,tire=[235,236,238,237];
  if(risk>=80){event=Math.random()>.45?"Hard braking & swerve":"Low tire pressure alert";speed=78+jitter(18);ax=-6.1+jitter(1.4);ay=5.2+jitter(1.8);gz=.82+jitter(.24);steering=.5+jitter(.16);heart=126+jitter(12);spo2=95+jitter(2);quality=83+jitter(8);tire=[176+jitter(8),235+jitter(4),238+jitter(4),236+jitter(4)];}
  else if(risk>=60){event="Sharp lane movement";speed=68+jitter(16);ax=-2.1+jitter(2);ay=3.7+jitter(1.4);gz=.58+jitter(.2);steering=.34+jitter(.15);heart=105+jitter(10);spo2=96+jitter(1);quality=89+jitter(6);}
  else if(risk>=30){event="Busy stop-and-go traffic";speed=34+jitter(17);ax=(tick%5<2?-1.6:.8)+jitter(.6);ay=1.1+jitter(.8);gz=.19+jitter(.12);steering=.12+jitter(.1);heart=88+jitter(8);spo2=97+jitter(1);quality=93+jitter(4);}
  const heading=(position.bearing+jitter(8)+360)%360,rpm=Math.max(780,900+speed*34+jitter(180)),throttle=Math.max(3,Math.min(88,18+speed*.45+ax*2+jitter(8))),load=Math.max(10,Math.min(96,32+speed*.45+Math.abs(ax)*4+jitter(8)));
  return {position,risk:Math.round(risk),event,speed:Math.max(5,Math.round(speed)),ax,ay,az:9.80665+jitter(.14),gx:jitter(.035),gy:jitter(.035),gz,steering,heart:Math.round(heart),spo2:Math.round(spo2),quality:Math.round(quality),tire:tire.map(Math.round),heading,rpm:Math.round(rpm),throttle:Math.round(throttle),load:Math.round(load)};
}

function riskPresentation(score){
  if(score>=80)return ["VERY HIGH","Immediate attention","Several unusual readings changed together.","#d45454"];
  if(score>=60)return ["HIGH","Elevated activity","Strong movement or driver changes are being demonstrated.","#e98140"];
  if(score>=30)return ["MEDIUM","Watch conditions","Some readings are outside the steady demo range.","#d89424"];
  return ["LOW","Conditions look steady","No unusual movement in this demo.","#20a574"];
}

function initMap(){
  if(!window.L){$("actual-map").innerHTML='<div class="map-unavailable"><strong>Street map unavailable</strong><span>Connect to the internet and refresh to load OpenStreetMap.</span></div>';return;}
  state.map=L.map("actual-map",{zoomControl:true}).setView([route[2].lat,route[2].lon],14);
  L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png",{maxZoom:19,attribution:'&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'}).addTo(state.map);
  L.polyline(route.map(point=>[point.lat,point.lon]),{color:"#667d79",weight:5,opacity:.55,dashArray:"4 9"}).addTo(state.map);
  state.progressLine=L.polyline([],{color:"#0f766e",weight:6,opacity:.9}).addTo(state.map);
  const icon=L.divIcon({className:"vehicle-marker-wrap",html:'<div class="vehicle-map-marker" aria-label="Moving demo vehicle"><svg viewBox="0 0 24 24"><path d="M5 16v-5l2-4h10l2 4v5"/><path d="M3 14h18v5H3zM6 19v2M18 19v2M7 16h.01M17 16h.01"/></svg></div>',iconSize:[46,46],iconAnchor:[23,23]});
  state.marker=L.marker([route[0].lat,route[0].lon],{icon,title:"Moving demo vehicle"}).addTo(state.map).bindPopup("<strong>Demo vehicle</strong><br>Live randomized readings");
}

function updateMap(data){if(!state.map||!state.marker)return;const point=[data.position.lat,data.position.lon];state.marker.setLatLng(point);state.progressLine.setLatLngs([...route.slice(0,data.position.index+1).map(p=>[p.lat,p.lon]),point]);if(state.tick%4===0)state.map.panTo(point,{animate:true,duration:.6});}
function sparkline(values){if(values.length<2)return;const max=Math.max(...values,100),min=Math.min(...values,0),points=values.map((v,i)=>`${i/(values.length-1)*360},${72-(v-min)/(max-min||1)*60}`);$("speed-path").setAttribute("d",`M${points.join(" L")}`);$("speed-area").setAttribute("d",`M0,80 L${points.join(" L")} L360,80 Z`);}
function compass(degrees){const names=["N","NE","E","SE","S","SW","W","NW"];return `${Math.round(degrees)}° ${names[Math.round(degrees/45)%8]}`;}
function fixed(value,digits=2){return Number(value).toFixed(digits);}

function render(data){
  updateMap(data);$("place-name").textContent=data.position.name;$("coordinates").textContent=`${fixed(data.position.lat,5)}, ${fixed(data.position.lon,5)}`;
  $("speed").textContent=data.speed;$("heart-rate").textContent=data.heart;$("spo2").textContent=data.spo2;$("quality").textContent=data.quality;$("event-name").textContent=data.event;
  $("gps-lat").textContent=`${fixed(data.position.lat,5)}°`;$("gps-lon").textContent=`${fixed(data.position.lon,5)}°`;$("heading").textContent=compass(data.heading);$("gps-speed").textContent=`${data.speed} km/h`;
  $("accel").textContent=`${fixed(data.ax,1)} / ${fixed(data.ay,1)} / ${fixed(data.az,1)} m/s²`;$("gyro").textContent=`${fixed(data.gx)} / ${fixed(data.gy)} / ${fixed(data.gz)} rad/s`;$("lateral-force").textContent=`${fixed(Math.abs(data.ay)/9.80665)} g`;
  const swerving=Math.abs(data.gz)>.42||Math.abs(data.ay)>3;$("swerve").textContent=swerving?"Detected":"Not detected";$("swerve").className=`reading-pill ${swerving?"alert":"normal"}`;
  $("rpm").textContent=`${data.rpm.toLocaleString()} rpm`;$("throttle").textContent=`${data.throttle}%`;$("engine-load").textContent=`${data.load}%`;$("steering").textContent=`${fixed(data.steering*180/Math.PI,1)}°`;
  ["fl","fr","rl","rr"].forEach((name,index)=>{$(`tire-${name}`).textContent=`${data.tire[index]} kPa`;$(`tire-${name}`).classList.toggle("warning-reading",data.tire[index]<200);});
  const [level,message,detail,color]=riskPresentation(data.risk);$("risk-score").textContent=data.risk;$("risk-chip").textContent=level;$("risk-message").textContent=message;$("risk-detail").textContent=detail;$("risk-gauge").style.setProperty("--score",data.risk);$("risk-gauge").style.setProperty("--green",color);$("risk-chip").style.color=color;$("risk-card").style.setProperty("--risk-color",color);
  $("heart-note").textContent=data.heart>110?"Sharply raised in this demo":data.heart>85?"Moderately raised in this demo":"Normal demo range";
  $("travelled").textContent=`${state.distance.toFixed(2)} km`;$("data-points").textContent=state.tick;$("packet-status").textContent=`Packet ${state.tick}`;$("elapsed").textContent=`${String(Math.floor(state.tick/60)).padStart(2,"0")}:${String(state.tick%60).padStart(2,"0")}`;$("last-update").textContent=`Updated ${new Date().toLocaleTimeString([],{hour:"2-digit",minute:"2-digit",second:"2-digit"})}`;
  state.speeds.push(data.speed);if(state.speeds.length>25)state.speeds.shift();sparkline(state.speeds);
}

async function sendTick(){
  if(!state.running||state.sending)return;state.sending=true;const data=simulatedReading(state.tick),timestamp=Date.now()/1000,heading=data.heading*Math.PI/180;
  const frame={timestamp,stream_id:state.stream,source:"browser_continuous_demo",gps_latitude:data.position.lat,gps_longitude:data.position.lon,gps_speed:data.speed/3.6,gps_heading:heading,vehicle_speed:data.speed/3.6,rpm:data.rpm,throttle_position:data.throttle,engine_load:data.load,steering_angle:data.steering,tire_pressure_fl:data.tire[0],tire_pressure_fr:data.tire[1],tire_pressure_rl:data.tire[2],tire_pressure_rr:data.tire[3],accel_x:data.ax,accel_y:data.ay,accel_z:data.az,gyro_x:data.gx,gyro_y:data.gy,gyro_z:data.gz,heart_rate:data.heart,spo2:data.spo2,physiology_signal_quality:data.quality/100};
  try{const response=await fetch("/ingest",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(frame)});if(!response.ok)throw new Error(`Server returned ${response.status}`);state.prediction=await response.json();$("connection").className="connection online";$("connection").querySelector("span").textContent="Live demo connected";state.tick++;state.distance+=data.speed/3600;render(data);}catch(error){$("connection").className="connection offline";$("connection").querySelector("span").textContent="Connection lost";showToast(`Could not send demo data: ${error.message}`);}finally{state.sending=false;}
}

function showToast(message){const toast=$("toast");toast.textContent=message;toast.classList.add("show");clearTimeout(showToast.timer);showToast.timer=setTimeout(()=>toast.classList.remove("show"),3200);}
async function restart(){state.running=false;try{await fetch("/session/start",{method:"POST"});}catch{showToast("The server could not restart the trip.");}Object.assign(state,{running:true,tick:0,distance:0,speeds:[],stream:`demo-${Date.now()}`,prediction:null,risk:18,targetRisk:18,riskIndex:0,riskOrder:[]});if(state.progressLine)state.progressLine.setLatLngs([]);$("toggle-button").classList.remove("paused");$("toggle-button").setAttribute("aria-pressed","false");$("toggle-button").querySelector("span").textContent="Pause drive";sendTick();showToast("A fresh continuous drive has started.");}

$("toggle-button").addEventListener("click",()=>{state.running=!state.running;const button=$("toggle-button");button.classList.toggle("paused",!state.running);button.setAttribute("aria-pressed",String(!state.running));button.querySelector("span").textContent=state.running?"Pause drive":"Continue drive";showToast(state.running?"Live feed continued.":"Live feed paused.");if(state.running)sendTick();});
$("reset-button").addEventListener("click",restart);
initMap();restart();setInterval(sendTick,1000);
