/**
 * SCENARIO DEFINITIONS
 * Maps the prompt's 17 scenarios to internal IDs and Descriptions.
 */
const SCENARIOS = [
    { id: 1, text: "Green light but a pedestrian is in front while the car wants to go straight." },
    { id: 2, text: "Green light but a pedestrian is on the right while the car wants to turn right." },
    { id: 3, text: "Green light but a pedestrian is on the left while the car wants to turn left." },
    { id: 4, text: "No green left-turn arrow and an opposite car is present while the car wants to turn left." },
    { id: 5, text: "No green left-turn arrow and a car is coming from the left or right while the car wants to turn left." },
    { id: 6, text: "No green left-turn arrow and the intersection is clear while the car wants to turn left (gap acceptance)." },
    { id: 7, text: "Green left-turn arrow is on and the car wants to turn left (protected left turn)." },
    { id: 8, text: "Red light and the car wants to turn right with no car from the left and no pedestrian on the right." },
    { id: 9, text: "A car from the left or a pedestrian at the right blocks a right turn." },
    { id: 10, text: "Stop sign with cross-traffic present from left or right." },
    { id: 11, text: "Stop sign with no cross-traffic present." },
    { id: 12, text: "Stop sign when turning right and cars/pedestrians may appear on the right or from the left." },
    { id: 13, text: "Red light while the car wants to go straight and must stop before proceeding." },
    { id: 14, text: "Green light but the car is currently stopped and must decide when to proceed." },
    { id: 15, text: "A side car is present during lane-keeping, merging, or turning near the intersection." },
    { id: 16, text: "Uncontrolled intersection with no lights or stop sign but cross-traffic is present." },
    { id: 17, text: "Stop sign present simultaneously with an active traffic light signal (conflicting controls)." }
];

// Populate Select
const selectEl = document.getElementById('scenario-select');
SCENARIOS.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s.id;
    opt.textContent = `${s.id}. ${s.text.substring(0, 60)}...`;
    selectEl.appendChild(opt);
});

const descEl = document.getElementById('scenario-desc');
selectEl.addEventListener('change', (e) => {
    const sc = SCENARIOS.find(s => s.id == e.target.value);
    if(sc) descEl.textContent = sc.text;
});

// Speed Slider Logic
const speedSlider = document.getElementById('speed-slider');
const speedDisplay = document.getElementById('speed-display');
speedSlider.addEventListener('input', (e) => {
    speedDisplay.textContent = parseFloat(e.target.value).toFixed(1) + 'x';
});

// Export Project Button Logic
// document.getElementById('export-project-btn').addEventListener('click', () => {
//     // Get the full HTML content of the current page
//     const htmlContent = document.documentElement.outerHTML;
//     
//     // Create a blob
//     const blob = new Blob([htmlContent], { type: 'text/html' });
//     
//     // Create a link to download
//     const url = URL.createObjectURL(blob);
//     const a = document.createElement('a');
//     a.href = url;
//     a.download = 'av_simulator_project.html';
//     document.body.appendChild(a);
//     a.click();
//     
//     // Cleanup
//     document.body.removeChild(a);
//     URL.revokeObjectURL(url);
// });

/**
 * MOCK BACKEND: get_controller(scenario_text)
 * Returns the JSON structure of the FSA.
 */
function get_controller(scenarioId) {
    // Base template for the FSA graph
    const fsa = {
        "scenario_id": scenarioId,
        "graph": {
            "nodes": [
                { "id": 0, "description": "Initial" },
                { "id": 1, "description": "Action: go_straight" },
                { "id": 2, "description": "Action: stop" },
                { "id": 3, "description": "Action: wait" },
                { "id": 4, "description": "Action: turn_left" },
                { "id": 5, "description": "Action: turn_right" }
            ],
            "edges": []
        }
    };

    const edges = fsa.graph.edges;

    // Logic to build edges based on the scenario intent
    switch(parseInt(scenarioId)) {
        case 1: // Green light + Ped Front -> Stop, then Wait/Go
            edges.push(
                { from: 0, to: 2, condition: "pedestrian_in_front", action: "stop" },
                { from: 2, to: 3, condition: "pedestrian_in_front", action: "wait" },
                { from: 3, to: 1, condition: "!pedestrian_in_front", action: "go_straight" }
            );
            break;
        case 2: // Turn Right blocked by Ped
            edges.push(
                { from: 0, to: 2, condition: "pedestrian_at_right", action: "stop" },
                { from: 2, to: 3, condition: "pedestrian_at_right", action: "wait" },
                { from: 3, to: 5, condition: "!pedestrian_at_right", action: "turn_right" }
            );
            break;
        case 3: // Turn Left blocked by Ped
            edges.push(
                { from: 0, to: 2, condition: "pedestrian_at_left", action: "stop" },
                { from: 2, to: 3, condition: "pedestrian_at_left", action: "wait" },
                { from: 3, to: 4, condition: "!pedestrian_at_left", action: "turn_left" }
            );
            break;
        case 4: // Left Turn blocked by Opposing Car
            edges.push(
                { from: 0, to: 3, condition: "opposite_car", action: "wait" },
                { from: 3, to: 4, condition: "!opposite_car", action: "turn_left" }
            );
            break;
        case 5: // Unprotected Left Turn
            edges.push(
                { from: 0, to: 3, condition: "car_from_left", action: "wait" },
                { from: 3, to: 4, condition: "!car_from_left", action: "turn_left" }
            );
            break;
        case 7: // Protected Left
            edges.push(
                { from: 0, to: 4, condition: "green_left_turn_light", action: "turn_left" }
            );
            break;
        case 8: // Red light right turn on clear
            edges.push(
                { from: 0, to: 2, condition: "true", action: "stop" }, // Full stop first
                { from: 2, to: 5, condition: "!car_from_left && !pedestrian_at_right", action: "turn_right" }
            );
            break;
        case 10: // Stop sign cross traffic
            edges.push(
                { from: 0, to: 2, condition: "stop_sign", action: "stop" },
                { from: 2, to: 3, condition: "car_from_left || car_from_right", action: "wait" },
                { from: 3, to: 1, condition: "!car_from_left && !car_from_right", action: "go_straight" } 
            );
            break;
        case 13: // Red light straight
            edges.push(
                { from: 0, to: 2, condition: "!green_traffic_light", action: "stop" },
                { from: 2, to: 3, condition: "!green_traffic_light", action: "wait" },
                { from: 3, to: 1, condition: "green_traffic_light", action: "go_straight" }
            );
            break;
        case 11: // Stop sign no traffic
            edges.push(
                 { from: 0, to: 2, condition: "stop_sign", action: "stop" },
                 { from: 2, to: 1, condition: "true", action: "go_straight" }
            );
            break;
        case 14: // Green light but currently stopped
             edges.push(
                 { from: 0, to: 1, condition: "green_traffic_light", action: "go_straight" }
             )
             break;
        default:
            // Generic safety fallback for other scenarios
            edges.push(
                { from: 0, to: 3, condition: "obstacle_detected", action: "wait" },
                { from: 0, to: 1, condition: "clear_path", action: "go_straight" }
            );
            break;
    }

    return JSON.stringify(fsa, null, 2);
}

/**
 * SIMULATION ENGINE
 */
const canvas = document.getElementById('sim-canvas');
const ctx = canvas.getContext('2d');

let animationId;
let simState = {
    car: { x: 0, y: 0, angle: 0, speed: 0, intent: 'straight' }, // intent: straight, left, right
    vars: { // The Formal Model Variables
        green_traffic_light: false,
        green_left_turn_light: false,
        opposite_car: false,
        car_from_left: false,
        car_from_right: false,
        pedestrian_at_left: false,
        pedestrian_at_right: false,
        pedestrian_in_front: false,
        side_car: false,
        stop_sign: false,
        action: 'stop'
    },
    time: 0,
    phase: 'approach' // approach, intersection, departing
};

// Resize Canvas
function resizeCanvas() {
    const parent = canvas.parentElement;
    canvas.width = parent.clientWidth;
    canvas.height = parent.clientHeight;
}
window.addEventListener('resize', resizeCanvas);
resizeCanvas();

// Initialize Simulation based on Scenario
function initSimulation(scenarioId) {
    // Reset State
    simState.time = 0;
    simState.phase = 'approach';
    simState.car = { x: canvas.width/2, y: canvas.height - 50, angle: -Math.PI/2, speed: 0, intent: 'straight' };
    
    // Default Vars (All safe)
    for (let key in simState.vars) if (key !== 'action') simState.vars[key] = false;
    simState.vars.action = 'stop'; // Start stopped

    // Scenario Specific Setup
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;

    switch(parseInt(scenarioId)) {
        case 1: // Green, Ped Front
            simState.vars.green_traffic_light = true;
            simState.vars.pedestrian_in_front = true; // Will need to animate pedestrian walking away
            simState.car.intent = 'straight';
            break;
        case 2: // Green, Turn Right, Ped Right
            simState.vars.green_traffic_light = true;
            simState.vars.pedestrian_at_right = true;
            simState.car.intent = 'right';
            break;
        case 3: // Green, Turn Left, Ped Left
            simState.vars.green_traffic_light = true;
            simState.vars.pedestrian_at_left = true;
            simState.car.intent = 'left';
            break;
        case 4: // No Green Arrow, Opp Car, Turn Left
            simState.vars.green_traffic_light = true; // Main green usually on if arrow off
            simState.vars.green_left_turn_light = false;
            simState.vars.opposite_car = true;
            simState.car.intent = 'left';
            break;
        case 5: // No Arrow, Car Left/Right, Turn Left (Unprotected)
            simState.vars.green_traffic_light = true;
            simState.vars.green_left_turn_light = false;
            simState.vars.car_from_left = true; // Conflict
            simState.car.intent = 'left';
            break;
        case 6: // Gap Acceptance
            simState.vars.green_traffic_light = true;
            simState.vars.green_left_turn_light = false;
            simState.car.intent = 'left';
            // No obstacles
            break;
        case 7: // Protected Left
            simState.vars.green_left_turn_light = true;
            simState.car.intent = 'left';
            break;
        case 8: // Red, Turn Right, Clear
            simState.vars.green_traffic_light = false;
            simState.car.intent = 'right';
            break;
        case 9: // Red/Green, Turn Right, Blocked
            simState.vars.green_traffic_light = false; // or true
            simState.vars.car_from_left = true;
            simState.car.intent = 'right';
            break;
        case 10: // Stop Sign, Cross Traffic
            simState.vars.stop_sign = true;
            simState.vars.car_from_left = true;
            simState.car.intent = 'straight';
            break;
        case 11: // Stop Sign, Clear
            simState.vars.stop_sign = true;
            simState.car.intent = 'straight';
            break;
        case 12: // Stop Sign, Turn Right, Risk
            simState.vars.stop_sign = true;
            simState.car.intent = 'right';
            simState.vars.car_from_left = true; // Initial risk
            break;
        case 13: // Red Light Straight
            simState.vars.green_traffic_light = false;
            simState.car.intent = 'straight';
            break;
        case 14: // Green, Stopped
            simState.vars.green_traffic_light = true;
            simState.car.intent = 'straight';
            simState.vars.action = 'stop';
            break;
        case 15: // Side Car
            simState.vars.green_traffic_light = true;
            simState.vars.side_car = true;
            simState.car.intent = 'straight';
            break;
        case 16: // Uncontrolled
            simState.vars.green_traffic_light = false;
            simState.vars.stop_sign = false;
            simState.vars.car_from_right = true; // Yield to right usually
            simState.car.intent = 'straight';
            break;
        case 17: // Conflict (Stop + Green)
            simState.vars.green_traffic_light = true;
            simState.vars.stop_sign = true;
            simState.car.intent = 'straight';
            break;
    }
}

// The Logic Controller (Runs every frame)
function updateController(scenarioId) {
    const v = simState.vars;
    let nextAction = 'stop'; // Default safety

    // Dynamic logic simulating the FSA processing
    // We implement the "Edges" logic here
    
    // 1. Initial State Check (Are we at the line?)
    const distToStopLine = Math.abs(simState.car.y - (canvas.height/2 + 60));
    // Increased the detection buffer slightly for higher speed stability
    const atLine = distToStopLine < 20; 
    const passedLine = simState.car.y < (canvas.height/2 + 50);

    // Logic Tree based on Scenarios
    if (!passedLine) {
        // APPROACHING
        if (v.green_traffic_light && v.pedestrian_in_front) nextAction = 'stop'; // S1
        else if (v.green_traffic_light && !v.pedestrian_in_front && !v.stop_sign) nextAction = 'go_straight'; // S14
        else if (!v.green_traffic_light && !v.green_left_turn_light && !v.stop_sign && scenarioId !== 16) nextAction = 'stop'; // Red light
        else if (v.stop_sign) {
            // Must stop first. In this sim, we check speed.
            if (simState.car.speed > 0.5 && !atLine) nextAction = 'stop'; // Decelerate to stop
            else if (atLine && simState.time < 80) nextAction = 'stop'; // Wait a bit
            else {
                // After stop, check traffic
                if (v.car_from_left || v.car_from_right || v.pedestrian_at_right) nextAction = 'wait';
                else nextAction = simState.car.intent === 'right' ? 'turn_right' : 'go_straight';
            }
        }
        else {
            nextAction = 'go_straight'; // Default approach
        }
        
        // Specific Intent overrides
        if(nextAction !== 'stop' && nextAction !== 'wait') {
            if(simState.car.intent === 'left') nextAction = 'turn_left';
            if(simState.car.intent === 'right') nextAction = 'turn_right';
        }

    } else {
        // IN INTERSECTION
        if (simState.car.intent === 'left') {
            // Check conflicts
            if (v.green_left_turn_light) nextAction = 'turn_left';
            else if (v.opposite_car || v.car_from_left || v.pedestrian_at_left) nextAction = 'wait';
            else nextAction = 'turn_left';
        }
        else if (simState.car.intent === 'right') {
            if (v.pedestrian_at_right || v.car_from_left) nextAction = 'wait';
            else nextAction = 'turn_right';
        }
        else {
             nextAction = 'go_straight';
        }
    }

    // Global override for Pedestrians in path
    if (v.pedestrian_in_front && !passedLine) nextAction = 'stop';
    
    // Update Action Variable
    simState.vars.action = nextAction;
    
    // Update Physics based on Action
    updatePhysics(nextAction);
}

function updatePhysics(action) {
    // Get Speed Factor from Slider
    const timeScale = parseFloat(document.getElementById('speed-slider').value);

    // Base Constants (calibrated for 1.0x)
    const acceleration = 0.25; 
    const maxSpeed = 7;       
    const turnSpeed = 3.5;    
    const turnRate = 0.06;    
    const friction = 0.4;

    // Handle Environment Changes (simulating passage of time)
    simState.time += timeScale; // Time moves faster with slider
    if (simState.time > 150) {
         // Simulate obstacles clearing after some time
         simState.vars.pedestrian_in_front = false;
         simState.vars.opposite_car = false;
         simState.vars.car_from_left = false;
         simState.vars.car_from_right = false;
         // Traffic light change for S13
         if(document.getElementById('scenario-select').value == 13 && simState.time > 200) simState.vars.green_traffic_light = true;
    }

    // Car Physics scaling with timeScale
    if (action === 'stop' || action === 'wait') {
        simState.car.speed = Math.max(0, simState.car.speed - (friction * timeScale)); 
    } else if (action === 'go_straight') {
        simState.car.speed = Math.min(maxSpeed, simState.car.speed + (acceleration * timeScale));
    } else if (action === 'turn_left') {
        simState.car.speed = Math.min(turnSpeed, simState.car.speed + (acceleration * timeScale));
        if (simState.car.y < canvas.height/2 + 20) {
            simState.car.angle = Math.max(-Math.PI, simState.car.angle - (turnRate * timeScale));
        }
    } else if (action === 'turn_right') {
        simState.car.speed = Math.min(turnSpeed, simState.car.speed + (acceleration * timeScale));
        if (simState.car.y < canvas.height/2 + 20) {
            simState.car.angle = Math.min(0, simState.car.angle + (turnRate * timeScale));
        }
    }

    // Move Position (Velocity * Time)
    simState.car.x += Math.cos(simState.car.angle) * simState.car.speed * timeScale;
    simState.car.y += Math.sin(simState.car.angle) * simState.car.speed * timeScale;
}

// Drawing Functions
function drawIntersection() {
    const w = canvas.width;
    const h = canvas.height;
    const cx = w/2;
    const cy = h/2;
    const roadW = 120;

    ctx.fillStyle = '#334155'; // Road color
    ctx.fillRect(0, cy - roadW/2, w, roadW); // Horizontal
    ctx.fillRect(cx - roadW/2, 0, roadW, h); // Vertical

    // Dashed Lines
    ctx.strokeStyle = '#94a3b8';
    ctx.setLineDash([20, 20]);
    ctx.lineWidth = 2;
    
    ctx.beginPath();
    ctx.moveTo(0, cy); ctx.lineTo(w, cy);
    ctx.moveTo(cx, 0); ctx.lineTo(cx, h);
    ctx.stroke();
    ctx.setLineDash([]);

    // Stop Lines
    ctx.strokeStyle = '#e2e8f0';
    ctx.lineWidth = 4;
    ctx.beginPath();
    // Bottom (Car start)
    ctx.moveTo(cx, cy + roadW/2); ctx.lineTo(cx + roadW/2, cy + roadW/2);
    // Left
    ctx.moveTo(cx - roadW/2, cy); ctx.lineTo(cx - roadW/2, cy - roadW/2);
    // Top
    ctx.moveTo(cx, cy - roadW/2); ctx.lineTo(cx - roadW/2, cy - roadW/2);
    // Right
    ctx.moveTo(cx + roadW/2, cy); ctx.lineTo(cx + roadW/2, cy + roadW/2);
    ctx.stroke();

    // Crosswalks
    ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
    const xw = 30;
    // Bottom
    ctx.fillRect(cx - roadW/2, cy + roadW/2 + 10, roadW, xw);
    // Top
    ctx.fillRect(cx - roadW/2, cy - roadW/2 - 10 - xw, roadW, xw);
    // Left
    ctx.fillRect(cx - roadW/2 - 10 - xw, cy - roadW/2, xw, roadW);
    // Right
    ctx.fillRect(cx + roadW/2 + 10, cy - roadW/2, xw, roadW);

    // Traffic Light Box (Top Right relative to car)
    drawTrafficLight(cx + roadW/2 + 20, cy + roadW/2 + 20);
}

function drawTrafficLight(x, y) {
    ctx.fillStyle = '#1e293b';
    ctx.fillRect(x, y, 30, 80);
    ctx.strokeStyle = '#64748b';
    ctx.strokeRect(x, y, 30, 80);

    // Red
    ctx.fillStyle = (!simState.vars.green_traffic_light && !simState.vars.green_left_turn_light) ? '#ef4444' : '#450a0a';
    ctx.beginPath(); ctx.arc(x + 15, y + 15, 8, 0, Math.PI*2); ctx.fill();

    // Yellow (Transition - ignored for simplicity, or blink if wait)
    ctx.fillStyle = '#422006';
    ctx.beginPath(); ctx.arc(x + 15, y + 40, 8, 0, Math.PI*2); ctx.fill();

    // Green
    ctx.fillStyle = (simState.vars.green_traffic_light) ? '#22c55e' : '#052e16';
    ctx.beginPath(); ctx.arc(x + 15, y + 65, 8, 0, Math.PI*2); ctx.fill();

    // Left Arrow (Separate box usually, but adding here for viz)
    if (simState.vars.green_left_turn_light) {
         ctx.fillStyle = '#22c55e';
         ctx.font = '16px sans-serif';
         ctx.textAlign = 'center';
         ctx.textBaseline = 'middle';
         ctx.fillText('←', x - 20, y + 65);
    }
}

function drawEntities() {
    const w = canvas.width;
    const h = canvas.height;
    const cx = w/2;
    const cy = h/2;

    // Ego Car
    ctx.save();
    ctx.translate(simState.car.x, simState.car.y);
    ctx.rotate(simState.car.angle + Math.PI/2); // Adjust for drawing upright
    
    // Car Body
    ctx.fillStyle = '#3b82f6'; // Blue
    ctx.shadowColor = 'rgba(0,0,0,0.5)';
    ctx.shadowBlur = 10;
    ctx.fillRect(-15, -30, 30, 60); // w 30, h 60
    
    // Windshield
    ctx.fillStyle = '#93c5fd';
    ctx.fillRect(-12, -20, 24, 15);
    
    // Headlights
    ctx.fillStyle = '#fef08a';
    ctx.shadowColor = '#fef08a';
    ctx.shadowBlur = 5;
    ctx.fillRect(-12, -32, 6, 4);
    ctx.fillRect(6, -32, 6, 4);
    ctx.shadowBlur = 0;
    
    ctx.restore();

    // Obstacles
    const v = simState.vars;

    // Pedestrian Front (Walking across top crosswalk or bottom)
    if (v.pedestrian_in_front) {
        drawPedestrian(cx, cy + 90); // Blocking start
    }
    if (v.pedestrian_at_right) {
        drawPedestrian(cx + 80, cy);
    }
    if (v.pedestrian_at_left) {
        drawPedestrian(cx - 80, cy);
    }

    // Other Cars
    if (v.opposite_car) {
        drawEnemyCar(cx - 15, cy - 120, 0); // Facing down
    }
    if (v.car_from_left) {
        drawEnemyCar(cx - 150, cy - 15, -Math.PI/2);
    }
    if (v.car_from_right) {
        drawEnemyCar(cx + 150, cy + 15, Math.PI/2);
    }
    if (v.stop_sign) {
        drawStopSign(cx + 70, cy + 70);
    }
}

function drawPedestrian(x, y) {
    ctx.fillStyle = '#eab308';
    ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = 'black';
    ctx.font = '10px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('PED', x, y+3);
}

function drawEnemyCar(x, y, angle) {
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(angle);
    ctx.fillStyle = '#ef4444'; // Red
    ctx.fillRect(-15, -30, 30, 60);
    ctx.restore();
}

function drawStopSign(x, y) {
    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    // Octagon approx
    ctx.moveTo(x-10, y-24); ctx.lineTo(x+10, y-24);
    ctx.lineTo(x+24, y-10); ctx.lineTo(x+24, y+10);
    ctx.lineTo(x+10, y+24); ctx.lineTo(x-10, y+24);
    ctx.lineTo(x-24, y+10); ctx.lineTo(x-24, y-10);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = 'white';
    ctx.font = 'bold 10px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('STOP', x, y+4);
}

function checkLTL() {
    // Evaluates the LTL properties based on current state
    const v = simState.vars;
    const report = [];

    // Helper for report
    const check = (name, condition, desc) => {
        const status = condition ? "PASS" : "FAIL";
        const color = condition ? "text-green-500" : "text-red-500 font-bold";
        return `<div class="flex justify-between items-center bg-slate-800/50 p-1.5 rounded border border-slate-700/50"><span class="text-slate-400 font-semibold">${name}</span> <span class="${color} text-[10px] tracking-wider border border-current px-1 rounded">${status}</span></div>`;
    };

    // Phi 1: Pedestrian -> Stop (Simplified check: if ped exists, action must be stop or wait)
    if (v.pedestrian_in_front) {
        report.push(check("Φ1 (Ped Safe)", v.action === 'stop' || v.action === 'wait', "Pedestrian Safety"));
    }
    // Phi 3: Red Light -> No Go Straight
    if (!v.green_traffic_light && v.action === 'go_straight') {
        report.push(check("Φ3 (Red Light)", false, "Ran Red Light"));
    } else {
         // report.push(check("Φ3 (Red Light)", true, "")); // Reduce noise
    }
    // Phi 4: Stop Sign
    if (v.stop_sign) {
        // Must eventually stop. If moving fast through it, fail.
        // Simplified: If sim time < 50 and speed > 2 and action != stop
        // Adjusted check for higher speeds: grace period is shorter
        const ranStop = (v.action !== 'stop' && simState.time < 30 && simState.car.speed > 2);
        report.push(check("Φ4 (Stop Compliance)", !ranStop, "Stop Compliance"));
    }
    // Phi 5: Turn Right Safety
    if (v.action === 'turn_right') {
        report.push(check("Φ5 (Turn Safe)", !v.car_from_left && !v.pedestrian_at_right, "Right Turn Safety"));
    }

    if(report.length === 0) {
        document.getElementById('ltl-monitor').innerHTML = '<div class="text-slate-500 italic text-center py-2">No active constraints violated.</div>';
    } else {
        document.getElementById('ltl-monitor').innerHTML = report.join('');
    }
}

function loop() {
    // Clear
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Update Logic
    const id = document.getElementById('scenario-select').value;
    updateController(id);

    // Draw
    drawIntersection();
    drawEntities();

    // UI Updates
    document.getElementById('current-action').textContent = simState.vars.action.toUpperCase();
    
    // Check LTL
    checkLTL();

    // Update Var List UI
    const varList = Object.entries(simState.vars)
        .filter(([k]) => k !== 'action') // Action shown elsewhere
        .map(([k, val]) => {
            const color = val ? 'text-blue-400 font-bold' : 'text-slate-600';
            const icon = val ? 'check_circle' : 'radio_button_unchecked';
            return `<div class="flex justify-between items-center p-1 hover:bg-slate-800 rounded">
                        <span class="text-slate-400">${k}</span>
                        <span class="${color} flex items-center gap-1 text-[10px]">
                            ${val ? 'TRUE' : 'FALSE'}
                        </span>
                    </div>`;
        }).join('');
    document.getElementById('variables-monitor').innerHTML = varList;

    animationId = requestAnimationFrame(loop);
}

/**
 * Event Listeners
 */
document.getElementById('run-btn').addEventListener('click', () => {
    const id = document.getElementById('scenario-select').value;
    if (!id) {
        alert("Please select a scenario first.");
        return;
    }

    // Stop existing
    if (animationId) cancelAnimationFrame(animationId);

    // Generate Controller JSON text
    const jsonText = get_controller(id);
    document.getElementById('json-output').textContent = jsonText;

    // Init State
    initSimulation(id);
    
    // Update Status
    document.getElementById('sim-status-dot').className = "w-2.5 h-2.5 rounded-full bg-green-500 animate-pulse shadow-[0_0_8px_rgba(34,197,94,0.6)]";
    document.getElementById('sim-status-text').textContent = "RUNNING SCENARIO " + id;
    document.getElementById('sim-status-text').className = "text-xs font-mono text-green-400 font-bold";

    // Start Loop
    loop();
});

document.getElementById('reset-btn').addEventListener('click', () => {
     if (animationId) cancelAnimationFrame(animationId);
     ctx.clearRect(0, 0, canvas.width, canvas.height);
     document.getElementById('json-output').textContent = "// JSON output will appear here...";
     document.getElementById('current-action').textContent = "WAIT";
     document.getElementById('variables-monitor').innerHTML = '<div class="text-slate-500 py-4 text-center border border-dashed border-slate-700 rounded-lg">Waiting for simulation...</div>';
     document.getElementById('ltl-monitor').innerHTML = '<div class="text-slate-500 italic">No constraints active.</div>';
     document.getElementById('sim-status-dot').className = "w-2.5 h-2.5 rounded-full bg-gray-500";
     document.getElementById('sim-status-text').textContent = "IDLE";
     document.getElementById('sim-status-text').className = "text-xs font-mono text-slate-300";
     drawIntersection();
});

// Initial Draw
setTimeout(drawIntersection, 100);

