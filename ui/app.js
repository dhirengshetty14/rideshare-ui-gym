// ============================================================================
// Rideshare UI Gym — frontend logic
//
// Architecture: vanilla JS, no framework. Single page renders the whole
// dispatch console. Every interactable element has a `data-test-id` so
// agents (and tests) can target it deterministically.
//
// Data flow:
//   1. User picks task + seed -> POST /api/reset
//   2. UI re-renders from /api/state response
//   3. User clicks an action button -> shows a modal (or directly applies)
//   4. Modal confirm -> POST /api/apply with kind+args
//   5. Server returns new state -> UI re-renders
//   6. User clicks "Submit task" -> POST /api/apply { kind: "finish" }
//                                   then POST /api/verify, show result
//
// Agents run the same flow but synthesize the click/type events themselves.
// ============================================================================

const API = {
  tasks:    () => fetch("/api/tasks").then(r => r.json()),
  reset:    (task_id, seed) => fetch("/api/reset", {
              method: "POST", headers: {"Content-Type": "application/json"},
              body: JSON.stringify({ task_id, seed }),
            }).then(r => r.json()),
  state:    () => fetch("/api/state").then(r => r.json()),
  apply:    (kind, args) => fetch("/api/apply", {
              method: "POST", headers: {"Content-Type": "application/json"},
              body: JSON.stringify({ kind, args }),
            }).then(r => r.json()),
  verify:   () => fetch("/api/verify", {
              method: "POST", headers: {"Content-Type": "application/json"},
            }).then(r => r.json()),
};


// --------------------------------------------------------------------------- //
// Render
// --------------------------------------------------------------------------- //

function render(state) {
  // Banner
  document.getElementById("task-id").textContent = state.task_id;
  document.getElementById("task-difficulty").textContent = state.task_difficulty;
  document.getElementById("task-brief").textContent = state.task_brief;
  document.getElementById("step-counter").textContent = state.step;

  // Driver fleet
  const driverRows = document.getElementById("driver-rows");
  const drivers = Object.values(state.drivers).sort((a, b) => a.id.localeCompare(b.id));
  document.getElementById("driver-count").textContent = `(${drivers.length})`;
  driverRows.innerHTML = drivers.map(d => `
    <tr data-test-id="driver-row-${d.id}" class="${d.flagged ? 'flagged-row' : ''} border-b last:border-b-0">
      <td class="py-1 font-mono">${d.id}</td>
      <td>${d.name}</td>
      <td class="text-center font-mono text-xs">(${d.location[0]}, ${d.location[1]})</td>
      <td class="text-center"><span class="status-pill status-${d.status}">${d.status}</span></td>
      <td class="text-center">${d.rating}</td>
      <td class="text-right space-x-1">
        <button data-test-id="btn-dispatch-${d.id}"
                class="bg-blue-600 hover:bg-blue-500 disabled:opacity-30 text-white px-2 py-0.5 rounded text-xs"
                ${d.status === "idle" ? "" : "disabled"}
                onclick="openDispatchModal('${d.id}')">Dispatch</button>
        <button data-test-id="btn-msg-driver-${d.id}"
                class="bg-slate-200 hover:bg-slate-300 px-2 py-0.5 rounded text-xs"
                onclick="openMessageModal('driver', '${d.id}')">Msg</button>
      </td>
    </tr>
  `).join("");

  // Rider queue
  const riderRows = document.getElementById("rider-rows");
  const riders = Object.values(state.riders).sort((a, b) => a.id.localeCompare(b.id));
  document.getElementById("rider-count").textContent = `(${riders.length})`;
  riderRows.innerHTML = riders.map(r => `
    <tr data-test-id="rider-row-${r.id}" class="${r.priority ? 'priority-rider' : ''} border-b last:border-b-0">
      <td class="py-1 font-mono">${r.id}${r.priority ? ' <span class="text-red-600 text-xs">PRIORITY</span>' : ''}</td>
      <td>
        <div>${r.pickup_label}</div>
        <div class="text-xs text-slate-500 font-mono">(${r.pickup[0]}, ${r.pickup[1]})</div>
      </td>
      <td class="text-center text-sm">${r.waited_minutes}m</td>
      <td class="text-center"><span class="status-pill status-${r.status}">${r.status}</span></td>
      <td class="text-right">
        <button data-test-id="btn-msg-rider-${r.id}"
                class="bg-slate-200 hover:bg-slate-300 px-2 py-0.5 rounded text-xs"
                onclick="openMessageModal('rider', '${r.id}')">Msg</button>
      </td>
    </tr>
  `).join("");

  // Fraud panel
  const fraudRows = document.getElementById("fraud-rows");
  const fraudEmpty = document.getElementById("fraud-empty");
  const alerts = Object.values(state.fraud_alerts);
  if (alerts.length === 0) {
    fraudEmpty.classList.remove("hidden");
    fraudRows.innerHTML = "";
  } else {
    fraudEmpty.classList.add("hidden");
    fraudRows.innerHTML = alerts.map(fa => `
      <div data-test-id="fraud-row-${fa.account_id}"
           class="border rounded p-2 ${fa.severity === 'high' ? 'border-red-400 bg-red-50' : 'border-slate-300'}">
        <div class="flex items-center justify-between">
          <div>
            <span class="font-mono text-sm">acct ${fa.account_id}</span>
            <span class="severity-${fa.severity} text-xs uppercase font-bold ml-1">${fa.severity}</span>
          </div>
          <button data-test-id="btn-block-${fa.account_id}"
                  class="bg-red-600 hover:bg-red-500 text-white px-2 py-0.5 rounded text-xs"
                  onclick="openBlockModal(${fa.account_id})">Block</button>
        </div>
        <div class="text-xs text-slate-700 mt-1"><b>Pattern:</b> ${fa.pattern}</div>
        <ul class="text-xs text-slate-600 mt-1 list-disc pl-5">
          ${fa.evidence.map(e => `<li>${escapeHtml(e)}</li>`).join("")}
        </ul>
      </div>
    `).join("");
  }

  // Trips
  document.getElementById("trips-list").innerHTML = Object.values(state.trips).length
    ? Object.values(state.trips).map(t => `
        <li data-test-id="trip-${t.id}" class="font-mono">
          ${t.id}: ${t.driver_id} → ${t.rider_id} (eta ${t.eta_minutes}m)
        </li>`).join("")
    : '<li class="text-slate-400 italic">none</li>';

  // Messages
  document.getElementById("messages-list").innerHTML = state.messages.length
    ? state.messages.map(m => `
        <li data-test-id="message-${m.id}">
          <span class="font-mono text-xs">[step ${m.sent_at_step}]</span>
          <b>→ ${m.recipient_kind} ${m.recipient_id}:</b> ${escapeHtml(m.body)}
        </li>`).join("")
    : '<li class="text-slate-400 italic">none</li>';

  // Audit
  document.getElementById("audit-list").innerHTML = state.audit_log.length
    ? state.audit_log.map(e => `
        <li data-test-id="audit-${e.step}-${e.action}">
          <span class="font-mono text-xs">[${e.step}]</span>
          <b>${e.action}</b> ${e.target_id}: ${escapeHtml(e.reason)}
        </li>`).join("")
    : '<li class="text-slate-400 italic">none</li>';

  // Submit button state
  const submitBtn = document.getElementById("submit-btn");
  submitBtn.disabled = state.finished;
  if (state.finished) submitBtn.textContent = "✅ Submitted";
}


// --------------------------------------------------------------------------- //
// Modals
// --------------------------------------------------------------------------- //

let pendingApply = null;             // function to call on confirm

function openModal(title, bodyHtml, onConfirm) {
  document.getElementById("modal-title").textContent = title;
  document.getElementById("modal-body").innerHTML = bodyHtml;
  document.getElementById("modal-root").classList.remove("hidden");
  pendingApply = onConfirm;
}

function closeModal() {
  document.getElementById("modal-root").classList.add("hidden");
  pendingApply = null;
}

document.getElementById("modal-cancel").onclick = closeModal;
document.getElementById("modal-confirm").onclick = async () => {
  if (pendingApply) {
    const fn = pendingApply;
    pendingApply = null;
    document.getElementById("modal-root").classList.add("hidden");
    await fn();
  }
};


function openDispatchModal(driverId) {
  // Riders that are still waiting:
  const state = window.__lastState;
  const waitingRiders = Object.values(state.riders).filter(r => r.status === "waiting");
  if (waitingRiders.length === 0) {
    alert("No waiting riders to dispatch to.");
    return;
  }

  const opts = waitingRiders.map(r =>
    `<option value="${r.id}">${r.id} — ${r.pickup_label}${r.priority ? ' (PRIORITY)' : ''}</option>`
  ).join("");

  openModal(
    `Dispatch ${driverId}`,
    `
      <p class="text-sm text-slate-600">Select a rider to assign to ${driverId}:</p>
      <select id="dispatch-rider-select" data-test-id="select-dispatch-rider"
              class="w-full border rounded px-2 py-1">${opts}</select>
    `,
    async () => {
      const riderId = document.getElementById("dispatch-rider-select").value;
      const r = await API.apply("dispatch", { driver_id: driverId, rider_id: riderId });
      if (!r.ok) showError(r.result.error || "dispatch failed");
      render(r.state); window.__lastState = r.state;
    },
  );
}


function openMessageModal(kind, recipientId) {
  openModal(
    `Send message to ${kind} ${recipientId}`,
    `
      <textarea id="message-body" data-test-id="textarea-message-body"
                rows="4" placeholder="Type your message..."
                class="w-full border rounded px-2 py-1"></textarea>
    `,
    async () => {
      const body = document.getElementById("message-body").value;
      const r = await API.apply("send_message", {
        recipient_id: recipientId, recipient_kind: kind, body,
      });
      if (!r.ok) showError(r.result.error || "send failed");
      render(r.state); window.__lastState = r.state;
    },
  );
}


function openBlockModal(accountId) {
  openModal(
    `Block account ${accountId}`,
    `
      <p class="text-sm text-slate-600">Provide a reason — this will be added to the audit log.</p>
      <input id="block-reason" data-test-id="input-block-reason"
             placeholder="e.g. fake_pickups_confirmed"
             class="w-full border rounded px-2 py-1" />
    `,
    async () => {
      const reason = document.getElementById("block-reason").value;
      const r = await API.apply("block_account", { account_id: accountId, reason });
      if (!r.ok) showError(r.result.error || "block failed");
      render(r.state); window.__lastState = r.state;
    },
  );
}


// --------------------------------------------------------------------------- //
// Submit / verify
// --------------------------------------------------------------------------- //

document.getElementById("submit-btn").onclick = async () => {
  const r1 = await API.apply("finish", {});
  render(r1.state); window.__lastState = r1.state;

  const v = await API.verify();
  const result = v.verifier;
  const el = document.getElementById("verifier-result");

  const color = result.success ? "text-emerald-700" : "text-red-700";
  const headline = result.success
    ? `✅ SUCCESS — score ${result.score.toFixed(2)}`
    : `❌ ${result.error_category.toUpperCase()} — score ${result.score.toFixed(2)}`;

  el.innerHTML = `
    <div class="${color} font-bold">${headline}</div>
    <ul class="mt-1 ml-4 list-disc text-slate-700">
      ${(result.details || []).map(d => `<li>${escapeHtml(d)}</li>`).join("")}
    </ul>
  `;
};


// --------------------------------------------------------------------------- //
// Reset
// --------------------------------------------------------------------------- //

document.getElementById("reset-btn").onclick = async () => {
  const taskId = document.getElementById("task-picker").value;
  const seed = parseInt(document.getElementById("seed-input").value || "0", 10);
  const r = await API.reset(taskId, seed);
  render(r.state); window.__lastState = r.state;
  document.getElementById("verifier-result").innerHTML = "";
};


// --------------------------------------------------------------------------- //
// Bootstrap
// --------------------------------------------------------------------------- //

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, c => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]
  ));
}

function showError(msg) {
  const el = document.getElementById("verifier-result");
  el.innerHTML = `<div class="text-red-700">⚠ ${escapeHtml(msg)}</div>`;
}

(async function init() {
  const tasksResp = await API.tasks();
  const sel = document.getElementById("task-picker");
  sel.innerHTML = tasksResp.tasks.map(t => `<option value="${t}">${t}</option>`).join("");
  // Default to the first task with seed 0
  await document.getElementById("reset-btn").click();
})();
